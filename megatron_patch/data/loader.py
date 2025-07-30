import time
import json
import functools
import datasets
import random
import queue
import threading
import itertools
from typing import Union
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch.utils.data import SequentialSampler, IterDataPipe, MapDataPipe, _utils
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter, _MultiProcessingDataLoaderIter, _DatasetKind, _sharding_worker_init_fn
from torch.utils.data._utils.worker import _IterableDatasetStopIteration

from transformers import AutoTokenizer

def convert_result_list_to_tensor(result):
    for key in result:
        if isinstance(result[key], list):
            result[key] = torch.tensor(result[key], dtype=torch.long)
    return result

def index_loop(index_queues, sampler_iter, num_workers, prefetch_factor,
        worker_queue_idx_cycle, send_idx, task_info, worker_status, done_event):
    while not done_event.is_set():
        worker_queue_idx = next(worker_queue_idx_cycle)
        if not worker_status[worker_queue_idx]:
            continue
        index_queue = index_queues[worker_queue_idx]
        if index_queue.qsize() < prefetch_factor:
            try:
                index=next(sampler_iter)
            except StopIteration:
                index_queue.put((send_idx, _IterableDatasetStopIteration(worker_queue_idx)))
                done_event.set()
                break
            index_queue.put((send_idx, index))
            task_info[send_idx] = (worker_queue_idx,)
            send_idx += 1

def worker_loop(dataset_kind, dataset, index_queue, prefetch_factor, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers, shared_seed, concat_fn):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    from torch.utils.data._utils import signal_handling, HAS_NUMPY, MP_STATUS_CHECK_INTERVAL
    from torch.utils.data._utils.worker import _generate_state, WorkerInfo, ManagerWatchdog, _ResumeIteration

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np
            np.random.seed(np_seed)

        from torch.utils.data import IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        shared_rng = torch.Generator()
        if isinstance(dataset, IterDataPipe):
            assert shared_seed is not None
            shared_rng.manual_seed(shared_seed)
            dataset = apply_random_seed(dataset, shared_rng)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
                                  seed=seed, dataset=dataset)

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader worker process {worker_id}")

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        concating_data = {}
        while watchdog.is_alive():
            if data_queue.qsize() > prefetch_factor:
                continue
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            else:
                idx, index = r
                data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
                if init_exception is not None:
                    data = init_exception
                    init_exception = None
                elif isinstance(index, _IterableDatasetStopIteration):
                    data = _IterableDatasetStopIteration(worker_id)
                    iteration_end = True
                    data_queue.put((idx, data))
                    continue
                else:
                    data = fetcher.fetch(index)
                    try:
                        data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
                    except Exception as e:
                        if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                            data = _IterableDatasetStopIteration(worker_id)
                            # Set `iteration_end`
                            #   (1) to save future `next(...)` calls, and
                            #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                            iteration_end = True
                        else:
                            # It is important that we don't store exc_info in a variable.
                            # `ExceptionWrapper` does the correct thing.
                            # See NOTE [ Python Traceback Reference Cycle Problem ]
                            data = ExceptionWrapper(
                                where=f"in DataLoader worker process {worker_id}")
            if len(concating_data) == 0:
                for key in data:
                    concating_data[key]=[]
            result = concat_fn(concating_data, data)
            if result is not None:
                data_queue.put((idx, result))
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()

class DataConcatingMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        _BaseDataLoaderIter.__init__(self, loader)
        self._concat_fn = loader._concat_fn

        self._prefetch_factor = loader.prefetch_factor

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank)

        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue, self._prefetch_factor,
                      self._worker_result_queue, self._workers_done_event, self._auto_collation,
                      self._collate_fn, self._drop_last, self._base_seed, self._worker_init_fn,
                      i, self._num_workers, self._persistent_workers, self._shared_seed,
                      self._concat_fn))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
        self.index_put_done_event = threading.Event()
        # index_put_thread = threading.Thread(
        #     target=index_loop,
        #     args=(self._index_queues, self._sampler_iter, self._num_workers,
        #         self._prefetch_factor, self._worker_queue_idx_cycle,
        #         self._send_idx, self._task_info, self._workers_status, self.index_put_done_event))
        # index_put_thread.daemon = True
        # index_put_thread.start()
        # self._index_put_thread = index_put_thread
        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
                custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
                current_device = custom_device_mod.current_device()
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      current_device,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue  # type: ignore[assignment]

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        _BaseDataLoaderIter._reset(self, loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.
        self._workers_status = [True for i in range(self._num_workers)]
        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # We resume the prefetching in case it was enabled
        self.index_put_done_event.set()
        if hasattr(self, "_index_put_thread"):
            self._index_put_thread.join()
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration(self._shared_seed))
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        self.index_put_done_event.clear()
        index_put_thread = threading.Thread(
            target=index_loop,
            args=(self._index_queues, self._sampler_iter, self._num_workers,
                self._prefetch_factor, self._worker_queue_idx_cycle,
                self._send_idx, self._task_info, self._workers_status, self.index_put_done_event))
        index_put_thread.daemon = True
        index_put_thread.start()
        self._index_put_thread = index_put_thread

    def _next_data(self):
        while True:
            idx, data = self._get_data()
            # Check for _IterableDatasetStopIteration
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                if self._persistent_workers:
                    self._workers_status[data.worker_id] = False
                else:
                    self._mark_worker_as_unavailable(data.worker_id)
                    self._shutdown_workers()
                raise StopIteration

            return convert_result_list_to_tensor(data)

    def _shutdown_workers(self):
        super()._shutdown_workers()
        self.index_put_done_event.set()
        if hasattr(self, "_index_put_thread"):
            self._index_put_thread.join()

class DataLoaderWithDataConcatingIterator(DataLoader):
    def __init__(self, **kwargs):
        if 'concat_fn' in kwargs:
            self._concat_fn = kwargs.pop('concat_fn')
        super().__init__(**kwargs)
        assert self.num_workers > 0

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        return DataConcatingMultiProcessingDataLoaderIter(self)
