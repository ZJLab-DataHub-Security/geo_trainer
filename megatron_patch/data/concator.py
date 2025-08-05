from dataclasses import dataclass
from typing import List, Dict, Any

# ensure all the keys in data have the same batch_size and seqlen
def data_validation(data: Dict[str, List[Any]]):
    keys = list(data.keys())
    pkey = keys[0]
    keys = keys[1:]
    pseq_list = []
    for batch in data[pkey]:
        pseq_list.append(len(batch))
    for key in keys:
        batches = data[key]
        assert len(batches) == len(pseq_list), f"different batchsize: {pkey}: {len(pseq_list)}, {key}: {len(batches)}"
        seq_list = [len(batch) for batch in batches]
        assert not any([x-y for x, y in zip(pseq_list, seq_list)]), f"different seqlen: {pkey}: {pseq_list}, {key}: {seq_list}"

@dataclass
class CPTConcatFn:
    batch_size: int
    max_seqlen: int

    def __call__(self, data_concating: Dict[str, List[Any]], data: Dict[str, List[Any]]):
        # data_validation(data)
        result = {}
        for key in data:
            assert key in ("input_ids", "labels")
            value = data[key]
            bs = len(value)
            for b in range(bs):
                if len(data_concating[key]) == 0:
                    data_concating[key].append([])
                data_concating[key][-1] += value[b]
                while(len(data_concating[key][-1])) > self.max_seqlen:
                    prev_seqs = data_concating[key][-1]
                    next_seqs = prev_seqs[self.max_seqlen:]
                    prev_seqs = prev_seqs[:self.max_seqlen]
                    data_concating[key][-1] = prev_seqs
                    data_concating[key].append(next_seqs)
            concated_bs = len(data_concating[key])
            if concated_bs > self.batch_size:
                result[key] = data_concating[key][:self.batch_size]
                data_concating[key] = data_concating[key][self.batch_size:]
        if len(result) == 0:
            return None
        if 'input_ids' in result:
            result['position_ids'] = [list(range(self.max_seqlen)) for _ in range(len(result['input_ids']))]
        return result

@dataclass
class SFTConcatFn:
    batch_size: int
    max_seqlen: int
    pad_token_id: int

    def __call__(self, data_concating: Dict[str, List[Any]], data: Dict[str, List[Any]]):
        data_validation(data)
        result = {}
        for key in data:
            assert key in ("input_ids", "labels", "position_ids")
            value = data[key]
            bs = len(value)
            for b in range(bs):
                if len(data_concating[key]) == 0:
                    data_concating[key].append([])
                prev_seqs = data_concating[key][-1]
                prev_seqlen = len(prev_seqs)
                seqlen = len(value[b])
                if seqlen > self.max_seqlen:
                    continue
                elif prev_seqlen + seqlen > self.max_seqlen:
                    if key in ("input_ids", "labels"):
                        pad = self.pad_token_id
                    else:
                        pad = 0
                    prev_seqs.extend([pad,]*(self.max_seqlen-len(prev_seqs)))
                    data_concating[key].append(value[b])
                else:
                    prev_seqs.extend(value[b])
            concated_bs = len(data_concating[key])
            if concated_bs > self.batch_size:
                result[key] = data_concating[key][:self.batch_size]
                data_concating[key] = data_concating[key][self.batch_size:]
        if len(result) == 0:
            return None
        return result
