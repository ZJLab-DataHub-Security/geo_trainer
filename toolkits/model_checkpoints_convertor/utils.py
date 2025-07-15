import os, shutil, gc
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from typing import Union, Tuple, List
import numpy as np
from collections.abc import Mapping, Sequence

@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device.
    """
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem

def save_state_dict(args, model_chunks, checkpoint_name, has_vpp: bool=False, save_args: bool=True):
    """
    Save some model chunks to a megatron checkpoint file
    """
    state_dict = {}
    if save_args:
        state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0
    if not has_vpp:
        state_dict['model'] = model_chunks[0]
    else:
        for vpp_id in range(len(model_chunks)):
            state_dict[f"model{vpp_id}"] = model_chunks[vpp_id]
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)
    del state_dict
    gc.collect()

def save_hfmodel(args, model, max_shard_size='10GB'):
    output_state_dict = model
    if not isinstance(model, dict):
        output_state_dict = model.state_dict()
    save_safetensors = (not USE_TRANSFORMERS_SAVE) or args.save_safetensors
    os.makedirs(args.save, exist_ok=True)

    # NOTE: remove all old index files
    if os.path.exists(os.path.join(args.save, SAFE_WEIGHTS_INDEX_NAME)):
        os.remove(os.path.join(args.save, SAFE_WEIGHTS_INDEX_NAME))
    if os.path.exists(os.path.join(args.save, WEIGHTS_INDEX_NAME)):
        os.remove(os.path.join(args.save, WEIGHTS_INDEX_NAME))

    index = None
    if USE_TRANSFORMERS_SAVE:
        weight_file = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
        index_file = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size, weights_name=weight_file)
    else:
        if not args.save_safetensors:
            logging.warning("Since Transformer v4.47.0, the HF ckpt can only be saved with safetensors in the scripts.")
        weight_file = SAFETENSORS_WEIGHTS_FILE_PATTERN
        index_file = SAFETENSORS_INDEX_FILE
        state_dict_split = split_torch_state_dict_into_shards(output_state_dict, max_shard_size=max_shard_size, filename_pattern=weight_file)
        shards = {}
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shards[filename] = {tensor: output_state_dict[tensor] for tensor in tensors}
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

    for shard_file, shard in shards.items():
        target_file = os.path.join(args.save, shard_file)
        print(f'huggingface model is save to {target_file}')
        if save_safetensors:
            save_file(clone_state_dict(shard), target_file, metadata={"format": "pt"})
        else:
            torch.save(clone_state_dict(shard), target_file)

    if index is not None:
        save_index_file = os.path.join(args.save, index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

def replace_tokenizer_cfg(
    save_dir: Union[str, os.PathLike],
    task_type: str,
    model_class: str,
) -> None:
    if not dist.get_rank() == 0:
        return

    # only replace tokenizer config when training sft model
    if task_type != "sft":
        return

    tokenizer_dir = get_default_tokenizer_dir(task_type, model_class)
    if tokenizer_dir is None:
        return

    tokenizer_files = [
        "generation_config.json",
        "tokenizer_config.json",
    ]

    for file_name in tokenizer_files:
        src_path = os.path.join(tokenizer_dir, file_name)
        dst_path = os.path.join(save_dir, file_name)
        if os.path.exists(src_path):
            print("file {} exists, copy it to {}".format(file_name, dst_path))
            shutil.copy(src_path, dst_path)
        else:
            print("file {} doesn't exists".format(file_name))


def copy_model_cfg(
    src_dir: str,
    dst_dir: Union[str, os.PathLike],
) -> None:
    if not dist.get_rank() == 0:
        return

    if src_dir == dst_dir:
        return

    os.makedirs(dst_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(src_dir)
    config.save_pretrained(dst_dir)
    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    tokenizer.save_pretrained(dst_dir)
    shutil.copy(os.path.join(src_dir, "generation_config.json"), dst_dir)
