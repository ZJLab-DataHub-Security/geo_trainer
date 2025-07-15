"""
Wrapper class for deepspeed trainer
"""

import yaml
import json
import os
import sys
import logging
import argparse
import importlib

from typing import Dict, List, Union
from datasets import dataset_dict

from torch.utils.data import ConcatDataset, Dataset


path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
megatron_path = os.path.join("/workspace", "Megatron-LM-0.11.0")
megatron_patch_path = os.path.join(path_dir, "megatron_patch")
sys.path.append(megatron_patch_path)
sys.path.append(megatron_path)
sys.path.append(path_dir)

from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

from toolkits.model_checkpoints_convertor.hf2mcore_qwen2_dense_and_moe_gqa import (
    add_extra_args,
)

from toolkits.model_checkpoints_convertor.utils import copy_model_cfg, replace_tokenizer_cfg
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


DatasetType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]
PathType = Union[str, os.PathLike]
logger = logging.getLogger()


def import_model_functions(model_class, func_name):
    if model_class == "qwen2":
        module_path = "examples.qwen.run_qwen"
    elif model_class == "llama3":
        module_path = "examples.llama.run_llama"
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    module = importlib.import_module(module_path)

    function = getattr(module, func_name)

    return function


def import_convert_functions(model_class, func_name):
    if model_class == "qwen2":
        module_path = (
            "toolkits.model_checkpoints_convertor.hf2mcore_qwen2_dense_and_moe_gqa"
        )
    elif model_class == "llama3":
        module_path = "toolkits.model_checkpoints_convertor.hf2mcore"
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    module = importlib.import_module(module_path)

    function = getattr(module, func_name)

    return function


def tqlm_args(parser):
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--hf_model_dir", type=str)
    parser.add_argument("--mg_model_dir", type=str)
    parser.add_argument("--hf_ckpt_dir", type=str, default="none")
    parser.add_argument(
        "--convert_option",
        choices=["hf2mg", "mg2hf"],
        default="hf2mg",
        help="type of model convert option",
    )
    parser.add_argument(
        "--task_type",
        choices=["cpt", "sft"],
        default="sft",
        help="type of task",
    )
    parser.add_argument(
        "-m",
        "--model_class",
        type=str,
        choices=["qwen2", "llama3"],
        help="tokenizer model",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--use_tqlm_spec", action="store_true")
    return parser


def tqlm_patch_args(parser):
    parser = tqlm_args(parser)
    parser = add_extra_args(parser)
    return parser


def convert_hf_to_megatron():
    args = get_args()
    print(f"args:{args}")

    # save model/tokenizer cfg at first
    copy_model_cfg(args.load, args.save)
    dtype = "bfloat16" if args.bf16 else "float16"

    hf_model = AutoModelForCausalLM.from_pretrained(args.load, torch_dtype=dtype)
    hf_model.config.save_pretrained(args.save)

    model_provider = import_model_functions(args.model_class, "model_provider")
    convert_checkpoint_from_transformers_to_megatron = import_convert_functions(
        args.model_class, "convert_checkpoint_from_transformers_to_megatron"
    )
    save_mgmodel = import_convert_functions(args.model_class, "save_mgmodel")

    mg_model = model_provider()
    convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    save_mgmodel(mg_model, args)

    hf_model.config.save_pretrained(
        args.save
    )  # save_mgmodel will copy config.json from load dir, but the dtype migth changed in hf_model


def convert_megatron_to_hf():
    args = get_args()

    # save model/tokenizer cfg at first
    copy_model_cfg(args.load, args.save)

    config = AutoConfig.from_pretrained(args.save)
    if args.hf_ckpt_path == "none":
        log.info("no hf_ckpt_dir set, build hf_model from config")
        args.hf_ckpt_path = args.save
        hf_model = AutoModelForCausalLM.from_config(config)
    else:
        log.info(
            f"hf_ckpt_dir is [{args.hf_ckpt_path}], load hf_model from hf_ckpt_dir"
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_ckpt_path, torch_dtype=config.torch_dtype
        )

    load_megatron_model = import_convert_functions(
        args.model_class, "load_megatron_model"
    )
    if args.model_class == "mistral":
        load_megatron_model = import_convert_functions(
            args.model_class, "load_megatron_model_external"
        )

    convert_checkpoint_from_megatron_to_transformers = import_convert_functions(
        args.model_class, "convert_checkpoint_from_megatron_to_transformers"
    )
    save_hfmodel = import_convert_functions(args.model_class, "save_hfmodel")

    mg_model = load_megatron_model(args)
    convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
    save_hfmodel(args, hf_model)
    # replace_tokenizer_cfg(args.save, args.task_type, args.model_class)


def to_megatron_args(cfg_path, load, save, model_class, tp, pp, ep):

    kw_args = {
        "attention-dropout": 0.0,
        "disable-bias-linear": True,
        "hidden-dropout": 0.0,
        "micro-batch-size": 1,
        "no-async-tensor-model-parallel-allreduce": True,
        "no-bias-swiglu-fusion": True,
        "no-rope-fusion": True,
        "normalization": "RMSNorm",
        "save-interval": 1,
        "seq-length": 1,
        "swiglu": True,
        "use-mcore-models": True,
        "use-rotary-position-embeddings": True,
        "load": load,
        "save": save,
        "hidden-dropout": 0.0,
        "save-safetensors": True,
    }

    # load config
    # cfg = Config.from_file(cfg_path)
    # for k, v in cfg.items():
    #     cfg[k] = update_var_from_env(k, v)

    # copy model relate params to megatron args
    hf_config = AutoConfig.from_pretrained(load, use_fast=False)
    kw_args["ffn-hidden-size"] = hf_config.intermediate_size
    kw_args["hidden-size"] = hf_config.hidden_size
    kw_args["num-attention-heads"] = hf_config.num_attention_heads
    kw_args["num-layers"] = hf_config.num_hidden_layers
    kw_args["num-query-groups"] = hf_config.num_key_value_heads
    kw_args["norm-epsilon"] = hf_config.rms_norm_eps
    kw_args["max-position-embeddings"] = hf_config.max_position_embeddings
    kw_args["untie-embeddings-and-output-weights"] = not bool(
        hf_config.tie_word_embeddings
    )
    kw_args["rotary-base"] = int(hf_config.rope_theta)

    # copy model related params from mg model config
    curr_path = os.path.abspath(__file__)
    model_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(curr_path))),
        "configs",
        "megatron",
        "model_config.json",
    )
    with open(model_cfg_path, "r") as file:
        model_cfg = json.load(file)
        mg_model_cfg = model_cfg[model_class]
        for k, v in mg_model_cfg.items():
            if k == "base-vocab-size":
                kw_args["extra-vocab-size"] = hf_config.vocab_size - v
            else:
                kw_args[k] = v

    kw_args["bf16"] = True
    kw_args["target-expert-model-parallel-size"] = ep
    kw_args["target-tensor-model-parallel-size"] = tp
    kw_args["target-pipeline-model-parallel-size"] = pp
    kw_args["use-cpu-initialization"] = True
    kw_args["no-load-optim"] = True
    kw_args["no-load-rng"] = True

    # change to arg list
    args_list = []
    for k, v in kw_args.items():
        if isinstance(v, bool):
            if v:
                args_list.append(f"--{k}")
        elif v is not None:
            args_list.append(f"--{k}={v}")
    return args_list


if __name__ == "__main__":
    parser = tqlm_args(argparse.ArgumentParser())
    args = parser.parse_args()
    print(args)

    if args.convert_option == "hf2mg":
        args_list = to_megatron_args(
            args.config_path,
            args.hf_model_dir,
            args.mg_model_dir,
            args.model_class,
            args.tp,
            args.pp,
            args.ep,
        )
        args_list.append("--model_class={}".format(args.model_class))
    elif args.convert_option == "mg2hf":
        args_list = to_megatron_args(
            args.config_path,
            args.mg_model_dir,
            args.hf_model_dir,
            args.model_class,
            args.tp,
            args.pp,
            args.ep,
        )
        args_list.append("--hf-ckpt-path={}".format(args.hf_ckpt_dir))
        args_list.append("--model_class={}".format(args.model_class))
    print(args_list)

    sys.argv.extend(args_list)
    initialize_megatron(extra_args_provider=tqlm_patch_args)
    args = get_args()
    if args.use_tqlm_spec:
        args.te_spec_version="tqlm"

    if args.convert_option == "hf2mg":
        convert_hf_to_megatron()
    elif args.convert_option == "mg2hf":
        convert_megatron_to_hf()
    else:
        raise Exception(
            "convert option only support [hf2mg, mg2hf], but got {}".format(
                args.convert_option
            )
        )
