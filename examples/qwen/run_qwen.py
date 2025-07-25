# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
from functools import partial
from typing import Union
import torch
import torch._dynamo
import datasets
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec as get_gpt_layer_with_transformer_engine_spec_tq
from megatron.core.models.gpt import GPTModel
from megatron.core.num_microbatches_calculator import get_num_microbatches

# from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler

from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.data import build_pretrain_dataset_from_original
from megatron_patch.data.utils import get_batch_on_this_tp_rank_original, get_batch_on_this_tp_rank_idxmap_sft, get_batch_on_this_tp_rank_online_packing
from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
from megatron_patch.model.qwen2.layer_specs import get_gpt_layer_local_spec

# from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron_patch.model.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron_patch.training import pretrain

torch._dynamo.config.suppress_errors = True


def model_provider(
    pre_process=True, post_process=True
) -> GPTModel:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.core.models.gpt.GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)

    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    if args.transformer_impl == "transformer_engine":
        if args.te_spec_version == "base":
            print_rank_0("build model in TE")
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.te_spec_version== "tqlm":
            # this spec fuses input_layernorm and linear_qkv
            print_rank_0("build model in TE_tq")
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_tq(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
    else:
        print_rank_0("build model in mcore")
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
    )
    return model

def build_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    if args.online_packing:
        from megatron_patch.data.loader import DataLoaderWithDataConcatingIterator
        from megatron_patch.data.concator import SFTConcatFn, CPTConcatFn
        from megatron_patch.data.collator import DataCollatorForSFTRawText, DataCollatorForCPTRawText
        data_collator = None
        data_collator = None
        tokenizer = get_tokenizer()
        if args.train_mode == 'pretrain':
            data_collator = DataCollatorForCPTRawText(tokenizer=tokenizer.tokenizer)
            data_concator = CPTConcatFn(args.micro_batch_size, args.seq_length)
        elif args.train_mode == 'finetune':
            data_collator = DataCollatorForSFTRawText(tokenizer=tokenizer.tokenizer, max_padding_length=args.seq_length)
            data_concator = SFTConcatFn(args.micro_batch_size, args.seq_length, tokenizer.pad_token_id)
        # DataLoaderWithDataConcatingIterator only support num_workers>0
        dataloader = DataLoaderWithDataConcatingIterator(dataset=dataset,
                                                        batch_sampler=batch_sampler,
                                                        num_workers=1,
                                                        pin_memory=True,
                                                        persistent_workers=True,
                                                        collate_fn=data_collator,
                                                        concat_fn=data_concator
                                                        )
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )
    return dataloader

def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None
    args = get_args()
    if args.online_packing:
        batch = get_batch_on_this_tp_rank_online_packing(data_iterator)
        num_seqs = batch.pop('num_seqs')
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)
        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    if "-Raw" in args.dataset:
        if args.train_mode == "pretrain":
            raise ValueError('The LLama-SFT-Raw dataset should only be used for finetuning!')
        batch = get_batch_on_this_tp_rank_original(data_iterator)
        num_seqs = batch.pop('num_seqs')
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)
        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif "-Idxmap" in args.dataset:
        # get batches based on the TP rank you are on
        if args.train_mode == "pretrain":
            batch = get_batch_on_this_tp_rank(data_iterator)
        else:
            batch = get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=False)

        packed_seq_params = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                # mbs = 1
                position_ids = position_ids[0] # shape: [seq_length]
                start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                if seqlens.shape != torch.Size([0]):
                    cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
                cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                cu_seqlens[-1] = position_ids.shape[0]
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd'
                )

        if packed_seq_params is not None and args.context_parallel_size > 1:
            raise ValueError('Sequence Packing is not supported when CP>1 !')
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs', None)
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")


def per_batch_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat(
            [torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)]
        )
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}


def per_token_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask).view(1)
    num_tokens = loss_mask.sum().view(1)
    losses = torch.cat(
        [loss, num_tokens]
    )
    torch.distributed.all_reduce(losses, group=mpu.get_data_parallel_group(with_context_parallel=True))
    loss = losses[0]
    num_tokens = losses[1]
    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )
    return loss, num_tokens.to(torch.int), {"lm loss": (loss.item(), num_tokens.item())}

def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    if args.calculate_per_token_loss:
        loss_func = per_token_loss_func
    else:
        loss_func = per_batch_loss_func
    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    print_rank_0("> building train, validation, and test datasets for GPT ...")
    if args.online_packing:
        ds = datasets.load_dataset('text', data_files=args.train_data_path or args.data_path, split='train')
        train_ds, valid_ds, test_ds = ds, ds, ds
    elif "-Raw" in args.dataset:
        train_ds, valid_ds, test_ds = build_pretrain_dataset_from_original(args.dataset)
    else:
        config = core_gpt_dataset_config_from_args(args)

        # NOTE: in preparation scripts, the sequence is collect into (seq, labels)
        # therefore we need to double the seqlen
        if args.train_mode != "pretrain":
            config.sequence_length = config.sequence_length * 2

        if config.mock:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds



if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
        dataloader_provider=build_data_loader
    )
