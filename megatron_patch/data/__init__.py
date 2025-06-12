# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
try:
    from megatron import get_args
except:
    from megatron.training import get_args

from megatron_patch.tokenizer import build_tokenizer
from .llama import LLamaRawDataset, LLamaSFTDataset

def build_evaluation_dataset(dataset):

    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT' or dataset == 'LLama-Pretrain-Raw':
        val_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)
        return val_dataset
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

def build_finetune_dataset(dataset):

    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT':
        train_dataset = LLamaRawDataset(args.train_data_path, args.max_padding_length)
        valid_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)

        return train_dataset, valid_dataset
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

def build_pretrain_dataset_from_original(dataset):

    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-Pretrain-Raw':
        train_dataset = LLamaRawDataset(args.train_data_path, args.max_padding_length)
        #valid_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)
        #test_dataset = LLamaRawDataset(args.test_data_path, args.max_padding_length)
        # customize your validation and test dataset here

        return train_dataset, train_dataset, train_dataset


    elif dataset == 'LLama-SFT-Raw':
        train_dataset = LLamaSFTDataset(args.train_data_path, args.max_padding_length)

        return train_dataset, train_dataset, train_dataset
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))