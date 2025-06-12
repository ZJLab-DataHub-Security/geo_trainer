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
import io
import copy
import json
import torch
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm

from megatron_patch.tokenizer import get_tokenizer

"""
PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task,'
             ' paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}'
             '\n\n### Input:\n{input}\n\n### Response:'),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }

PROMPT_DICT = {
    'prompt_input': ('[INST]{instruction} {input}\n[/INST]\n'),
    'prompt_no_input':('[INST]{instruction}\n[/INST]\n'),
}

PROMPT_DICT = {
    'prompt_input': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction} {input}<|im_end|>\n<|im_start|>assistant\n',
    'prompt_no_input':'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
}
"""

PROMPT_DICT = {
    'prompt_input': ('{instruction} {input}'),
    'prompt_no_input': ('{instruction}'),
}


class LLamaRawDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""

    def __init__(self, path, max_padding_length, split='train'):
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        if "-Pretrain" in args.dataset:
            self.max_padding_length = max_padding_length + 1
        else:
            self.max_padding_length = max_padding_length

        list_data_dict = load_dataset(
            'json',
            data_files=path[0],
            split=split,
        )

        train_dataset = list_data_dict.map(
            self.preprocess,
            batched=True,
            batch_size=3000,
            num_proc=16,
            remove_columns=list_data_dict.column_names,
            load_from_cache_file=False,
            desc="Running Encoding"
        )

        self.input_ids = np.array(train_dataset['input_ids'])
        self.labels = np.array(train_dataset['labels'])
        self.samples = []

        for inputs, labels in tqdm(zip(self.input_ids, self.labels)):
            if self.tokenizer.eos_token_id not in inputs: continue
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """

        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

        sources = []
        if 'input' not in examples:
            if 'instruction' in examples:
                for instruction in examples['instruction']:
                    sources.append(prompt_no_input.format_map({"instruction": instruction}))
            elif 'query' in examples:
                for query in examples['query']:
                    sources.append(prompt_no_input.format_map({"instruction": query}))
        else:
            if 'instruction' in examples:
                for instruction, minput in zip(examples['instruction'], examples['input']):
                    sources.append(prompt_input.format_map({"instruction": instruction, "input": minput}))
            elif 'query' in examples:
                for query, minput in zip(examples['query'], examples['input']):
                    sources.append(prompt_input.format_map({"instruction": query, "input": minput}))

        if 'output' in examples:
            key = 'output'
        elif 'content' in examples:
            key = 'content'
        elif 'response' in examples:
            key = 'response'

        targets = [
            example + self.tokenizer.eos_token
            for example in examples[key]
        ]

        examples_raw = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self.tokenize(strings, self.tokenizer)
            for strings in (examples_raw, sources)
        ]
        input_ids = examples_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels,
                                     sources_tokenized['input_ids_lens']):
            label[:source_len] = self.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """
        Tokenize a list of strings.
        Args:
            strings (List[str]): a list of strings to be tokenized
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the tokenized strings
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
                add_special_tokens=False
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample

class LLamaSFTDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""

    def __init__(self, path, max_padding_length, split='train'):
        self.tokenizer = get_tokenizer()
        assert hasattr(self.tokenizer, 'apply_chat_template'), \
            "The LLama-SFT-Raw Dataset is valid for tokenizers with chat template, please provide a template."
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        self.max_padding_length = max_padding_length

        list_data_dict = load_dataset(
            'json',
            data_files=path[0],
            split=split,
        )

        train_dataset = list_data_dict.map(
            self.preprocess,
            batched=True,
            batch_size=3000,
            num_proc=16,
            remove_columns=list_data_dict.column_names,
            load_from_cache_file=False,
            desc="Running Encoding"
        )

        self.input_ids = np.array(train_dataset['input_ids'])
        self.labels = np.array(train_dataset['labels'])
        self.samples = []

        for inputs, labels in tqdm(zip(self.input_ids, self.labels)):
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """


        datas = []
        if 'instruction' in examples:
            datas = [ examples['instruction']]
        elif 'query' in examples:
            datas = [ examples['query']]
        else:
            raise ValueError('Cannot find key instruction or query!')

        if 'input' in examples:
            datas.append(examples['input'])

        if 'output' in examples:
            datas.append(examples['output'])
        elif 'content' in examples:
            datas.append(examples['content'])
        elif 'response' in examples:
            datas.append(examples['response'])
        else:
            raise ValueError('Cannot find output key `output`, `content` or `response`!')
        
        input_ids = []
        labels = []
        for data in zip(*datas):
            text = [
                {'role': 'user', 'content': ''.join(data[:-1])},
                {'role': 'assistant', 'content': data[-1]}
            ]
            source = self.tokenizer.apply_chat_template(text[:-1])
            full = self.tokenizer.apply_chat_template(text)

            # for t1, t2 in zip(source, full):
            #     assert t1 == t2, "The user input_ids are not a prefix of the full input_ids! Please check the template."
            
            if len(source) >= self.max_padding_length:
                continue

            if len(full) >= self.max_padding_length:
                full = full[:self.max_padding_length]
            
            if self.max_padding_length > len(full):
                full = full + [self.IGNORE_INDEX] * (self.max_padding_length - len(full))
            
            # NOTE: in get_batch_on_this_tp_rank_original, tokens use [:-1] and labels use [1:]
            # we add an extra token to use the old api
            # TODO: update get_batch_on_this_tp_rank_original and replace the following line with
            # label = [self.IGNORE_INDEX] * (len(source) - 1) + full[len(input_ids):] + [self.IGNORE_INDEX]
            full = full + [self.IGNORE_INDEX]
            label = [self.IGNORE_INDEX] * len(source) + full[len(source):]

            input_ids.append(full)
            labels.append(label)

        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """
        This API is only for consistency and not used in the SFT dataset.
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
                add_special_tokens=False
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample