import json
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForSFTRawText:

    tokenizer: PreTrainedTokenizerBase
    max_padding_length: int = 4096

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        labels = []
        position_ids = []
        for feature in features:
            text = json.loads(feature['text'])['messages']
            source = self.tokenizer.apply_chat_template(text[:-1])
            full = self.tokenizer.apply_chat_template(text)
            if len(full) >= self.max_padding_length:
                continue

            full = full + [self.tokenizer.pad_token_id]
            label = [self.tokenizer.pad_token_id] * len(source) + full[len(source):]

            input_ids.append(full[:-1])
            labels.append(label[1:])
            position_ids.append(list(range(len(full)-1)))

        return dict(input_ids=input_ids, labels=labels, position_ids=position_ids)

@dataclass
class DataCollatorForCPTRawText:

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        labels = []
        position_ids = []
        for feature in features:
            text = json.loads(feature['text'])['text']
            tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
            tokens = tokens + [self.tokenizer.pad_token_id]

            input_ids.append(tokens[:-1])
            labels.append(tokens[1:])

        return dict(input_ids=input_ids, labels=labels)
