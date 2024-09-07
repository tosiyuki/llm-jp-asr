import copy
import json
import os
import librosa

from dataclasses import dataclass
from typing import Dict

from typing import Sequence

import torch
import transformers

from PIL import Image
from torch.utils.data import Dataset

from llm_asr import conversation as conversation_lib
from llm_asr.constants import DEFAULT_AUDIO_TOKEN, IGNORE_INDEX, AUDIO_TOKEN_INDEX
from llm_asr.train.arguments_dataclass import DataArguments


def tokenizer_audio_token(prompt, tokenizer, AUDIO_TOKEN_INDEX=AUDIO_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<audio>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [AUDIO_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning.
    data_dict example:
        {
            audio: ['fine_tune/022CmvRb8aM5pKF_1.mp3', 'fine_tune/022CmvRb8aM5pKF_2.mp3', 'fine_tune/022CmvRb8aM5pKF_3.mp3', 'fine_tune/022CmvRb8aM5pKF_4.mp3', 'fine_tune/022CmvRb8aM5pKF_5.mp3']
            sentence: ['幸ありて', '普通の人ならたいして問題にすまいこのことが、', '私の心を暗くした。', 'もし耳がこのまま聞こえなくなったら、', 'その時は自殺するよりほかはないと思った。']    
        }
    """
    def __init__(
        self, 
        data_dict: dict[str, list[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        feature_extractor,
    ):
        super(LazySupervisedDataset, self).__init__()

        print("Formatting inputs...Skip in lazy mode")
        self.data_dict = data_dict
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data_dict["audio"])
    
    """
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    """

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        audio = self.data_dict["audio"][i]
        prompt = ""
        sentence = self.data_dict["sentence"][i]

        # audioの前処理
        #audio_array, sampling_rate = sf.read(audio, samplerate=16000)
        audio_array, sampling_rate = librosa.load(audio, sr=16000)
        input_features = self.feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]

        # sentenceの前処理
        #labels = self.tokenizer(sentence).input_ids

        # create input ids
        default_prompt = f"ユーザー: {DEFAULT_AUDIO_TOKEN}\n{prompt}\nシステム: "
        input_ids = tokenizer_audio_token(f"{default_prompt}{sentence}", self.tokenizer, return_tensors='pt')
        targets = copy.deepcopy(input_ids)
        targets[:len(tokenizer_audio_token(default_prompt, self.tokenizer))-2] = IGNORE_INDEX
        
        data_dict = dict(
            input_ids=input_ids,
            labels=targets,
        )

        data_dict['audios'] = input_features

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    audio_processor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'audios' in instances[0]:
            input_features = [{"input_features": instance['audios']} for instance in instances]
            audios = self.audio_processor.feature_extractor.pad(input_features, return_tensors="pt")
            batch['audios'] = audios["input_features"]

        return batch