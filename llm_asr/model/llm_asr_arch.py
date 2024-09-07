#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch

from llm_asr.constants import IGNORE_INDEX, AUDIO_TOKEN_INDEX
from llm_asr.model.whisper_encoder import WhisperAudioTower
from llm_asr.model.audio_projector import get_vision_projector


class LlmAsrMetaModel:

    def __init__(self, config):
        super(LlmAsrMetaModel, self).__init__(config)

        if hasattr(config, "mm_audio_tower"):
            self.initialize_audio_modules(config)
        else:
            self.audio_tower = None
            self.mm_projector = None

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def initialize_audio_modules(self, model_args):
        audio_tower = model_args.audio_tower if hasattr(model_args, "audio_tower") else model_args.mm_audio_tower
        mm_audio_select_layer = model_args.mm_audio_select_layer
        mm_audio_select_feature = model_args.mm_audio_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter if hasattr(model_args, "pretrain_mm_mlp_adapter") else None

        self.config.mm_audio_tower = audio_tower
        self.config.audio_encoder_type = model_args.audio_encoder_type if hasattr(model_args, 'audio_encoder_type') else None
        if model_args.audio_encoder_type == "Whisper":
            self.audio_tower = WhisperAudioTower(
                audio_tower, 
                mm_audio_select_layer,
                mm_audio_select_feature,
                delay_load=True,
            )
        else:
            raise ValueError(f"model_args.audio_encoder_type: {model_args.audio_encoder_type} not find")
        
        if self.audio_tower.is_loaded is False:
            self.audio_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_audio_select_layer = mm_audio_select_layer
        self.config.mm_audio_select_feature = mm_audio_select_feature
        self.config.audio_encoder_type = model_args.audio_encoder_type
        self.config.mm_hidden_size = self.audio_tower.hidden_size

        self.mm_projector = get_vision_projector(self.config)

        # In case it is frozen by LoRA
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):
    base_model = "" # gpt2 or llama or gptneox

    @abstractmethod
    def get_model(self):
        pass

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def encode_audios(self, audios):
        audio_features = self.get_model().get_audio_tower()(audios) #.to(torch.bfloat16) # TODO 推論時に必要？
        audio_features = self.get_model().mm_projector(audio_features)
        return audio_features
    
    def embed(self, input_ids):
        if self.base_model == "gpt2":
            return self.transformer.wte(input_ids)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, audios
    ):
        audio_tower = self.get_audio_tower()
        if audio_tower is None or audios is None or input_ids.shape[1] == 1:
            if past_key_values is not None and audio_tower is not None and audios is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        audio_features = self.encode_audios(audios).to(self.device)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_audio_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_audios = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            if num_audios == 0:
                cur_audio_features = audio_features[cur_audio_idx]
                cur_input_embeds_1 = self.embed(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_audio_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_audio_idx += 1
                continue

            audio_token_indices = [-1] + torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            # AUDIO_TOKEN_INDEXで前後にtokenを分割
            # ex. input_ids -> cur_input_ids_noim
            # [1 2 3 -200 4 5 6] -> [1 2 3], [4 5 6]
            for i in range(len(audio_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[audio_token_indices[i]+1:audio_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[audio_token_indices[i]+1:audio_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            
            # cur_input_embeds_no_im[0].size() (27, 768)
            # cur_input_embeds_no_im[1].size() (xxx, 768)
            cur_input_embeds = self.embed(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # AUDIO_TOKEN_INDEXの部分を画像特徴量に置き換える
            # cur_audio_fearures.size() (576, 768)
            for i in range(num_audios + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_audios:
                    cur_audio_features = audio_features[cur_audio_idx]
                    cur_audio_idx += 1
                    cur_new_input_embeds.append(cur_audio_features)
                    cur_new_labels.append(torch.full((cur_audio_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as audio embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
