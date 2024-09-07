from typing import Optional

import torch
import torch.nn as nn

from transformers import (
    WhisperModel, WhisperProcessor, WhisperConfig, WhisperFeatureExtractor
)


class WhisperAudioTower(nn.Module):
    def __init__(
        self, 
        audio_tower_name: str="openai/clip-vit-large-patch14-336", 
        mm_audio_select_layer: int=-2, # v1.5 is -2
        mm_audio_select_feature: str="patch",
        delay_load: bool=False,
        requires_grad: bool=False,
    ):
        super().__init__()

        self.is_loaded = False
        self.requires_grad = requires_grad

        self.audio_tower_name = audio_tower_name
        self.select_layer = mm_audio_select_layer
        self.select_feature = mm_audio_select_feature

        self.audio_processor = None
        self.audio_tower = None
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)

    def load_model(self):
        self.audio_processor = WhisperProcessor.from_pretrained(self.audio_tower_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.audio_tower_name)
        self.audio_tower = WhisperModel.from_pretrained(self.audio_tower_name).encoder
        self.audio_tower.requires_grad_(self.requires_grad)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        """
        if self.select_feature == 'patch':
            audio_features = audio_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            audio_features = audio_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        """
        return audio_features

    @torch.no_grad()
    def forward(self, audios):
        if type(audios) is list:
            audio_features = []
            for audio in audios:
                audio_feature = self._forward_feature(audio.unsqueeze(0))
                audio_features.append(audio_feature)
        else:
            audio_features = self._forward_feature(audios)
            
        return audio_features
    
    def _forward_feature(self, inputs):
        audio_forward_outs = self.audio_tower(inputs.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        audio_features = self.feature_select(audio_forward_outs) #.to(torch.float32)
        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):        
        return self.config.hidden_size
    
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
