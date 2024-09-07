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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         GPT2LMHeadModel, GPT2Config, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from llm_asr.model.llm_asr_arch import LlmAsrMetaModel, LlavaMetaForCausalLM


class LlmAsrConfig(GPT2Config):
    model_type = "llm-asr-jp"


class LlmAsrGpt2Model(LlmAsrMetaModel, PreTrainedModel):
    config_class = LlmAsrConfig

    def __init__(self, config: GPT2Config):
        super(LlmAsrGpt2Model, self).__init__(config)


class LlmAsrGpt2ForCausalLM(GPT2LMHeadModel, LlavaMetaForCausalLM):
    config_class = LlmAsrConfig
    base_model = "gpt2"

    def __init__(self, config):
        super(LlmAsrGpt2ForCausalLM, self).__init__(config)
        self.model = LlmAsrGpt2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                audios
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        audios = kwargs.pop("audios", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if audios is not None:
            _inputs['audios'] = audios
        return _inputs

AutoConfig.register("llm-asr-jp", LlmAsrConfig)
AutoModelForCausalLM.register(LlmAsrConfig, LlmAsrGpt2ForCausalLM)
