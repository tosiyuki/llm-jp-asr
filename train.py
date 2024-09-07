import os
import pathlib
import pandas as pd
from typing import Dict

import torch
import transformers
from transformers import set_seed

from llm_asr.model.asr_gpt2 import LlmAsrGpt2ForCausalLM
from llm_asr.train.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from llm_asr.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llm_asr.train.llm_asr_trainer import LlmAsrTrainer


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # parameterのnameにgammaやbetaが入っているとfrom_pretrainedでロードする際、
        # 以下のような_fix_keyメソッドがあり正常に重みをロードできない.
        # def _fix_key(key):
        #    if "beta" in key:
        #        return key.replace("beta", "bias")
        #    if "gamma" in key:
        #        return key.replace("gamma", "weight")
        #    return key
        # gammaをweightに置き換える
        cpu_state_dict = {
            key.replace("gamma", "weight"): value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'audio_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def make_supervised_data_module(
        train_dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args
    ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        train_dataset,
        tokenizer=tokenizer,
        feature_extractor=data_args.feature_extractor
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, audio_processor=data_args.audio_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # seed初期化
    set_seed(training_args.seed)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map="auto",
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.base_model == "gpt2":
        model = LlmAsrGpt2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="eager",
            **bnb_model_from_pretrained_args
        )
    else:
        print(f"{model_args.base_model} is not found")
        exit(-1)

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        # Adapterの重みを調整するときに使うみたい
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        print("target_modules")
        if model_args.base_model == "gpt2":
            target_modules = ["c_attn"]
        elif model_args.base_model == "gpt_neox":
            target_modules = ["query_key_value"]
        elif model_args.base_model == "llama":
            target_modules = find_all_linear_names(model)
        else:
            print(f"{model_args.base_model} is not found")
            exit(-1)

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.unk_token

    if model.get_model().audio_tower is None:
        model.get_model().initialize_audio_modules(
            model_args=model_args,
        )
    
    audio_tower = model.get_audio_tower()
    audio_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.audio_processor = audio_tower.audio_processor
    data_args.feature_extractor = audio_tower.feature_extractor

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.get_audio_tower().audio_tower.requires_grad_(model_args.tune_audio_tower)

    # check parameter info
    # for name, param in model.named_parameters():
    #    print(f'Layer: {name}, requires_grad: {param.requires_grad}')

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # ゼロから学習する場合
    # データの読み込み
    # 訓練データの読み込み
    train = pd.read_csv(f"./data/train.csv")
    train_details = pd.read_csv(f"./data/train_details.csv")
    df_train = pd.merge(train, train_details,on="ID")

    # sentenseがnanの行は削除
    df_train = df_train.dropna(subset=["target_slice"])

    # datasetの作成
    train_audio = [f"train_vad_rm_noise/{i}.mp3" for i in df_train["DETAIL_ID"].to_list()]
    train_sentence = df_train["target_slice"].tolist()

    train_dataset = {
        "audio":train_audio,
        "sentence":train_sentence
    }

    data_module = make_supervised_data_module(
        train_dataset,
        tokenizer=tokenizer,
        data_args=data_args
    )
    trainer = LlmAsrTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        

if __name__ == '__main__':
    train()