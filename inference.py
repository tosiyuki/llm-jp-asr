import copy
import os
import glob
import librosa
import pandas as pd

import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperFeatureExtractor

from llm_asr.constants import DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from llm_asr.model.asr_gpt2 import LlmAsrGpt2ForCausalLM
from llm_asr.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llm_asr.train.dataset import tokenizer_audio_token

BASE_MODEL = "openai/whisper-large-v3"
MODEL_NAME = './output/checkpoints/llm-jp-asr'
INPUT_PATH = "./data"
EXP = "004-2"
OUTOUT_PATH = f"output/exp_{EXP}"


if __name__ == "__main__":
    # 保存用のディレクトリの作成
    os.makedirs(OUTOUT_PATH, exist_ok=True)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_path = MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = LlmAsrGpt2ForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="eager",

    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
    model.eval()

    # テストデータの読み込み
    # テストデータの読み込み
    df_test = pd.read_csv(f"{INPUT_PATH}/test.csv")
    default_prompt = f"ユーザー: {DEFAULT_AUDIO_TOKEN}\nシステム: "

    list_transcription = [] #文字起こし結果
    list_transcription_vad = []
    list_audio_file_names = []

    for audio_path in tqdm(df_test["audio_path"].to_list(), total=len(df_test["audio_path"].to_list())):
        # 拡張子を取り除いたファイル名を取り出す
        audio_file_name = os.path.basename(audio_path)
        file_pattern = os.path.join("test_vad_rm_noise", f"{audio_file_name}_*.mp3")
        audio_file_names = glob.glob(file_pattern)
        list_audio_file_names.extend(audio_file_names)

        transcription = ""
        for i in range(len(audio_file_names)):
            ## 音声データ読み込み
            audio, sr = librosa.load(f"test_vad_rm_noise/{audio_file_name}_{i+1}.mp3", sr=16000)

            ##音声データを、テンソルに変換：Whisper入力用
            input_features = feature_extractor(audio, sampling_rate=sr).input_features[0]
            audio_tensor = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")

            # プロンプト作成
            prompt = copy.copy(default_prompt)
            input_ids = tokenizer_audio_token(
                prompt, 
                tokenizer, 
                AUDIO_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0)
            if device == "cuda":
                input_ids = input_ids.to(device)
            input_ids = input_ids[:, :-1] # </sep>がinputの最後に入るので削除する

            # predict
            with torch.inference_mode():
                output_id = model.generate(
                    inputs=input_ids,
                    audios=audio_tensor,
                    do_sample=False,
                    temperature=0.,
                    max_new_tokens=512,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id
                )[0]
            output_id = output_id.to("cpu")
            transcription += tokenizer.decode(output_id[:][input_ids.shape[1]:], skip_special_tokens=True)
        
    # データの出力
    """
    vad_result = pd.DataFrame({
        "audio_path": list_audio_file_names,
        "target": list_transcription_vad,
    })
    vad_result.to_csv(f"{OUTOUT_PATH}/{EXP}_vad_output.csv", index=False)
    """

    result = pd.DataFrame({
        "ID": df_test["ID"].to_list(),
        "target": list_transcription,
    })
    result.to_csv(f"{OUTOUT_PATH}/{EXP}_submission.csv", index=False)