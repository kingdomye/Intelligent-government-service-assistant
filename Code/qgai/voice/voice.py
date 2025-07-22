import librosa
import io
import torch

import time
import whisper
from opencc import OpenCC
import os
import numpy as np


model_root=R"E:\Program\$knowlage\$AI\Model\whisper"
model_level="medium.pt"


print("<voice>:voice loading checkpoint "+model_level+" ...")
model = whisper.load_model(os.path.join(model_root, model_level))
print("<voice>:loading successful")

cc = OpenCC('t2s')


def bin_decode(bin_array,code_type="wav"):
    if code_type == "wav":
        PCM_array,_ = librosa.load(
            io.BytesIO(audio),  # 二进制数据转文件流
            sr=16000,  # 可选：重采样到 16kHz（Whisper 推荐）
            mono=True  # 转为单声道
        )
        return PCM_array

def voice2text(audio_array):
    print("<voice>:get voice to text requirement")


    result = model.transcribe(audio=torch.tensor(audio_array, dtype=torch.float32))
    content = result["text"]

    print("<voice>:transcribe successful!")
    return cc.convert(content)

def text2voice(text):
    print("<voice>:get voice to text requirement")


audio = open("晋升后交谈1.wav", "rb").read()


print(voice2text(bin_decode(audio)))




