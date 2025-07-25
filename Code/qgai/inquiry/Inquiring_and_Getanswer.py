import numpy
import requests
from dotenv import load_dotenv
import os
import json
import re

load_dotenv(r"qna_key.env")  # 从.env文件加载环境变量

api_key = os.getenv('API_KEY')
headers = {"Authorization": f"Bearer {api_key}"}

def similarity(text1, text2):
    api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    payload = {"inputs": {"source_sentence": text1, "sentences": [text2]}}
    res = requests.post(api_url, headers=headers, json=payload)
    return res.json()          # 返回余弦相似度分数列表

# print(similarity("今天天气很好", "今日天气不错"))

def extract_keywords(text):
    API_URL = "https://router.huggingface.co/hf-inference/models/uer/roberta-base-finetuned-cluener2020-chinese"
    payload = {"inputs": f"提取关键词:{text}"}  # 中文提示词
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

list = list([],)

def Inquiring(data_dict):
        """
        检查字典中是否有缺失字段，并返回提示信息
        参数:
            data_dict: 要检查的字典数据
            required_fields: 必填字段列表
        返回:
            list: 缺失字段的提示信息列表，若无缺失则返回空列表
        """
        for field in data_dict.keys():
            # 检查字段是否存在或值为空（None/空字符串/空列表等）
            if not data_dict[field]:
                return {field:f"请问您的「{field}」是什么？"}

        return None

def Get_Answer(text, required_field):
    if re.search(r'[123456789一二三四五六七八九]{4,}', text):
        result = re.findall(r'[123456789一二三四五六七八九]{4,}', text)
        result = [p for p in result if len(p) == 11]
        return result[0]

    else:
        best_match = 0.0
        keywords = extract_keywords(text)
        for i in range(len(keywords)):
            new_text = keywords[i]["word"]
            score = similarity(new_text, required_field)[0]
            if score > best_match:
                result = keywords[i]["word"]
                best_match = score
        return result

example = "陆潇锋的电话号码是11123426354"
print(Get_Answer(example,"号码"))
