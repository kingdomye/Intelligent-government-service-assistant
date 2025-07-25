import requests
import os
from dotenv import load_dotenv

load_dotenv(r"HF_API_TOKEN.env")#先搞绝对路径，后面进行统一
HF_API_TOKEN=os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("请设置HF_API_TOKEN环境变量")
API_URL="https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
headers={
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# 定义一个函数classify，用于对文本进行分类
def classify(text,label_dict):
    print("classify")
    if not text.strip():
        raise ValueError("输入文本不能为空")
    if not label_dict:
        raise ValueError("标签字典不能为空")
    # 获取label_dict的键，即候选标签
    candidate_labels=list(label_dict.keys())
    # 构造payload，包括输入文本和候选标签
    payload={
        "inputs":text,
        "parameters":{"candidate_labels":candidate_labels}
    }
    # 发送post请求，将payload作为json数据发送到API_URL
    print("classify 2")
    response=requests.post(API_URL,headers=headers,json=payload)
    # 获取响应结果
    result=response.json()
    print("classify 3")
    top_label=result["labels"][0]
    top_score=result["scores"][0]
    # 获取分类结果中的第一个标签
    #根据出来的概率，如果概率小于一定的数值，则判断是与其无关的话语或者是type_dic里面没有的
    #直接返回None
    if top_score<0.8:
        print("输入文本与分类无关或者没有该服务项目")
        return None
    else:
        # 返回对应的标签
        return label_dict[top_label]

#下一类一个直接分
#这个全局变量，后面修改...
type_dic={
    "身份证":0,
    "户口本":1,
}


print(classify("我该上哪里去办理身份证",type_dic))