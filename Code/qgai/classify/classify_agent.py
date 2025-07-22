#主要分为关键字提取进行文本处理
# 还有文本相似度比对找到需要办理的业务
#先跟用户确认是否是这个业务
#然后根据业务找到对应的流程和表单

import requests

API_KEY="WwUfCgc1jttNeugXbP5NjFtb"
SECRET_KEY="yOnwOMTVs0fNkbXtlywpUZKxDtqwqFMB"
EXTRACT_URL="https://aip.baidubce.com/rpc/2.0/nlp/v1/txt_keywords_extraction"#关键字提取api
CLASSIFY_URL="https://aip.baidubce.com/rpc/2.0/nlp/v2/simnet"#文本相似度比对api


class BaiduNLP:
    def __init__(self, ):
        pass
    def _get_access_token(self):
        pass
    def extract_keywords(self, text,top_k=5):
        """提取输入文本关键词"""
        pass
    def calculate_similarity(self,text1,text2):
        """计算输入文本与对应业务关键词的相似度"""
        pass


class ClassifyAgent:
    def __init__(self,):
        pass
    def classify(self,text):
        """根据输入文本进行业务分类"""
        pass
    def get_process(self,):
        """根据输入文本获取对应流程"""
        pass
    def get_form(self,):
        """根据输入文本获取对应表单"""
        pass
    def handle(self,):
        """处理输入文本，返回对应流程和表单"""
        pass