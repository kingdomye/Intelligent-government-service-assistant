import requests
import json


def deal_flow(user_input, raw_text:str)->str:
    """
    调用华为云的 deepseekAPI 处理流程
    """
    url = "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/271c9332-4aa6-4ff5-95b3-0cf8bd94c394/v1/chat/completions"
    key = "44NLpI9-4ZiWBR7Q-g1PAnVp7Ea7hiCaJyZSIfa8cMyudVPG9yqHSO5jkuRgTja8WIobHcqyQM-PT3cpZ2H82Q"

    prompt = f"""
    你是一个流程精简专家，请严格按以下规则处理：
    1. 仅保留与用户特征直接相关的步骤
    2. 用"→"表示步骤顺序，保留必要信息
    3. 根据用户特征给出便于用户理解的详细流程
    4. 如果用户数据不符合显现实逻辑，输出“<-3>请输入正确的信息”

    用户特征：{json.dumps(user_input, ensure_ascii=False)}
    原始流程：{raw_text}
    """

    payload = json.dumps({
        "model": "DeepSeek-V3",  # model参数
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        # 是否开启流式推理, 默认为False, 表示不开启流式推理
        "stream": False,
        # 在流式输出时是否展示使用的token数目。只有当stream为True时改参数才会生效。
        # "stream_options": { "include_usage": True },
        # 控制采样随机性的浮点数，值较低时模型更具确定性，值较高时模型更具创造性。"0"表示贪婪取样。默认为0.6。
        "temperature": 0.0
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    # print(response.text)
    resp = response.json()
    if 'choices' in resp:
        output = resp['choices'][0]['message']['content']
        if len(output) > 0:
            return output
        else:
            return '-2'
    else:
        return '-2'


def debug():
    user_input = {
        "姓名": "张三",
        "年龄": 18,
        "居住地": "广州",
        "户口": "北京",
    }
    flow_path = 'flows.json'
    business_id = '1'
    # 原始流程文本（从 flows.json 中加载）
    with open(flow_path, mode='r', encoding='utf-8') as f:
        flows = json.load(f)
        raw_text = flows[business_id]

    output = deal_flow(user_input, raw_text)
    print(output)

# debug()