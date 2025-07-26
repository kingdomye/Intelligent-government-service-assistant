import requests
import json
import re

def deal_flow_l(user_input, raw_text:str, cat_gril = False)->str:
    """
    调用local的 deepseekAPI 处理流程
    :param user_input: user info
    :param raw_text: original flow
    :param cat_gril: ^v^
    :return: final flow
    """
    # url = "http://localhost:11434/api/chat"
    url = 'http://192.168.1.165:11434/api/chat'
    headers = {"Content-Type": "application/json"}

    if cat_gril:
        character = "你是一只活泼的猫娘，对用户的称呼为‘主人’，你自称‘小linlin’，经常用语气词‘喵~’，打招呼时说'ヽ(=^･ω･^=)'"
        temperature = 0.01
    else:
        character = "谨记你是一个简洁干练的政务流程精简专家，除了用户需要的政务流程不会输出多余的内容，而且内容格式规范"
        temperature = 0.01

    prompt = f"""
    严格按以下规则处理：
    1. 仅保留与用户特征相关的步骤;
    2. 保留必要信息和详细流程;
    3. 根据用户特征，便于用户理解;
    4. 如果用户数据不符合显现实逻辑，输出“<-3>请输入正确的信息”，并附上原因;
    5. 使用markdown语法，直接输出流程不带标注,如：
    ---
    ##步骤
    -解释1
    -解释2
    ---

    用户特征：{json.dumps(user_input, ensure_ascii=False)}
    原始流程：{raw_text}
    """

    payload = json.dumps({
        "model": "DeepSeek-r1:7b",  # model参数
        "messages": [
            {"role": "system", "content": character},
            {"role": "user", "content": prompt}
        ],
        # 是否开启流式推理, 默认为False, 表示不开启流式推理
        "stream": False,
        # 在流式输出时是否展示使用的token数目。只有当stream为True时改参数才会生效。
        # "stream_options": { "include_usage": True },
        # 控制采样随机性的浮点数，值较低时模型更具确定性，值较高时模型更具创造性。"0"表示贪婪取样。默认为0.6。
        "temperature": temperature,
    }, ensure_ascii=False)


    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    # print(response.text)

    resp = response.json()
    output = resp['message']['content']
    output = re.sub(r'<think>.*?</think>\n\n', '', output, flags=re.DOTALL)
    if len(output) > 0:
        return output
    else:
        return '-2'


def debug():
    user_input = {
        "姓名": "陆潇峰",
        "年龄": 19,
        "性别":"不愿透露",
        "民族": "汉",
        "联系电话": "14111451541",
        "籍贯": "肇庆",
        "政治面貌": "群众",
        "宗教信仰": "无",
        "婚姻状况": "未婚",
        "出生日期": "",
        "居住地": "广州",
        "户口": "肇庆",
        "业务类型":"缴纳个人所得税"
    }
    flow_path = 'flows.json'
    business_id = '4'
    # 原始流程文本（从 flows.json 中加载）
    with open(flow_path, mode='r', encoding='utf-8') as f:
        flows = json.load(f)
        raw_text = flows[business_id]

    output = deal_flow_l(user_input, raw_text, cat_gril=True)
    print(output)


debug()