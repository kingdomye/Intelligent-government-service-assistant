import requests
import json

# 替换为你的 DeepSeek API Key
API_KEY = "sk-hbaufxiqvknbxmglongfeplcmpfuydhkemqrahulmjecfwoc"  # 从开放平台获取
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 用户输入（可扩展为表单/前端输入）
user_input = {
    "姓名": "张三",
    "年龄": 65,
    "居住地": "广州",
    "户口": "上海",
    "业务": "迁移户口"
}
flow_path = 'flows.json'
business_id = '1'
# 原始流程文本（从 flows.json 中加载）
with open(flow_path, mode='r', encoding='utf-8') as f:
    flows = json.load(f)
    raw_text = flows[business_id]


# 构造 Prompt
prompt = f"""
你是一个流程精简专家，请严格按以下规则处理：
1. 仅保留与用户特征直接相关的步骤
2. 删除所有冗余说明
3. 用"→"表示步骤顺序，每个步骤不超过20字

用户特征：{json.dumps(user_input, ensure_ascii=False)}
原始流程：{raw_text}
"""

# 请求数据
data = {
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [
        {"role": "system", "content": "你是一个流程精简专家，请根据用户信息生成个性化流程。"},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 100
}

# 发送请求
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(API_URL, headers=headers, json=data, timeout=30)

# 解析结果
if response.status_code == 200:
    result = response.json()["choices"][0]["message"]["content"]
    print("个性化流程：")
    print(result)
else:
    print(f"请求失败：{response.status_code}")
    print(response.text)
