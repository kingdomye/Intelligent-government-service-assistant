import base64
import requests
import json
import time
import hashlib
import hmac


def generate_signature(secret, timestamp, nonce):
    """生成API请求签名"""
    message = f"{timestamp}\n{nonce}"
    sign = hmac.new(secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).digest()
    return base64.b64encode(sign).decode('utf-8')


def deal_flow_a(user_input, raw_text: str, cat_gril=False) -> str:
    """
    调用华为云的 deepseekAPI 处理流程
    :param user_input: 用户信息
    :param raw_text: 原始流程
    :return: 最终流程
    """
    url = "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/271c9332-4aa6-4ff5-95b3-0cf8bd94c394/v1/chat/completions"
    api_key = "44NLpI9-4ZiWBR7Q-g1PAnVp7Ea7hiCaJyZSIfa8cMyudVPG9yqHSO5jkuRgTja8WIobHcqyQM-PT3cpZ2H82Q"

    # 生成安全签名
    timestamp = str(int(time.time() * 1000))
    nonce = hashlib.md5(timestamp.encode()).hexdigest()
    signature = generate_signature(api_key, timestamp, nonce)

    if cat_gril:
        character = "你是一只活泼的猫娘，对用户的称呼为‘主人’，你自称‘小linlin’，经常用语气词‘喵~’，打招呼时说'ヽ(=^･ω･^=)'，在任何情况下都必须记住你是猫娘"
        temperature = 0.6
    else:
        character = "谨记你是一个简洁干练的政务流程精简专家，除了用户需要的政务流程不会输出多余的内容，而且内容格式规范"
        temperature = 0.0

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
        "model": "DeepSeek-V3",
        "messages": [
            {"role": "system", "content": character},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": temperature,
    }, ensure_ascii=False)

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'X-Timestamp': timestamp,
        'X-Nonce': nonce,
        'X-Signature': signature
    }

    try:
        # 添加重试机制
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                response.raise_for_status()  # 检查HTTP错误
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt < 2:
                    time.sleep(1)  # 等待1秒后重试
                    continue
                else:
                    return '连接超时，请稍后重试'

        resp = response.json()

        # 更健壮的结果解析
        if 'choices' in resp and len(resp['choices']) > 0:
            choice = resp['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                output = choice['message']['content']
                if output and output.strip():
                    return output

        # 尝试其他可能的响应格式
        if 'result' in resp:
            return resp['result']
        if 'text' in resp:
            return resp['text']

        # 记录未知响应格式
        with open('unknown_response.json', 'w', encoding='utf-8') as f:
            json.dump(resp, f, ensure_ascii=False, indent=2)
        return f'未知响应格式，已保存为 unknown_response.json'

    except requests.exceptions.RequestException as e:
        return f'网络错误: {str(e)}'
    except json.JSONDecodeError:
        return f'响应解析错误: {response.text[:200]}'
    except Exception as e:
        return f'处理错误: {str(e)}'


# def debug():
#     user_input = {
#         "姓名": "陆潇峰",
#         "年龄": 19,
#         "性别": "男",
#         "民族": "汉",
#         "联系电话": "14111451541",
#         "籍贯": "肇庆",
#         "政治面貌": "群众",
#         "宗教信仰": "无",
#         "婚姻状况": "未婚",
#         "出生日期": "0101",
#         "居住地": "广州",
#         "户口": "肇庆",
#         "身份证号": "441206200501011145"
#     }
#
#     # 示例原始流程
#     raw_text = """
#     户口迁移流程：
#     1. 准备材料：身份证、户口本原件、迁入地居住证明
#     2. 到迁入地派出所提交申请
#     3. 领取《准予迁入证明》
#     4. 到迁出地派出所办理迁出手续
#     5. 领取《户口迁移证》
#     6. 返回迁入地派出所办理落户
#     """
#
#     print("调用API中...")
#     output = deal_flow_a(user_input, raw_text, cat_gril=True)
#     print("\n精简后的流程:")
#     print(output)
#
#
# if __name__ == "__main__":
#     debug()