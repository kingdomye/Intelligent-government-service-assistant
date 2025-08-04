import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys
import threading
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer  # 引入迭代式流式处理器
)
from peft import PeftModel
import torch

import asyncio


# 1. 路径配置
base_model_path = r"C:\PythonProject\deepseek-r1-7b"  # 原始模型路径
lora_path = "C:/PythonProject/25summer/ds_lr/deepseek_lora_final"  # LoRA权重路径

offload_dir = "./offload_inference"  # 参数卸载目录

# 创建卸载目录（如果不存在）
os.makedirs(offload_dir, exist_ok=True)

# 2. 配置4-bit量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4-bit量化
    bnb_4bit_use_double_quant=True,  # 双量化，进一步节省显存
    bnb_4bit_quant_type="nf4",  # 使用NF4量化类型（适合LLM）
    bnb_4bit_compute_dtype=torch.float16  # 计算时使用FP16精度
)

# 3. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 4. 以4-bit量化方式加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,  # 应用4-bit量化配置
    device_map="auto",  # 自动分配设备
    trust_remote_code=True,
    offload_folder=offload_dir,  # 卸载路径
    offload_state_dict=True
)

# 5. 加载LoRA权重
model = PeftModel.from_pretrained(
    model,
    lora_path,
    device_map="auto",
    offload_folder=offload_dir
)
model.eval()  # 切换到推理模式


# 6. 流式推理函数（使用TextIteratorStreamer实现）
async def generate_streaming_response(user_input, raw_text=None, max_length=1024):
    prompt = f"""
    # 严格按以下规则处理：
    # 1. 仅保留与用户特征相关的步骤;
    # 2. 保留必要信息和详细流程;
    # 3. 根据用户特征，便于用户理解;
    # 4. 如果用户数据不符合显现实逻辑，输出“<-3>请输入正确的信息”，并附上原因;
    # 5. 根据用户决定使用的语言
    # 6. 使用markdown语法，直接输出流程不带标注,如：

    ---
    ##步骤
    -解释1
    -解释2
    ---

    原始流程：{raw_text}
    用户特征：{json.dumps(user_input, ensure_ascii=False)}

    """
    # 原始流程：{raw_text}
    character = "谨记你是一个简洁干练的政务流程精简专家，除了用户需要的政务流程不会输出多余的内容，而且内容格式规范"

    # 构建正确的对话格式
    messages = [
        {"role": "system", "content": character},
        {"role": "user", "content": prompt}
    ]
    # 应用模型的聊天模板
    payload = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        payload,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)

    # 创建TextIteratorStreamer实例
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,  # 跳过提示词部分
        skip_special_tokens=True,  # 跳过特殊token
        timeout=10.0
    )

    # 配置生成参数
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3
    }

    # 在单独的线程中运行生成过程
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    # 从streamer迭代获取生成的内容
    for text in streamer:
        yield text

    # 等待线程结束
    thread.join()


# async def main():
#     user_input = {
#         "姓名": "陆潇峰",
#         "年龄": 19,
#         "性别": "不愿透露",
#         "民族": "汉",
#         "联系电话": "14111451541",
#         "籍贯": "肇庆",
#         "政治面貌": "群众",
#         "宗教信仰": "无",
#         "婚姻状况": "未婚",
#         "出生日期": "20060606",
#         "居住地": "广州",
#         "户口": "肇庆",
#         "业务类型": "户口迁移"
#     }
#     flow_path = 'flowdata.json'
#     business_id = '0'
#
#     try:
#         print("测试流式输出：")
#         print(end=' ', flush=True)
#
#         # 流式接收并打印结果
#         pr = 0
#         async for chunk in generate_streaming_response(user_input=user_input):
#             if "--" in chunk:
#                 pr = 1
#             if pr == 1:
#                 print(chunk, end='', flush=True)
#                 sys.stdout.flush()  # 确保内容即时显示
#
#         print()  # 最后换行
#         pr = 0
#
#     except Exception as e:
#         print(f"推理出错：{e}")
#
# # 测试流式输出
# if __name__ == "__main__":
#      asyncio.new_event_loop().run_until_complete(main())
