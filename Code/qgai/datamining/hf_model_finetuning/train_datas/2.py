import json
from datasets import Dataset


def process_dataset(raw_dataset_path, processed_path):
    # 1. 读取原始JSON数据
    with open(raw_dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    for item in raw_data:
        # 2. 将嵌套的user_features转换为结构化文本
        user_features = item["instruction"]["user_features"]
        # 生成类似"姓名：李坤；年龄：40；性别：男；..."的文本
        instruction_text = "; ".join([f"{k}：{v}" for k, v in user_features.items()])

        # 3. 构建新的条目（instruction为字符串，output保持不变）
        processed_item = {
            "instruction": instruction_text,  # 字符串格式，避免类型错误
            "output": item["output"]
        }
        processed_data.append(processed_item)

    # 4. 保存处理后的数据为新JSON文件
    with open(processed_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 5. 测试加载处理后的数据
    dataset = Dataset.from_json(processed_path)
    print("数据处理成功！样本数：", len(dataset))
    return dataset


# 使用示例
raw_path = "orgdata.json"  # 原始数据路径
processed_path = "processed_train_data.json"  # 处理后的数据路径
dataset = process_dataset(raw_path, processed_path)