import json
import numpy as np
from faker import Faker
import random
from datetime import datetime

fake = Faker('zh_CN')

# 字段与JSON文件路径的对应关系（顺序严格对应）
field_names = [
    "性别", "民族", "籍贯", "职业", "地址",
    "户口", "政治面貌", "婚姻状况", "宗教", "学历"
]
paths = [
    "sex.json", "nations.json", "cities.json", "job.json", "cities.json",
    "cities.json", "appearance.json", "marriage.json", "religions.json", "degree.json"
]

def drop_info(info_, num=5):
    """随机将部分字段设置为<UNK>"""
    keys = list(info_.keys())
    num_to_empty = random.randint(0, num)  # 随机选择0-5个字段
    if num_to_empty > 0 and len(keys) >= num_to_empty:
        keys_to_empty = random.sample(keys, num_to_empty)
        for key in keys_to_empty:
            info_[key] = "<UNK>"
    return info_


def is_valid_date(date_str):
    """检查日期字符串是否合法（格式：YYYY-MM-DD）"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def load_field_mappings(field_names, paths):
    """
    为每个字段加载独立的ID-名称映射表
    返回结构: {字段名: {"name_to_id": {}, "id_to_name": {}}}
    """
    field_mappings = {}
    for field, path in zip(field_names, paths):
        name_to_id = {}
        id_to_name = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for id_str, name in data.items():
                    try:
                        id_num = int(id_str)  # 统一转换为整数ID
                        name_to_id[name] = id_num
                        id_to_name[id_num] = name
                    except ValueError:
                        print(f"文件[{path}]中的ID[{id_str}]不是有效数字，已跳过")
            field_mappings[field] = {
                "name_to_id": name_to_id,
                "id_to_name": id_to_name
            }
        except Exception as e:
            print(f"读取字段[{field}]的文件[{path}]出错: {str(e)}")
            # 初始化空映射表，避免后续报错
            field_mappings[field] = {"name_to_id": {}, "id_to_name": {}}
    return field_mappings


def get_field_id_range(field_mappings, field):
    """获取字段的有效ID范围（最小/最大）"""
    id_to_name = field_mappings[field]["id_to_name"]
    if not id_to_name:  # 映射表为空时返回默认范围
        return (1, 10)
    ids = list(id_to_name.keys())
    return (min(ids), max(ids))


def map_features_to_id(user_features, field_mappings):
    """按字段专属映射表将名称转换为ID"""
    mapped = {}
    for key, value in user_features.items():
        if key in field_mappings:  # 只处理有映射表的字段
            name_to_id = field_mappings[key]["name_to_id"]
            if key == "职业":
                # 职业特殊处理：不存在则设为1
                mapped[key] = name_to_id.get(value, 1)
            else:
                # 其他字段不存在映射时使用默认ID
                mapped[key] = name_to_id.get(value, 1)
        else:
            # 无映射表的字段（如姓名、电话等）保留原值
            mapped[key] = value
    return mapped


def map_id_to_features(id_features, field_mappings):
    """按字段专属映射表将ID转换为名称"""
    original = {}
    for key, id_val in id_features.items():
        if key in field_mappings and id_val is not None:
            id_to_name = field_mappings[key]["id_to_name"]
            # 无法映射时保留原始ID（后续会处理）
            original[key] = id_to_name.get(id_val, id_val)
        else:
            original[key] = id_val
    return original


def generate_noisy_data(original_data, field_mappings, num_samples=2, epsilon=1.):
    """实现name→id→加噪→name→drop的完整流程，确保字段映射正确"""
    # 1. 将原始特征映射为ID（只处理有映射表的字段）
    id_features = map_features_to_id(original_data, field_mappings)

    # 2. 提取需要处理的字段和原始ID
    process_fields = field_names  # 只处理有映射表的字段
    original_ids = []
    for field in process_fields:
        # 确保ID是整数
        try:
            original_id = int(id_features[field])
        except (ValueError, TypeError):
            original_id = 1  # 无效ID时使用默认值
        original_ids.append(original_id)

    # 3. 处理出生日期（单独处理，确保格式正确）
    try:
        birth_date = datetime.strptime(original_data["出生日期"], "%Y-%m-%d")
        original_birth_int = int(birth_date.strftime("%Y%m%d"))
    except (ValueError, KeyError):
        original_birth_int = 20010101  # 默认出生日期

    # 4. 生成带噪声的ID数据
    # 创建原始ID矩阵并添加噪声
    id_matrix = np.array([original_ids for _ in range(num_samples)])
    noise = np.random.laplace(0, 1 / epsilon, id_matrix.shape)
    noisy_id_matrix = np.abs(np.rint(id_matrix + noise)).astype(int)

    # 5. 限制每个字段的ID在有效范围内
    for i in range(num_samples):
        for j, field in enumerate(process_fields):
            min_id, max_id = get_field_id_range(field_mappings, field)
            # 确保噪声ID在有效范围内
            noisy_id_matrix[i][j] = np.clip(noisy_id_matrix[i][j], min_id, max_id)

    # 6. 生成出生日期噪声（控制在合理范围）
    birth_dates = np.array([original_birth_int for _ in range(num_samples)])
    # 日期噪声范围：±3年（约1095天）
    birth_noise = np.random.laplace(0, 1095 / epsilon, num_samples).astype(int)
    noisy_birth_dates = birth_dates + birth_noise

    # 7. 生成最终样本
    noisy_samples = []
    for i in range(num_samples):
        # 创建带噪声的ID特征
        noisy_id_feature = {}
        for j, field in enumerate(process_fields):
            noisy_id_feature[field] = noisy_id_matrix[i][j]

        # 将带噪声的ID映射回名称
        noisy_name_feature = map_id_to_features(noisy_id_feature, field_mappings)

        # 处理无法映射的情况（使用原始值）
        for field in process_fields:
            if isinstance(noisy_name_feature[field], int):  # 仍为数字表示未映射成功
                noisy_name_feature[field] = original_data[field]

        # 构建完整样本
        sample = original_data.copy()
        # 更新处理过的字段
        for field in process_fields:
            sample[field] = noisy_name_feature[field]

        # 处理出生日期（确保格式正确）
        try:
            birth_int = int(noisy_birth_dates[i])
            birth_str = str(birth_int).zfill(8)  # 确保8位数字
            birth_format = f"{birth_str[:4]}-{birth_str[4:6]}-{birth_str[6:8]}"
            if not is_valid_date(birth_format):
                raise ValueError("无效日期")
            sample["出生日期"] = birth_format
        except:
            # 无效日期时使用原始日期
            sample["出生日期"] = original_data["出生日期"]

        # 生成随机个人信息（保护隐私）
        sample["姓名"] = fake.name()
        sample["电话号码"] = fake.phone_number()
        sample["身份证号"] = fake.ssn()
        sample["电子邮箱"] = fake.email()

        # 应用drop操作
        sample = drop_info(sample)

        noisy_samples.append(sample)

    return noisy_samples


if __name__ == "__main__":
    # 加载字段专属映射表
    field_mappings = load_field_mappings(field_names, paths)

    with open('zs_info.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    noisy_samples = []
    num_samples = 10

    for user_features in data:
        noisy_samples += generate_noisy_data(
            user_features,
            field_mappings,
            num_samples=num_samples,
            epsilon=1.0  # 噪声参数，值越大噪声越小
        )

    business_types = ["办理身份证", "户口迁移", "水电费缴纳", "医保参保", "缴纳个人所得税"]
    # 按要求格式整理数据（包含业务类型）
    final_data = []
    # 遍历每个生成的样本
    for sample in noisy_samples:
        # 为每种业务类型各生成一条数据
        for business in business_types:
            # 构建user_features字典（转换部分字段名以匹配示例格式）
            formatted_features = {
                "姓名": sample["姓名"],
                "性别": sample["性别"],
                "民族": sample["民族"],
                "联系电话": sample["电话号码"],
                "籍贯": sample["籍贯"],
                "政治面貌": sample["政治面貌"],
                "宗教信仰": sample["宗教"],
                "婚姻状况": sample["婚姻状况"],
                "出生日期": sample["出生日期"],
                "居住地": sample["地址"],
                "户口": sample["户口"],
                "业务类型": business
            }

            # 构建最终条目
            entry = {
                "instruction": {
                    "user_features": formatted_features
                },
                "output": ""
            }
            final_data.append(entry)

    # 保存结果到JSON文件
    output_file = "process_user_data_pro.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    total = len(noisy_samples) * len(business_types)
    print(f"成功生成{total}条数据（{len(noisy_samples)}个样本 × {len(business_types)}种业务），已保存到{output_file}")
