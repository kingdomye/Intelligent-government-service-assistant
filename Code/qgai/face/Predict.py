# ================================
# @File         : Predict.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 输入图片对模型进行id匹配（贝叶斯分类版本）
# ================================

import cv2
import joblib
import numpy as np
from face.model_loader import get_components

from face.utils.face_utils import (
    bin_to_image_array, preprocess_face, extract_features,
    detect_largest_face, normalize_features, img_to_bin
)

# 获取组件
components = get_components()
logger = components['logger']
paths = components['paths']
models = components['models']

# 模型路径（与训练代码保持一致）
model_path = paths['model_path']
le_path = paths['le_path']

# 加载特征提取模型、检测器等
embedder = models['embedder']
detectors = models['detectors']

# 加载贝叶斯分类器和标签编码器（与训练代码一致）
clf = joblib.load(model_path)
le = joblib.load(le_path)


def single_img_bin_predict(img_bin, min_acc=0.6):
    img = bin_to_image_array(img_bin)
    if img is None:
        logger.error("无法处理图像数据")
        return None

    # 转为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # 已经是灰度图

    # 使用公共人脸检测方法
    face_rect = detect_largest_face(gray, detectors)
    if face_rect is None:
        logger.warning("未检测到人脸")
        return None

    (x, y, w, h) = face_rect
    # 提取人脸区域并预处理
    face_roi = gray[y:y + h, x:x + w]
    face_roi = preprocess_face(face_roi)

    # 提取人脸特征
    features = extract_features(face_roi, embedder)

    # 使用公共特征归一化方法
    features = normalize_features(features)
    features = features.flatten().reshape(1, -1)

    # 识别人脸（使用贝叶斯分类器）
    try:
        predictions = clf.predict_proba(features)
        max_prob = np.max(predictions)
        predicted_class = np.argmax(predictions)

        # 转换为原始标签
        username = le.inverse_transform([predicted_class])[0]
        confidence_text = f"{max_prob * 100:.1f}%"

        # 应用置信度阈值
        if max_prob < min_acc:
            logger.info(f"置信度 {max_prob:.2f} 低于阈值 {min_acc}，识别失败")
            return None

        return username, confidence_text
    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        return None


def cv2_predict(imgs_bin, min_acc=0.6):
    """批量处理图像并返回预测结果及统计信息"""
    # 批量预测获取原始结果
    results = [single_img_bin_predict(img_bin, min_acc) for img_bin in imgs_bin]

    # 初始化标签-置信度列表字典
    label_confidences = {}

    # 处理结果生成字典（使用列表推导式预处理有效数据）
    # 先过滤出有效结果并转换为(标签, 置信度)元组列表
    valid_results = [
        (str(item[0]), float(item[1].strip('%')))
        for item in results
        if item is not None
    ]

    # 构建标签到置信度列表的映射
    for label, conf in valid_results:
        if label in label_confidences:
            label_confidences[label].append(conf)
        else:
            label_confidences[label] = [conf]

    # 计算其他统计信息
    count = {label: len(conf_list) for label, conf_list in label_confidences.items()}
    best_label = max(count, key=count.get)
    best_confidence = label_confidences[best_label]
    best_confidence_avg = np.mean(best_confidence)

    print(label_confidences)

    if best_confidence_avg < min_acc:
        return None

    return best_label


if __name__ == "__main__":
    # pass
    # 单张照片测试
    from PIL import Image

    test_img = Image.open("./facedata/test.jpg")
    img_bin = img_to_bin(test_img)
    res, _ = single_img_bin_predict(img_bin, min_acc=0.5)
    print(res)

    # # 照片组
    # import pickle
    #
    # with open("./facedata/face_6_bin_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     face_id = data['face_id']
    #     images = data['images']
    #
    # print(cv2_predict(images))
