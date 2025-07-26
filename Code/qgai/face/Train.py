# ================================
# @File         : Train_v1.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 对输入的图片进行预处理以及模型训练（贝叶斯分类版本）
#                 主函数cv2_train(imgs_bin, user_id, min_acc=0.6)
#                 传入二进制图片数组、用户ID以及准确率阀值
#                 如果模型训练成功则返回True
# ================================

import os
import time
from multiprocessing import Pool
from os import cpu_count

import cv2
import random
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB  # 导入贝叶斯分类器

from face.Predict import single_img_bin_predict
from face.model_loader import get_components
from face.utils.face_utils import (
    bin_to_image, preprocess_face,
    extract_features
)

components = get_components()
logger = components['logger']
paths = components['paths']
models = components['models']

model_path = paths['model_path']
le_path = paths['le_path']
history_features_path = paths['history_features_path']
history_labels_path = paths['history_labels_path']

# 使用模型组件
embedder = models['embedder']
detectors = models['detectors']
classifier = models['classifier']
label_encoder = models['label_encoder']


def process_single_image(face_img_bin):
    """
    处理单张图片，用于多进程处理

    :param face_img_bin:
    :return: 模型处理后的展平向量
    """
    try:
        # face_img = bin_to_image(face_img_bin)
        # # 图片转为灰度图
        # face_img = face_img.convert("L")
        # img_numpy = np.array(face_img, dtype='uint8')

        # 直接用cv2处理二进制数据，避免PIL转换
        img_numpy = cv2.imdecode(np.frombuffer(face_img_bin, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_numpy is None:
            logger.warning("图片解码失败")
            return None

        faces = None
        for detector in detectors:
            faces = detector.detectMultiScale(
                img_numpy,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                break

        if len(faces) == 0:
            logger.warning("未检测到图片的人脸")

            return None

        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_img = img_numpy[y:y + h, x:x + w]

        face_img = preprocess_face(face_img)
        features = extract_features(face_img, embedder)

        # 贝叶斯分类器要求输入非负，对特征进行归一化处理
        features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
        features = np.clip(features, 0, None)  # 确保没有负值
        features = features.flatten()

        return features

    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        return None


def get_feature_and_labels(imgs_bin, user_id):
    """
    获取特征向量和对应的标签，使用多进程加速处理

    :param imgs_bin: 二进制图像数组
    :param user_id: 用户对应的id
    :return: 特征数组和标签数组
    """
    logger.info(f"发现{len(imgs_bin)}个图像文件，开始并行处理... ...")

    # 多线程处理图像
    num_processes = min(cpu_count(), len(imgs_bin))
    with Pool(num_processes) as pool:
        results = pool.map(process_single_image, imgs_bin)

    # 过滤无效结果
    valid_results = [(res, user_id) for res in results if res is not None]
    if not valid_results:
        return [], []

    features, labels = zip(*valid_results)
    return list(features), list(labels)


def merge_labels(le, new_labels):
    """
    合并已有的标签和新标签，确保所有标签都被正确编码
    """
    # 检查编码器是否已拟合（通过判断classes_是否存在）
    if hasattr(le, 'classes_'):
        existing_labels = list(le.classes_)
    else:
        # 编码器未拟合，初始化为空列表
        existing_labels = []

    # 找出新标签中不存在于已有标签中的部分
    new_unique_labels = list(set(new_labels) - set(existing_labels))

    if new_unique_labels:
        logger.info(f"发现新的人脸ID: {new_unique_labels}，将添加到模型中")
        # 创建新的LabelEncoder并合并所有标签
        new_le = LabelEncoder()
        all_labels = existing_labels + new_unique_labels
        new_le.fit(all_labels)
        return new_le
    else:
        logger.info("没有发现新的人脸ID，使用已有的标签编码器")
        return le


def cv2_train(imgs_bin, user_id, min_acc=0.6):
    """

    :param imgs_bin: 二进制图像数组
    :param user_id: 用户对应的id
    :param min_acc: 准确率阀值
    :return: 训练评估成果，True成功，False失败
    """
    start_time = time.time()
    logger.info('开始提取特征并训练贝叶斯模型，请耐心等待....')

    new_features, new_labels = get_feature_and_labels(imgs_bin, user_id)

    if not new_features or not new_labels:
        logger.error("没有有效的训练数据，无法进行训练")
        return False

    # 加载组件
    clf = classifier
    le = label_encoder

    # 检查历史特征和标签文件是否存在
    if os.path.exists(history_features_path) and os.path.exists(history_labels_path):
        train_first = False
        history_features = np.load(history_features_path)
        history_labels = np.load(history_labels_path)
    else:
        # 首次训练时初始化空数组
        train_first = True
        history_features = None
        history_labels = None

    # 处理首次训练单类别问题
    if clf is None and le is None:
        if len(set(new_labels)) == 1:
            logger.info("检测到单类别数据，添加默认'unknown'类别以满足贝叶斯训练要求")
            # 生成少量变体作为未知类别样本
            unknown_features = [f * 1.1 for f in new_features[:2]]
            new_features.extend(unknown_features)
            new_labels.extend(['unknown'] * len(unknown_features))

        # 首次训练初始化标签编码器
        le = LabelEncoder()
        encoded_labels = le.fit_transform(new_labels)
        all_features = np.array(new_features)
        all_labels = encoded_labels
    else:
        # 增量训练：合并新旧标签
        merged_le = merge_labels(le, new_labels)
        new_encoded = merged_le.transform(new_labels)

        # 合并历史特征和新特征
        if history_features is not None and history_labels is not None:
            # 将历史标签转换为新的编码格式
            history_encoded = merged_le.transform(history_labels)
            all_features = np.vstack([history_features, new_features])
            all_labels = np.hstack([history_encoded, new_encoded])
            logger.info(f"合并历史数据({len(history_features)})和新数据({len(new_features)})")
        else:
            all_features = np.array(new_features)
            all_labels = new_encoded

        le = merged_le

    try:
        # 训练贝叶斯模型（MultinomialNB不支持partial_fit，使用全量数据重新训练实现增量效果）
        # clf = MultinomialNB()  # 初始化贝叶斯分类器
        clf.fit(all_features, all_labels)
        logger.info("贝叶斯模型训练完成！开始计算样本准确率，进行模型保存校验... ...")

        imgs_bin_sample = random.sample(imgs_bin, min(10, len(imgs_bin)))
        sample_results = []
        for i in range(len(imgs_bin_sample)):
            res, _ = single_img_bin_predict(imgs_bin_sample[i], clf=clf, le=le)
            if res is not None:
                sample_results.append(res)

        true_samples = sample_results.count(user_id)
        sample_accuracy = true_samples / len(imgs_bin_sample)

        if sample_accuracy >= min_acc or train_first:
            # 保存模型、标签编码器和历史特征
            if train_first:
                logger.info("首次训练，成功录入人脸模型！")
            else:
                logger.info(f"样本准确率为{sample_accuracy * 100}%，验证通过，保存模型")

            joblib.dump(clf, model_path)
            joblib.dump(le, le_path)
            np.save(history_features_path, all_features)
            np.save(history_labels_path, le.inverse_transform(all_labels))  # 保存原始标签
        else:
            logger.error(f"样本准确率为{sample_accuracy * 100}%，验证失败，重新录入人脸！")
            return False

        # 计算训练信息
        elapsed_time = time.time() - start_time
        unique_ids = len(le.classes_)

        logger.info(
            f"训练完成！模型共包含 {unique_ids} 个人脸，"
            f"总样本数: {len(all_features)}，本次新增: {len(new_features)}，"
            f"耗时 {elapsed_time:.2f} 秒。"
        )
        return True

    except Exception as e:
        logger.error(f"训练或保存模型时出错: {e}")
        return False


if __name__ == "__main__":
    # test_img = Image.open("./facedata/test1.jpg")
    # test_img_bin = img_to_bin(test_img)
    # imgs_bin = [test_img_bin]
    # cv2_train(imgs_bin, user_id=2)
    import pickle

    with open("./facedata/face_yingrui_bin_data.pkl", "rb") as f:
        data = pickle.load(f)
        face_id = data['face_id']
        images = data['images']

    print(cv2_train(images, face_id, min_acc=0.5))
