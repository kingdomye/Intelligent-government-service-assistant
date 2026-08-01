# ================================
# @File         : Train.py
# @Time         : 2025/08/02
# @description  : 对输入的图片进行预处理以及模型训练
#                 主函数cv2_train(imgs_bin, user_id, min_acc=0.6)
#                 传入二进制图片数组、用户ID以及准确率阀值
#                 如果模型训练成功则返回True
# ================================
import os
import time
from multiprocessing import Pool
from os import cpu_count
import random
import joblib
import numpy as np

from .Predict import single_face_image_predict
from .model_loader import get_components, load_knn_model
from .utils.face_utils import (
    extract_features, normalize_features
)

components = get_components()
logger = components['logger']
paths = components['paths']
models = components['models']

knn_model_path = paths['knn_model_path']
knn_history_data_path = paths['knn_history_data_path']

# 使用模型组件
rec_model = models['rec_model']


def process_single_image(face_image):
    """
    处理单张图片，用于多进程处理

    :param face_image:      人脸CV图像
    :return:                模型处理后的展平向量
    """
    try:
        features = extract_features(face_image, rec_model)  # 提取人脸特征
        features = normalize_features(features)  # 特征向量归一化

        return features

    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        return None


def get_features(face_images, single_pool_mode=True):
    """
    获取特征向量和对应的标签

    :param single_pool_mode:    指定是否为单线程模式，对于数量少的图片建议使用，默认开启
    :param face_images:         人脸CV图像列表
    :return:                    特征数组和标签数组
    """
    logger.info(f"【提取特征向量】发现{len(face_images)}个图像文件，开始提取图像组的特征向量... ...")
    features = []

    if single_pool_mode:
        for face_image in face_images:
            feature = process_single_image(face_image)
            if feature is not None:
                features.append(feature)
    else:
        # 多线程处理图像
        num_processes = min(cpu_count(), len(face_images))
        with Pool(num_processes) as pool:
            for feature in pool.imap(process_single_image, face_images):
                if feature is not None:
                    features.append(feature)

    return np.array(features)


def load_all_data(history_data_path, new_features, new_labels):
    if os.path.exists(history_data_path):
        # 加载历史特征和标签
        history_data = joblib.load(history_data_path)
        all_features = np.vstack([history_data['features'], new_features])  # 合并特征
        all_labels = history_data['labels'] + new_labels  # 合并标签
    else:
        # 首次训练，直接使用新数据
        all_features = new_features
        all_labels = new_labels

    return all_features, all_labels


def sample_check(face_images, user_id, min_acc=0.6, classifier=None, sample_nums=10):
    face_images_sample = random.sample(face_images, min(sample_nums, len(face_images)))
    sample_results = []
    for i in range(len(face_images_sample)):
        res, _ = single_face_image_predict(face_images_sample[i], classifier)
        if res is not None:
            sample_results.append(res)

    true_samples = sample_results.count(user_id)
    sample_accuracy = true_samples / len(face_images_sample)

    if sample_accuracy >= min_acc:
        logger.info(f"【模型样本校验成功】样本准确率为{sample_accuracy * 100}%，验证通过，保存模型")
        return True
    else:
        logger.error(f"【模型样本校验失败】样本准确率为{sample_accuracy * 100}%，验证失败，重新录入人脸！")
        return False


def cv2_train(face_images, user_id, min_acc=0.6):
    """

    :param face_images:     人脸CV图像列表
    :param user_id:         用户对应的id
    :param min_acc:         准确率阀值
    :return:                训练评估成果，True成功，False失败
    """
    start_time = time.time()

    clf = load_knn_model(knn_model_path)
    new_features = get_features(face_images)
    new_labels = [user_id] * len(new_features)
    all_features, all_labels = load_all_data(knn_history_data_path, new_features, new_labels)

    try:
        clf.fit(all_features, all_labels)
        logger.info("【模型样本校验】计算样本准确率，进行模型保存校验... ...")

        check_success = sample_check(face_images, user_id, min_acc=min_acc, classifier=clf)
        if check_success:
            joblib.dump(clf, knn_model_path)
            joblib.dump({
                'features': all_features,
                'labels': all_labels
            }, knn_history_data_path)
        else:
            return False

        # 计算训练信息
        elapsed_time = time.time() - start_time
        logger.info(
            f"【训练完成】模型共包含 {len(clf.classes_)} 个人脸，"
            f"总样本数: {len(all_features)}，本次新增: {len(new_features)}，"
            f"耗时 {elapsed_time:.2f} 秒。"
        )
        return True

    except Exception as e:
        logger.error(f"训练或保存模型时出错: {e}")
        return False


if __name__ == "__main__":
    # test_img = cv2.imread("./facedata/dalin1.jpg")
    # test_img_bin = img_to_bin(test_img)
    # face = face_fetcher(test_img_bin)
    #
    # f = process_single_image(face)
    # print(f)

    # print(f)
    # imgs_bin = [test_img_bin]
    # cv2_train(imgs_bin, user_id=2)
    import pickle
    from .face_fetcher import face_fetcher

    with open("./facedata/face_shen_bin_data.pkl", "rb") as f:
        data = pickle.load(f)
        face_id = data['face_id']
        images = data['images']

    aface_images = []
    for image in images:
        aface_images.append(face_fetcher(image))

    cv2_train(aface_images, face_id)
