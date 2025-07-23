# ================================
# @File         : Train.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 对输入的图片进行预处理以及模型训练
# ================================

import logging
import os
import random
import time
import io
from io import BytesIO
from multiprocessing import Pool
from os import cpu_count

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 确保训练结果目录存在
trainer_dir = './face_trainer_svm'
os.makedirs(trainer_dir, exist_ok=True)

# 模型和标签编码器路径
model_path = os.path.join(trainer_dir, 'svm_model.pkl')
le_path = os.path.join(trainer_dir, 'label_encoder.pkl')

# 人脸数据路径 - 确保这是一个目录
face_data_path = './facedata'
if not os.path.exists(face_data_path):
    logger.error(f"人脸数据路径不存在: {face_data_path}")
    exit()
if not os.path.isdir(face_data_path):
    logger.error(f"人脸数据路径不是一个目录: {face_data_path}")
    exit()

# 初始化检测器和特征提取模型
try:
    # 加载预训练的人脸特征提取模型（OpenFace）
    model_path = "nn4.small2.v1.t7"
    if not os.path.exists(model_path):
        logger.error(f"特征提取模型文件不存在: {model_path}")
        logger.info("请从 https://cmusatyalab.github.io/openface/models-and-accuracies/ 下载模型")
        exit()
    embedder = cv2.dnn.readNetFromTorch(model_path)

    # 加载多个检测器以提高检测鲁棒性
    detector_paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    ]

    detectors = []
    for detector_path in detector_paths:
        if not os.path.exists(detector_path):
            logger.warning(f"检测器文件不存在: {detector_path}")
            continue

        detector = cv2.CascadeClassifier(detector_path)
        if not detector.empty():
            detectors.append(detector)
        else:
            logger.warning(f"无法加载检测器: {detector_path}")

    if not detectors:
        logger.error("无法加载任何人脸检测器")
        exit()
except Exception as e:
    logger.error(f"初始化模型/检测器失败: {e}")
    exit()


def bin_to_image(bin_data):
    """
    将二进制图像数据转换为PIL图像对象

    :param bin_data: 图像的二进制数据
    :return: 转换后的PIL图像对象，如果转换失败则返回None
    """
    try:
        image_stream = BytesIO(bin_data)
        img = Image.open(image_stream)

        return img.copy()
    except Exception as e:
        logging.error(f"图像转换失败: {str(e)}")

        return None


def preprocess_face(face_img):
    """
    预处理人脸图像以适配特征提取模型

    :param face_img: 脸部图数据
    :return: 能够适用于特征提取的图像数据
    """
    # 直方图均衡化增强对比度
    face_img = cv2.equalizeHist(face_img)
    # 调整为特征提取模型需要的大小（96x96）
    face_img = cv2.resize(face_img, (96, 96))
    # 转为3通道（模型要求输入为3通道，即使是灰度图也需要复制通道）
    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

    return face_img


def extract_features(face_img):
    """
    使用预训练模型提取人脸特征向量

    :param face_img:
    :return: 返回128维特征向量
    """
    # 构建输入 blob
    face_blob = cv2.dnn.blobFromImage(
        face_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
    )
    embedder.setInput(face_blob)

    return embedder.forward()


def process_single_image(face_img_bin):
    """
    处理单张图片，用于多进程处理

    :param face_img_bin:
    :return: 模型处理后的展平向量
    """
    try:
        face_img = bin_to_image(face_img_bin)
        # 图片转为灰度图
        face_img = face_img.convert("L")
        img_numpy = np.array(face_img, dtype='uint8')

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
        face_img = img_numpy[y:y+h, x:x+w]

        face_img = preprocess_face(face_img)
        features = extract_features(face_img)

        return features.flatten()           # 展平特征向量

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


def load_existing_data():
    """
    加载已有的模型和标签编码器

    :return:
    """
    try:
        if os.path.exists(model_path) and os.path.exists(le_path):
            logger.info("发现已存在的模型，将在其基础上进行增量训练")
            clf = joblib.load(model_path)
            le = joblib.load(le_path)

            # 验证模型是否已训练
            try:
                # 尝试访问classes_属性来检查模型是否已训练
                if hasattr(clf, 'classes_'):
                    return clf, le
                else:
                    logger.warning("已存在的模型未经过训练，将创建新模型")
                    return None, None
            except NotFittedError:
                logger.warning("已存在的模型未经过训练，将创建新模型")
                return None, None
        else:
            logger.info("未发现已存在的模型，将创建新模型")
            return None, None
    except Exception as e:
        logger.error(f"加载已有模型时出错: {e}，将创建新模型")
        return None, None


def merge_labels(le, new_labels):
    """
    合并已有的标签和新标签，确保所有标签都被正确编码

    :param le:
    :param new_labels:
    :return:
    """
    # 获取已有的所有标签
    existing_labels = list(le.classes_)

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
    logger.info('开始提取特征并训练SVM模型，请耐心等待....')

    new_features, new_labels = get_feature_and_labels(imgs_bin, user_id)

    if not new_features or not new_labels:
        logger.error("没有有效的训练数据，无法进行训练")
        exit()

    try:
        # 加载已有的模型和标签编码器
        clf, le = load_existing_data()

        # 如果没有已存在的模型，则创建新的
        if clf is None or le is None:
            logger.info("")
            le = LabelEncoder()
            encoded_labels = le.fit_transform(new_labels)
            clf = SVC(kernel='linear', probability=True)
            clf.fit(new_features, encoded_labels)
            total_samples = len(new_features)
            unique_ids = len(np.unique(new_labels))
        else:
            # 合并标签，处理新出现的人脸ID
            merged_le = merge_labels(le, new_labels)

            # 对新标签进行编码
            new_encoded_labels = merged_le.transform(new_labels)

            # 获取旧模型的训练数据（如果可能）
            # 注意：SVM不直接存储训练数据，所以我们需要保留历史数据或重新加载
            # 这里我们假设只使用新数据进行增量训练
            # 对于更优的增量训练，应该保留历史特征和标签
            clf.fit(new_features, new_encoded_labels)

            # 更新标签编码器
            le = merged_le

            # 计算总样本数和唯一ID数
            total_samples = len(new_features) + len(le.classes_)  # 估计值
            unique_ids = len(le.classes_)

        # 保存更新后的模型和标签编码器
        joblib.dump(clf, model_path)
        joblib.dump(le, le_path)

        # 计算训练信息
        elapsed_time = time.time() - start_time
        logger.info(
            f"训练完成！模型共包含 {unique_ids} 个人脸，"
            f"本次新增 {len(new_features)} 个样本，"
            f"耗时 {elapsed_time:.2f} 秒。"
            f"模型已保存到 {model_path}，标签编码器已保存到 {le_path}"
        )

        print(new_features, new_labels)
        test_samples = random.sample(new_features, 1)
        test_samples = []

    except Exception as e:
        logger.error(f"训练或保存模型时出错: {e}")
        return False


if __name__ == "__main__":
    def img_to_bin(image):
        img_bin = io.BytesIO()
        image.save(img_bin, format='JPEG')
        image_bytes = img_bin.getvalue()

        return image_bytes

    img1 = Image.open("./facedata/test.jpg")
    img2 = Image.open("./facedata/test1.jpg")
    imgs_bin = [img_to_bin(img1), img_to_bin(img2)]
    cv2_train(imgs_bin, 1)
