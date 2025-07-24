# ================================
# @File         : face_utils.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 人脸处理公共工具类
# ================================

import io
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import cv2


def bin_to_image(bin_data):
    """
    将二进制图像数据转换为PIL图像对象
    """
    try:
        image_stream = BytesIO(bin_data)
        img = Image.open(image_stream)
        return img.copy()
    except Exception as e:
        logging.error(f"图像转换失败: {str(e)}")
        return None


def bin_to_image_array(bin_data):
    """
    将二进制图像数据转换为三维矩阵
    """
    try:
        image_stream = BytesIO(bin_data)
        img = Image.open(image_stream)
        img = np.array(img)
        return img.copy()
    except Exception as e:
        logging.error(f"图像转换失败: {str(e)}")
        return None


def preprocess_face(face_img):
    """
    预处理人脸图像以适配特征提取模型
    """
    # 直方图均衡化增强对比度
    face_img = cv2.equalizeHist(face_img)
    # 调整为特征提取模型需要的大小（96x96）
    face_img = cv2.resize(face_img, (96, 96))
    # 转为3通道（模型要求输入为3通道）
    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    return face_img


def extract_features(face_img, embedder):
    """
    使用预训练模型提取人脸特征向量
    """
    # 构建输入 blob
    face_blob = cv2.dnn.blobFromImage(
        face_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
    )
    embedder.setInput(face_blob)
    return embedder.forward()  # 返回128维特征向量


def detect_largest_face(gray_img, detectors):
    """
    从灰度图中检测最大的人脸区域
    """
    faces = None
    for detector in detectors:
        faces = detector.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            break

    if faces is None or len(faces) == 0:
        return None

    # 选择最大的人脸
    return max(faces, key=lambda f: f[2] * f[3])


def normalize_features(features):
    """
    特征归一化处理（确保非负）
    """
    features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    return np.clip(features, 0, None)  # 确保没有负值


def img_to_bin(image):
    """
    将PIL图像转换为二进制数据
    """
    img_bin = io.BytesIO()
    image.save(img_bin, format='JPEG')
    return img_bin.getvalue()


def cv_img_to_bin(image):
    """
    将cv捕捉的图像转成二进制
    :param image: OpenCV读取的图像（numpy数组格式）
    :return: 图像的二进制数据，如果转换失败则返回None
    """
    try:
        # 检查输入是否为有效的图像
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("输入不是有效的OpenCV图像")

        # 将图像编码为JPEG格式，返回值为(retval, buf)
        # retval为布尔值，表示编码是否成功
        # buf为包含编码后数据的字节流
        retval, buffer = cv2.imencode('.jpg', image)

        if not retval:
            raise RuntimeError("图像编码失败")

        # 将numpy数组转换为二进制数据
        binary_data = buffer.tobytes()

        return binary_data
    except Exception as e:
        print(f"图像转换为二进制时发生错误: {str(e)}")
        return None


if __name__ == "__main__":
    img = cv2.imread("./facedata/test1.jpg")
    img_bin = cv_img_to_bin(img)
    print(img_bin)
