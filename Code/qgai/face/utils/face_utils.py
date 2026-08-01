# ================================
# @File         : face_utils.py
# @Time         : 2025/07/22
# @description  : 人脸处理公共工具类
# ================================

import struct
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import logging
from cryptography.fernet import Fernet


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


def img_to_bin(image):
    """
    将OpenCV格式图像转为二进制数据
    :param image: OpenCV图像格式(numpy array)，BGR通道
    :return: 图像二进制数据
    """
    # 检查输入是否为有效的OpenCV图像
    if not isinstance(image, np.ndarray) or len(image.shape) not in (2, 3):
        raise ValueError("输入必须是OpenCV格式的图像(numpy数组)")

    ret, buf = cv2.imencode('.jpg', image)
    if not ret:
        raise RuntimeError("无法将图像编码为JPEG格式")

    # 将numpy数组转换为字节流
    return buf.tobytes()


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


def extract_features(face_image, embedder):
    """
    使用预训练模型提取人脸特征向量
    """
    # 构建输入 blob
    # face_blob = cv2.dnn.blobFromImage(
    #     face_img, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False
    # )
    feature = embedder.get_feat(face_image)
    feature = np.array(feature, dtype=np.float32).flatten()

    return feature


def normalize_features(features):
    features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    return features


def generate_key():
    return Fernet.generate_key()


def encrypt_feature(key, array):
    """高效加密NumPy数组"""
    # 获取数组基本信息（形状和数据类型）
    shape = array.shape
    dtype = array.dtype

    # 将形状信息转换为字节（使用struct打包）
    # 先写入维度数量，再写入各维度大小
    shape_bytes = struct.pack(f'>I{len(shape)}Q', len(shape), *shape)

    # 写入数据类型信息
    dtype_bytes = dtype.str.encode('utf-8') + b'\x00'  # 以空字节结尾作为分隔符

    # 获取数组数据的字节表示（直接使用内存缓冲区）
    data_bytes = array.tobytes()

    # 组合所有数据
    combined = shape_bytes + dtype_bytes + data_bytes

    # 加密
    fernet = Fernet(key)
    return fernet.encrypt(combined)


def decrypt_feature(key, encrypted_data):
    """高效解密为NumPy数组"""
    # 解密
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_data)

    # 解析形状信息
    # 先读取维度数量（4字节无符号整数，大端序）
    dim_count = struct.unpack('>I', decrypted[:4])[0]
    shape_end = 4 + dim_count * 8  # 每个维度用8字节存储
    shape = struct.unpack(f'>{dim_count}Q', decrypted[4:shape_end])

    # 解析数据类型
    dtype_end = decrypted.find(b'\x00', shape_end)
    dtype = np.dtype(decrypted[shape_end:dtype_end].decode('utf-8'))

    # 解析数据部分
    data_bytes = decrypted[dtype_end + 1:]  # +1跳过空字节分隔符

    # 从字节重建数组
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)


def get_laplacian_matrix(feature, loc=0, scale=1):
    laplacian_matrix = np.random.laplace(loc=loc, scale=scale, size=feature.shape)


if __name__ == "__main__":
    k = generate_key()
    print(k)
