# ================================
# @File         : Predict.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 输入图片对模型进行id匹配
# ================================
import io
from io import BytesIO

import cv2
import logging
import joblib
import numpy as np
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化特征提取模型和SVM分类器
try:
    # 加载预训练的人脸特征提取模型（OpenFace）
    model_path = "nn4.small2.v1.t7"
    if not cv2.os.path.exists(model_path):
        logger.error(f"特征提取模型文件不存在: {model_path}")
        logger.info("请从 https://cmusatyalab.github.io/openface/models-and-accuracies/ 下载模型")
        exit()
    embedder = cv2.dnn.readNetFromTorch(model_path)

    # 读取训练模型和标签编码器
    trainer_dir = './face_trainer_svm'
    svm_model_path = cv2.os.path.join(trainer_dir, 'svm_model.pkl')
    le_path = cv2.os.path.join(trainer_dir, 'label_encoder.pkl')

    if not cv2.os.path.exists(svm_model_path) or not cv2.os.path.exists(le_path):
        logger.error(f"训练模型文件不存在: {svm_model_path} 或 {le_path}")
        logger.error("请先运行训练程序生成模型")
        exit()

    # 加载SVM模型和标签编码器
    clf = joblib.load(svm_model_path)
    le = joblib.load(le_path)
    logger.info(f"成功加载训练模型: {svm_model_path} 和 {le_path}")
except Exception as e:
    logger.error(f"初始化模型失败: {e}")
    exit()

# 初始化人脸检测器，使用多个检测器提高鲁棒性
detector_paths = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
]

detectors = []
for detector_path in detector_paths:
    if not cv2.os.path.exists(detector_path):
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
logger.info("人脸检测器初始化完毕！")


def bin_to_image_array(bin_data):
    """
    将二进制图像数据转换为三维矩阵

    :param bin_data: 图像的二进制数据
    :return: 转换后的三位矩阵，如果转换失败则返回None
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
    """预处理人脸图像以适配特征提取模型"""
    # 与训练时保持一致的预处理
    face_img = cv2.equalizeHist(face_img)
    # 调整为特征提取模型需要的大小（96x96）
    face_img = cv2.resize(face_img, (96, 96))
    # 转为3通道（模型要求输入为3通道）
    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    return face_img


def extract_features(face_img):
    """使用预训练模型提取人脸特征向量"""
    # 构建输入 blob
    face_blob = cv2.dnn.blobFromImage(
        face_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
    )
    embedder.setInput(face_blob)
    return embedder.forward()  # 返回128维特征向量


def single_img_bin_predict(img_bin, min_acc=0.6):
    img = bin_to_image_array(img_bin)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = None
    for detector in detectors:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            break

    if faces is not None:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 提取人脸区域并预处理
            face_roi = gray[y:y + h, x:x + w]
            face_roi = preprocess_face(face_roi)

            # 提取人脸特征
            features = extract_features(face_roi)
            features = features.flatten().reshape(1, -1)  # 调整形状以适应SVM输入

            # 识别人脸
            predictions = clf.predict_proba(features)
            max_prob = np.max(predictions)
            predicted_class = np.argmax(predictions)

            # 转换为原始标签
            username = le.inverse_transform([predicted_class])[0]
            confidence_text = f"{max_prob * 100:.1f}%"

            # 设置一个置信度阈值，低于此阈值的视为未知
            if max_prob < min_acc:
                logger.info("识别失败，请重新输入人脸数据！")
                return False

            return username, confidence_text
    return None


def cv2_predict(imgs_bin, min_acc=0.8):
    pass


if __name__ == "__main__":
    def img_to_bin(image):
        img_bin = io.BytesIO()
        image.save(img_bin, format='JPEG')
        image_bytes = img_bin.getvalue()

        return image_bytes

    test_img = Image.open("./facedata/test.jpg")
    test_img_bin = img_to_bin(test_img)

    pred, confidence = single_img_bin_predict(test_img_bin)
    print(pred)
    print(confidence)
