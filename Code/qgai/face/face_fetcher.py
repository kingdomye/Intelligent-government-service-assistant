# ================================
# @File         : face_fetcher.py
# @Time         : 2025/07/30
# @description  : 人脸检测函数
#                 输入：二进制、三通道图像数据
#                 输出：人脸CV图像，没有人脸则返回None
# ================================

import cv2
import numpy as np
from .model_loader import get_components
from .utils.face_utils import img_to_bin

# 加载模型组件
components = get_components()
models = components['models']
logger = components['logger']
face_detectors = models['detectors']

# 1、尺寸阈值设置 (宽度, 高度)
# 小于该阀值的图片会自动调整图片大小
THRESHOLD_SIZE = (800, 800)

# 2、人脸抓取参数（ >1 ）
# 数值越大，识别准确率越低，速度越快
# 数值越小，识别准确率越高，速度越慢（自己看着办）
SCALE_FACTOR = 1.1

# 3、图片旋转参数
ROTATION_STEP = 15


def rotate_bound(image, angle):
    """
    旋转图像并保持完整内容不被裁剪

    :param image:   CV图像
    :param angle:   旋转角度
    :return:        旋转后的CV格式的图像
    """
    if angle % 360 == 0:
        return image

    # 获取图像尺寸并确定中心点
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 获取旋转矩阵（应用负角度实现顺时针旋转）
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算旋转后图像的新尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵以补偿平移
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 执行旋转并返回结果
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])


def new_resize(image):
    """
    智能调整CV图像，输出正方形的图片，小于尺寸阀值的用皮肤色填充

    :param image:   CV图像
    :return:        处理后更容易识别到人脸的CV图像
    """
    orig_height, orig_width = image.shape[:2]
    max_width = max(orig_width, THRESHOLD_SIZE[0])
    max_height = max(orig_height, THRESHOLD_SIZE[1])
    canvas_size = max(max_width, max_height)

    canvas = np.full((canvas_size, canvas_size, 3), 0, dtype=np.uint8)

    x_offset = (canvas_size - orig_width) // 2
    y_offset = (canvas_size - orig_height) // 2
    canvas[y_offset:y_offset + orig_height, x_offset:x_offset + orig_width] = image

    return canvas


def detect_faces_in_image(image):
    """
    在单张图像中检测人脸

    :param image:   一张CV图像
    :return:        一个包含人脸位置数据的列表，若没有识别到人脸，则返回None
    """
    detected_faces = []
    for detector in face_detectors:
        faces = detector.detectMultiScale(
            image,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected_faces.extend(faces)

    return None if len(detected_faces) == 0 else detected_faces


def face_fetcher(image_binary):
    """
    检测图中的人脸区域，返回区域边缘坐标
    若没有在图中发现人脸，则返回None

    :param image_binary:    图像二进制格式
    :return:                人脸CV图像
    """
    # 解码图像
    original_image = cv2.imdecode(
        np.frombuffer(image_binary, np.uint8),
        cv2.IMREAD_COLOR
    )
    # 首次检测
    detected_faces = detect_faces_in_image(original_image)

    # 首次检测就发现人脸，返回图中最大的人脸
    if detected_faces is not None:
        (x, y, w, h) = max(detected_faces, key=lambda f: f[2] * f[3])
        face_area = original_image[y:y + h, x:x + w]
        return face_area
    else:
        processed_image = new_resize(original_image)

        rotation_range = range(0, 360 - ROTATION_STEP, ROTATION_STEP)
        for angle in rotation_range:
            rotated_image = rotate_bound(processed_image, angle)
            detected_faces = detect_faces_in_image(rotated_image)
            if detected_faces is not None:
                (x, y, w, h) = max(detected_faces, key=lambda f: f[2] * f[3])
                face_area = rotated_image[y:y + h, x:x + w]
                return face_area

        logger.warning("图片中未检测到人脸！")
        return None


if __name__ == '__main__':
    pil_image = cv2.imread("./facedata/testt.jpg")
    image_bin = img_to_bin(pil_image)

    import time
    start = time.time()
    face_a = face_fetcher(image_bin)
    end = time.time()
    print("人脸捕捉总共耗时：", end - start)

    cv2.imshow("face_a", face_a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
