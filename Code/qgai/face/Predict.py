# ================================
# @File         : Predict.py
# @Time         : 2025/08/02
# @description  : 用于人脸预测，主函数：cv2_predict()
#                 输入一组二进制图像列表、置信度阀值
#                 输出预测的用户ID
# ================================

import time
import numpy as np

from .model_loader import get_components, load_knn_model
from .utils.face_utils import (
    extract_features, normalize_features
)

# 获取组件
components = get_components()
logger = components['logger']
paths = components['paths']
models = components['models']

# 加载特征提取模型
rec_model = models['rec_model']
knn_model_path = paths['knn_model_path']

# KNN最小距离阀值，当所有检测结果都大于该阈值时，判别为人脸未录入
# 参考值：录入的人脸（2.8），未录入的人脸（4.2）
DISTANCE_THRESHOLD = 3.600


def single_face_image_predict(face_image, classifier):
    """
    单张照片预测函数，返回用户ID和对应的置信度
    预测成功则返回 (用户ID)，失败返回 (None)
    """
    # 提取并处理特征
    try:
        features = extract_features(face_image, rec_model)
        features = normalize_features(features)
        features = features.reshape(1, -1)  # 转换为二维数组（符合模型输入要求）
    except Exception as e:
        logger.error(f"【特征提取失败】: {e}")
        return None, None

    # 识别人脸
    try:
        distances, indices = classifier.kneighbors(features)
        min_distance = distances[0][0]  # 最小距离

        predictions = classifier.predict(features)
        predicted_id = predictions[0]  # 预测的用户ID
        confidence_text = max(classifier.predict_proba(features)[0])

        # KNN拒识机制，确保距离不超过阈值
        if min_distance >= DISTANCE_THRESHOLD:
            logger.warning(f"【未录入人脸】用户 {predicted_id} 的最小距离 {min_distance:.4f} ≥ {DISTANCE_THRESHOLD}")
            return None, None

        return predicted_id, confidence_text

    except Exception as e:
        logger.error(f"【预测过程出错】: {e}")
        return None, None


def get_predictions(face_images, classifier):
    valid_results = []
    label_confidences = {}

    for face_image in face_images:
        # 直接调用处理函数，无需通过进程池
        username, confidence_text = single_face_image_predict(face_image, classifier)
        if username is not None and confidence_text is not None:
            # 转换类型并添加到有效结果列表
            username_str = str(username)
            confidence = float(confidence_text)
            valid_results.append((username_str, confidence))

            # 同时更新标签-置信度字典
            if username_str not in label_confidences:
                label_confidences[username_str] = []
            label_confidences[username_str].append(confidence)

    return valid_results, label_confidences


def cv2_predict(face_images, min_acc=0.96):
    """
    批量处理图像并返回预测结果

    :param face_images:         一组CV人脸图片列表
    :param min_acc:             置信度阀值，低于阀值的数据将不被认可
    :return:                    预测的用户ID
    """

    # 初始化标签-置信度列表字典
    logger.info(f"检测到{len(face_images)}个图像数据，开始筛选置信度高于阀值{min_acc}的有效人脸数据... ...")
    start_time = time.time()

    classifier = load_knn_model(knn_model_path)
    valid_results, label_confidences = get_predictions(face_images, classifier)

    end_time = time.time()
    logger.info(f"筛选出 {len(valid_results)} 个有效人脸数据，用时{end_time-start_time:.4f}s，开始预测")

    # 筛选后没有人脸
    if len(valid_results) == 0:
        return None

    # 计算其他统计信息
    count = {label: len(conf_list) for label, conf_list in label_confidences.items()}       # 每个标签预测的数量字典
    best_label = max(count, key=count.get)                                                  # 数量最多的标签
    best_confidence = label_confidences[best_label]                                         # 数量最多的标签对应的置信度列表
    best_confidence_avg = np.mean(best_confidence)                                          # 置信度列表求平均值

    logger.info(f"【预测结果】用户ID：{best_label} ，置信度：{best_confidence_avg * 100:.4f}%")

    if best_confidence_avg < min_acc:
        return None

    return best_label


if __name__ == "__main__":
    # 单张照片测试
    import cv2
    from .face_fetcher import face_fetcher
    from .utils.face_utils import img_to_bin

    test_img = cv2.imread("./facedata/51.jpg")  # 读取图像
    test_img = img_to_bin(test_img)  # 转二进制
    aface_image = face_fetcher(test_img)  # 获取人脸CV图像

    clf = load_knn_model(knn_model_path)
    name, conf = single_face_image_predict(aface_image, clf)  # 传入函数

    if conf is None:
        conf = 0
    print(f"预测为{name}, 置信度为{conf * 100:.4f}%")
    # cv2.imshow("", aface_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # 照片组
    # import pickle
    # from face_fetcher import face_fetcher
    #
    # with open("./facedata/face_xiaolin_bin_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     face_id = data['face_id']
    #     images = data['images']
    #
    # a_face_images = []
    # for image in images:
    #     a_face_images.append(face_fetcher(image))
    #
    # cv2_predict(a_face_images, min_acc=0.96)
