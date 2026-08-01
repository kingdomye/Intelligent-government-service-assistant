# ================================
# @File         : KNN.py
# @Time         : 2025/08/02
# @description  : 基于MindSpore的KNN分类器
# ================================

from mindspore import ops, nn, Tensor
import numpy as np


class KNNClassifier(nn.Cell):
    def __init__(self, n_neighbors=5, weights='uniform'):
        super(KNNClassifier, self).__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights  # 'uniform'或'distance'
        self._fit_X = None  # 训练特征
        self._fit_y = None  # 训练标签
        self._fit_X_tensor = None  # MindSpore格式的训练特征
        self.classes_ = None  # 类别标签

    def fit(self, X, y):
        """拟合模型，存储训练数据和标签"""
        self._fit_X = np.asarray(X, dtype=np.float32)
        self._fit_X_tensor = Tensor(self._fit_X)
        self._fit_y = np.asarray(y)
        self.classes_ = np.unique(y)  # 存储所有唯一类别
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """查找输入样本的最近邻（与sklearn接口一致）"""
        if self._fit_X is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        X = np.asarray(X, dtype=np.float32).reshape(1, -1)
        n_queries = X.shape[0]
        n_samples = self._fit_X.shape[0]

        all_distances = []
        all_indices = []

        for i in range(n_queries):
            x = Tensor(X[i:i + 1, :])
            x_tile = ops.tile(x, (n_samples, 1))
            square_diff = ops.square(x_tile - self._fit_X_tensor)
            square_dist = ops.sum(square_diff, 1)
            distances = ops.sqrt(square_dist)

            values, indices = ops.topk(-distances, n_neighbors)
            nn_distances = (-values).asnumpy()
            nn_indices = indices.asnumpy()

            all_distances.append(nn_distances)
            all_indices.append(nn_indices)

        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)

        if return_distance:
            return all_distances, all_indices
        else:
            return all_indices

    def predict(self, X):
        """预测输入样本的类别（与sklearn接口一致）"""
        distances, indices = self.kneighbors(X, return_distance=True)
        n_queries = X.shape[0]

        # 获取近邻的标签
        neighbor_labels = self._fit_y[indices]

        if self.weights == 'uniform':
            y_pred = []
            for labels in neighbor_labels:
                # 使用numpy找到出现次数最多的标签
                unique_labels, counts = np.unique(labels, return_counts=True)
                # 找到计数最大的索引（处理可能的平局）
                max_count_idx = np.argmax(counts)
                y_pred.append(unique_labels[max_count_idx])
            y_pred = np.array(y_pred)

        elif self.weights == 'distance':
            # 按距离加权投票（距离越近权重越大）
            y_pred = []
            for i in range(n_queries):
                # 避免除以零，添加极小值
                weights = 1.0 / (distances[i] + 1e-10)
                # 对每个类别计算加权和
                class_weights = {}
                for cls in self.classes_:
                    mask = (neighbor_labels[i] == cls)
                    class_weights[cls] = np.sum(weights[mask])
                # 选择权重最大的类别
                y_pred.append(max(class_weights, key=class_weights.get))
            y_pred = np.array(y_pred)
        else:
            raise ValueError("weights参数必须是'uniform'或'distance'")

        return y_pred

    def predict_proba(self, X):
        """预测输入样本属于每个类别的概率（与sklearn接口一致）"""
        distances, indices = self.kneighbors(X, return_distance=True)
        n_queries = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_queries, n_classes))

        # 获取近邻的标签
        neighbor_labels = self._fit_y[indices]

        for i in range(len(neighbor_labels)):
            if self.weights == 'uniform':
                # 等权重概率计算：直接映射标签到其在classes_中的索引
                # 避免使用np.searchsorted导致的索引越界问题
                label_indices = []
                for label in neighbor_labels[i]:
                    # 找到标签在classes_中的索引（因classes_是unique的，必存在）
                    idx = np.where(self.classes_ == label)[0][0]
                    label_indices.append(idx)
                counts = np.bincount(label_indices, minlength=n_classes)
                proba[i] = counts / self.n_neighbors
            elif self.weights == 'distance':
                # 按距离加权的概率计算
                weights = 1.0 / (distances[i] + 1e-10)
                for j, cls in enumerate(self.classes_):
                    mask = (neighbor_labels[i] == cls)
                    proba[i, j] = np.sum(weights[mask]) / np.sum(weights)
            else:
                raise ValueError("weights参数必须是'uniform'或'distance'")

        return proba


if __name__ == '__main__':
    pass
    # ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    #
    # # 1. 创建带标签的训练数据和测试数据
    # X_train = np.array([[i / 100, (i * 2) / 100, (i * 3) / 100] for i in range(128)], dtype=np.float32)
    # y_train = np.array([i % 4 for i in range(128)])  # 0-3循环的标签
    #
    # X_test = np.array([
    #     [0.1, 0.2, 0.3],  # 接近索引10，标签应为10%4=2
    #     [0.5, 1.0, 1.5],  # 接近索引50，标签应为50%4=2
    #     [1.2, 2.4, 3.6]  # 接近索引120，标签应为120%4=0
    # ], dtype=np.float32)
    #
    # # 2. 创建KNN分类器并拟合
    # knn = KNNClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train)
    #
    # # 3. 预测
    # print("测试样本的预测类别:", knn.predict(X_test))
    # print("\n测试样本的类别概率:")
    # print(knn.predict_proba(X_test))
    # print("\n类别标签:", knn.classes_)
    # import pickle
    # import cv2
    # from face_fetcher import face_fetcher
    # from model_loader import get_components
    # from utils.face_utils import normalize_features, img_to_bin, extract_features
    #
    # components = get_components()
    # models = components['models']
    # rec_model = models['rec_model']
    #
    # with open("./facedata/face_shen_bin_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     face_id = data['face_id']
    #     images = data['images']
    #
    # aface_images = []
    # for image in images:
    #     aface_images.append(face_fetcher(image))
    #
    # features = get_features(aface_images)
    # features = normalize_features(features)
    # labels = [face_id] * len(features)
    #
    # knn = KNNClassifier()
    # s = time.time()
    # knn.fit(features, labels)
    # e = time.time()
    # print(f"Mindspore KNN Train finished 用时 {e-s}s")
    #
    # test_img = cv2.imread("./facedata/gou.jpg")  # 读取图像
    # test_img = img_to_bin(test_img)  # 转二进制
    # aface_image = face_fetcher(test_img)  # 获取人脸CV图像
    # f = extract_features(aface_image, rec_model)
    # f = normalize_features(f)
    #
    # res = knn.predict(f)
    # f = f.reshape(1, -1)
    # res1 = knn.predict_proba(f)
    # print("Mindspore KNN predict result:", res)
    # print("Mindspore KNN predict_proba result:", res1)
