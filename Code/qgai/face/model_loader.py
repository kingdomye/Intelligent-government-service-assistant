# ================================
# @File         : model_loader.py
# @Time         : 2025/07/24
# @description  : 人脸贝叶斯模型初始化器，负责配置路径、日志和加载基础模型组件
# ================================

import logging
import os
import time
from pathlib import Path
import cv2
import joblib
from .utils.KNN import KNNClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FaceModelLoader:
    """人脸贝叶斯模型初始化器，负责配置路径、日志和加载基础模型组件"""

    def __init__(self):
        # 初始化路径
        runtime_dir = Path(os.getenv("QGAI_RUNTIME_DIR", "runtime"))
        self.trainer_dir = runtime_dir / "face_models"
        self.face_data_path = runtime_dir / "facedata"
        self.insightface_model_path = self.trainer_dir / "face_fetcher.pkl"

        # 模型和数据文件路径
        self.knn_model_path = self.trainer_dir / "knn_model.pkl"
        self.knn_history_data_path = self.trainer_dir / "knn_history_data.pkl"

        # 模型组件
        self.rec_model = None
        self.detectors = []

        # 初始化工作流
        self._setup_directories()
        self._load_rec_model()
        self._load_detectors()

    def _setup_directories(self):
        """确保训练结果目录存在"""
        try:
            self.trainer_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建训练目录失败: {e}")
            raise

    def _validate_data_path(self):
        """验证人脸数据路径是否有效"""
        if not self.face_data_path.exists():
            logger.error(f"人脸数据路径不存在: {self.face_data_path}")
            raise FileNotFoundError(f"人脸数据路径不存在: {self.face_data_path}")

        if not self.face_data_path.is_dir():
            logger.error(f"人脸数据路径不是一个目录: {self.face_data_path}")
            raise NotADirectoryError(f"人脸数据路径不是一个目录: {self.face_data_path}")

    def _load_rec_model(self):
        """加载预训练的人脸特征提取模型（InsightFace）"""
        try:
            if not self.insightface_model_path.exists():
                logger.error("InsightFace特征提取模型不存在: %s", self.insightface_model_path)
                raise FileNotFoundError(self.insightface_model_path)

            self.rec_model = joblib.load(str(self.insightface_model_path))

        except Exception as e:
            logger.error(f"加载特征提取模型失败: {e}")
            raise

    def _load_detectors(self):
        """加载多个检测器以提高检测鲁棒性"""
        detector_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        ]

        for detector_path in detector_paths:
            if not Path(detector_path).exists():
                logger.warning(f"检测器文件不存在: {detector_path}")
                continue

            detector = cv2.CascadeClassifier(detector_path)
            if not detector.empty():
                self.detectors.append(detector)
            else:
                logger.warning(f"无法加载检测器: {detector_path}")

        if not self.detectors:
            logger.error("无法加载任何人脸检测器")
            raise RuntimeError("无法加载任何人脸检测器")

    def get_components(self):
        """获取所有初始化的组件"""
        return {
            'logger': logger,
            'paths': {
                'trainer_dir': self.trainer_dir,
                'knn_model_path': self.knn_model_path,
                "knn_history_data_path": self.knn_history_data_path,
            },
            'models': {
                'rec_model': self.rec_model,
                'detectors': self.detectors,
            }
        }


# 全局单例组件，确保只初始化一次
_global_loader = None
_global_components = None


def get_components():
    global _global_loader, _global_components
    if _global_components is None:
        start_time = time.time()
        _global_loader = FaceModelLoader()
        _global_components = _global_loader.get_components()
        end_time = time.time()
        logger.info(f"人脸识别Agent组件加载完毕！共耗时{end_time - start_time:.4f}s")
    return _global_components


def load_knn_model(model_path):
    try:
        if Path(model_path).exists():
            knn_clf = joblib.load(str(model_path))
            return knn_clf
        else:
            return KNNClassifier()
    except Exception as e:
        logger.error("初始化KNN模型失败：%s", e)
        raise


if __name__ == '__main__':
    components = get_components()
    for k, v in components.items():
        print(k, v)
