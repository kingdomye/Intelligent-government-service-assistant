# ================================
# @File         : face_fetcher.py
# @Time         : 2025/07/24
# @Author       : Yingrui Chen
# @description  : 人脸捕捉，测试使用
# ================================
import pickle

import cv2
import os
from utils.face_utils import cv_img_to_bin

save_dir = "../facedata"
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"创建保存目录: {save_dir}")
    except Exception as e:
        print(f"无法创建保存目录: {e}")
        exit()

# 调用摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 加载人脸检测器
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_detector.empty():
    print("无法加载人脸检测器")
    exit()

face_id = input('请输入用户ID: ')
print('开始采集人脸数据，请看向摄像头...')

count = 0
imgs_bin = []

while True:
    # 从摄像头读取图片
    success, img = cap.read()
    if not success:
        print("无法获取图像")
        break

    # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 绘制矩形框
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # 保存图像
        # img_path = os.path.join(save_dir, f"{face_id}_{count}.jpg")
        try:
            img_bin = cv_img_to_bin(img)
            imgs_bin.append(img_bin)
            # cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            print(f"已保存 {count} 张图片")
        except Exception as e:
            print(f"保存图片失败: {e}")

        cv2.imshow('image', img)

    # 按键处理
    k = cv2.waitKey(1)
    if k == 27:  # ESC键退出
        break
    elif count >= 50:  # 采集10张后退出
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
print(f"采集完成，共保存 {count} 张图片")

# 保存imgs_bin
if imgs_bin:
    try:
        # 保存所有二进制图像数据到一个pickle文件
        bin_file_path = os.path.join(save_dir, f"face_{face_id}_bin_data.pkl")
        with open(bin_file_path, 'wb') as f:
            pickle.dump({
                'face_id': face_id,
                'count': count,
                'images': imgs_bin
            }, f)
        print(f"已将 {count} 张人脸二进制数据保存到 {bin_file_path}")

        # 可选：调用训练函数，使用采集的数据进行训练
        # Train_v2.train()
    except Exception as e:
        print(f"保存二进制图像数据失败: {e}")
else:
    print("没有可保存的人脸二进制数据")
