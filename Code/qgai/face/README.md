# face模块说明文档

## **模型训练**

### 输入输出设计

##### 输入

- 二进制RGB图片列表`imgs_bin`（`list`）
- 用户id `user_id`（`string`）
- 准确率阀值`min_acc`（`float`）

##### 输出

- 训练状态（`bool`）

### 参数说明

1、图片要求为三通道格式

2、若训练成功返回True，训练失败返回False

3、训练数据会在日志中显示

4、如果置信度低于准确率阀值会预判为预测失败

### 调用示例

```python
from face.Train import cv2_train

imgs_bin = [bin1, bin2, bin3, ...]
user_id = "2sj82vsd"
min_acc = 0.6

train_result = cv2_train(imgs_bin=imgs_bin, user_id=user_id, min_acc=min_acc)

```

------

## **模型预测**

### 输入输出设计

##### 输入

- 二进制RGB图片列表`imgs_bin`（`list`）
- 准确率阀值`min_acc`（`float`）

##### 输出

- 用户id（`string`）

### 参数说明

1、图片要求为三通道格式

2、函数返回预测的用户id，如果预测失败返回`None`

3、如果置信度低于准确率阀值会预判为预测失败

### 调用示例

```python
from face.Predict import cv2_predict

imgs_bin = [bin1, bin2, bin3, ...]
min_acc = 0.6

pred_id = cv2_train(imgs_bin=imgs_bin, min_acc=min_acc)

```

