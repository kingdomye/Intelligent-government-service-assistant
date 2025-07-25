import React, { useState, useRef, useEffect } from "react";
import {
  Card,
  Tabs,
  Form,
  Input,
  Button,
  message,
  Typography,
  Alert,
} from "antd";
import {
  UserOutlined,
  LockOutlined,
  SmileOutlined,
  MailOutlined,
  PhoneOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
const { Text } = Typography;
import { loginUser } from "../../store/slice/userSlice.js";
import { useDispatch } from "react-redux";
import { faceLogin } from "../../api/userservice/user.js";

const MAX_ATTEMPTS = 10;

const LoginForm = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [loginType, setLoginType] = useState("email");
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const handleAccountLogin = async (values) => {
    setLoading(true);
    try {
      const result = await dispatch(loginUser(values)).unwrap();
      console.log("登录成功", result.payload);
      message.success(`登录成功，欢迎 ${result.data.name}`);
      navigate("/Home");
    } catch (error) {
      message.error("登录失败");
      console.error("登录失败", error);
    } finally {
      setLoading(false);
    }
  };

  // 初始化摄像头
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (error) {
      message.error("无法访问摄像头");
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
  };

  // 拍照并返回 Blob 图像数据
  const takePhotoBlob = () => {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob);
      }, "image/jpeg");
    });
  };

  const toggleLoginType = () => {
    form.resetFields(); // 切换时清空字段
    setLoginType((prev) => (prev === "email" ? "phone" : "email"));
  };

  // const handleFaceLogin = async () => {
  //   setLoading(true);
  //   try {
  //     await new Promise((resolve) => setTimeout(resolve, 1000));
  //     //调用封装完的login函数
  //     message.success("人脸识别登录成功");
  //   } catch {
  //     message.error("人脸识别失败");
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  const handleFaceLogin = async () => {
    setLoading(true);
    let success = false;

    try {
      await startCamera();

      // 确保视频尺寸准备好
      if (
        !videoRef.current ||
        videoRef.current.videoWidth === 0 ||
        videoRef.current.videoHeight === 0
      ) {
        throw new Error("摄像头视频未就绪");
      }
    } catch (error) {
      message.error("摄像头初始化失败：" + error.message);
    }

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const blob = await takePhotoBlob();
      console.log(`第${attempt}次尝试识别人脸...`);
      const formData = new FormData();
      formData.append("image", blob, "face.jpg");

      try {
        const result = await faceLogin(formData);
        console.log(`第${attempt}次识别结果：`, result);
        if (result.code === 200) {
          success = true;
          message.success("人脸识别登录成功");
          break;
        } else {
          console.log(`第${attempt}次失败，重试中...`);
        }
      } catch (error) {
        console.error(`第${attempt}次请求失败`, error);
      }

      await new Promise((resolve) => setTimeout(resolve, 800));
    }

    if (!success) {
      message.error("识别失败次数过多，请稍后再试");
    }

    setLoading(false);
    stopCamera();
  };

  const tabItems = [
    {
      key: "account",
      label: "账号密码登录",
      children: (
        <Form form={form} onFinish={handleAccountLogin} autoComplete="off">
          {loginType === "email" ? (
            <Form.Item
              name="identity"
              rules={[
                { required: true, message: "请输入邮箱" },
                { type: "email", message: "请输入有效的邮箱" },
              ]}
            >
              <Input prefix={<MailOutlined />} placeholder="邮箱" />
            </Form.Item>
          ) : (
            <Form.Item
              name="identity"
              rules={[
                { required: true, message: "请输入手机号" },
                { pattern: /^1[3-9]\d{9}$/, message: "请输入有效的手机号" },
              ]}
            >
              <Input prefix={<PhoneOutlined />} placeholder="手机号" />
            </Form.Item>
          )}

          <Form.Item
            name="password"
            rules={[{ required: true, message: "请输入密码" }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="密码"
              autoComplete="current-password"
            />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block loading={loading}>
              登录
            </Button>
          </Form.Item>

          <Form.Item style={{ textAlign: "center", marginBottom: 0 }}>
            <Text
              type="secondary"
              style={{ cursor: "pointer" }}
              onClick={toggleLoginType}
            >
              {loginType === "email" ? "使用手机号登录" : "使用邮箱登录"}
            </Text>
          </Form.Item>
        </Form>
      ),
    },
    {
      key: "face",
      label: "人脸识别登录",
      children: (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "10px",
            width: "100%",
          }}
        >
          <Alert
            message="请确保保持人脸位于摄像头前，且光线充足"
            type="warning"
            style={{ width: "100%" }}
          />
          <Button
            type="primary"
            icon={<SmileOutlined />}
            block
            onClick={handleFaceLogin}
            loading={loading}
          >
            识别人脸并登录
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <video ref={videoRef} style={{ display: "none" }} muted />
      <Card style={{ width: 400, border: "none" }}>
        <Tabs defaultActiveKey="account" items={tabItems} />
      </Card>
    </div>
  );
};

export default LoginForm;
