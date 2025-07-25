import React, { useState } from "react";
import { Form, Input, Button, message, Card, Select } from "antd";
import {
  MailOutlined,
  LockOutlined,
  SafetyOutlined,
  PhoneOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { registerUser } from "../../store/slice/userSlice.js";

const RegisterForm = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [codeLoading, setCodeLoading] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const handleGetCode = async () => {
    try {
      const email = form.getFieldValue("email");
      if (!email) {
        message.warning("请先输入邮箱");
        return;
      }
      form.validateFields(["email"]);
      setCodeLoading(true);
      // 模拟发送验证码
      await new Promise((resolve) => setTimeout(resolve, 1000));
      message.success("验证码已发送");
      setCountdown(60);
      const timer = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch {
      // 验证失败不处理
    } finally {
      setCodeLoading(false);
    }
  };

  const handleRegister = async (values) => {
    console.log("注册信息", values);
    setLoading(true);
    try {
      // await new Promise((resolve) => setTimeout(resolve, 1000));
      const result = await dispatch(registerUser(values)).unwrap();
      message.success("注册成功，请收集人脸信息");
      navigate("/Collect");
    } catch (error) {
      message.error("注册失败");
      console.error("注册失败", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <Card style={{ width: 400, border: "none" }}>
        <Form form={form} onFinish={handleRegister} autoComplete="off">
          <Form.Item
            name="name"
            rules={[{ required: true, message: "请输入姓名" }]}
          >
            <Input placeholder="姓名" />
          </Form.Item>

          <Form.Item
            name="gender"
            rules={[{ required: true, message: "请选择性别" }]}
          >
            <Select placeholder="请选择性别">
              <Select.Option value="男">男</Select.Option>
              <Select.Option value="女">女</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="nationality"
            // label="民族"
            rules={[{ required: true, message: "请选择民族" }]}
          >
            <Select placeholder="请选择民族">
              <Select.Option value="汉族">汉族</Select.Option>
              <Select.Option value="蒙古族">蒙古族</Select.Option>
              <Select.Option value="回族">回族</Select.Option>
              <Select.Option value="藏族">藏族</Select.Option>
              <Select.Option value="维吾尔族">维吾尔族</Select.Option>
              <Select.Option value="苗族">苗族</Select.Option>
              <Select.Option value="彝族">彝族</Select.Option>
              <Select.Option value="壮族">壮族</Select.Option>
              <Select.Option value="布依族">布依族</Select.Option>
              <Select.Option value="朝鲜族">朝鲜族</Select.Option>
              <Select.Option value="满族">满族</Select.Option>
              <Select.Option value="侗族">侗族</Select.Option>
              <Select.Option value="瑶族">瑶族</Select.Option>
              <Select.Option value="白族">白族</Select.Option>
              <Select.Option value="土家族">土家族</Select.Option>
              <Select.Option value="哈尼族">哈尼族</Select.Option>
              <Select.Option value="哈萨克族">哈萨克族</Select.Option>
              <Select.Option value="傣族">傣族</Select.Option>
              <Select.Option value="黎族">黎族</Select.Option>
              <Select.Option value="傈僳族">傈僳族</Select.Option>
              <Select.Option value="佤族">佤族</Select.Option>
              <Select.Option value="畲族">畲族</Select.Option>
              <Select.Option value="高山族">高山族</Select.Option>
              <Select.Option value="拉祜族">拉祜族</Select.Option>
              <Select.Option value="水族">水族</Select.Option>
              <Select.Option value="东乡族">东乡族</Select.Option>
              <Select.Option value="纳西族">纳西族</Select.Option>
              <Select.Option value="景颇族">景颇族</Select.Option>
              <Select.Option value="柯尔克孜族">柯尔克孜族</Select.Option>
              <Select.Option value="土族">土族</Select.Option>
              <Select.Option value="达斡尔族">达斡尔族</Select.Option>
              <Select.Option value="仫佬族">仫佬族</Select.Option>
              <Select.Option value="羌族">羌族</Select.Option>
              <Select.Option value="布朗族">布朗族</Select.Option>
              <Select.Option value="撒拉族">撒拉族</Select.Option>
              <Select.Option value="毛南族">毛南族</Select.Option>
              <Select.Option value="仡佬族">仡佬族</Select.Option>
              <Select.Option value="锡伯族">锡伯族</Select.Option>
              <Select.Option value="阿昌族">阿昌族</Select.Option>
              <Select.Option value="普米族">普米族</Select.Option>
              <Select.Option value="塔吉克族">塔吉克族</Select.Option>
              <Select.Option value="怒族">怒族</Select.Option>
              <Select.Option value="乌孜别克族">乌孜别克族</Select.Option>
              <Select.Option value="俄罗斯族">俄罗斯族</Select.Option>
              <Select.Option value="鄂温克族">鄂温克族</Select.Option>
              <Select.Option value="德昂族">德昂族</Select.Option>
              <Select.Option value="保安族">保安族</Select.Option>
              <Select.Option value="裕固族">裕固族</Select.Option>
              <Select.Option value="京族">京族</Select.Option>
              <Select.Option value="塔塔尔族">塔塔尔族</Select.Option>
              <Select.Option value="独龙族">独龙族</Select.Option>
              <Select.Option value="鄂伦春族">鄂伦春族</Select.Option>
              <Select.Option value="赫哲族">赫哲族</Select.Option>
              <Select.Option value="门巴族">门巴族</Select.Option>
              <Select.Option value="珞巴族">珞巴族</Select.Option>
              <Select.Option value="基诺族">基诺族</Select.Option>
              <Select.Option value="其他">其他</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="idCard"
            rules={[
              { required: true, message: "请输入身份证号" },
              {
                pattern: /^\d{17}[\dXx]$/,
                message: "请输入有效的18位身份证号",
              },
            ]}
          >
            <Input placeholder="身份证号" maxLength={18} />
          </Form.Item>
          <Form.Item
            name="email"
            rules={[
              { required: true, message: "请输入邮箱" },
              { type: "email", message: "请输入有效的邮箱" },
            ]}
          >
            <Input prefix={<MailOutlined />} placeholder="邮箱" />
          </Form.Item>

          <Form.Item
            name="phone"
            rules={[
              { required: true, message: "请输入手机号" },
              { pattern: /^1[3-9]\d{9}$/, message: "请输入有效的手机号" },
            ]}
          >
            <Input prefix={<PhoneOutlined />} placeholder="手机号" />
          </Form.Item>

          {/* <Form.Item style={{ marginBottom: 0 }}>
            <Form.Item
              name="code"
              rules={[
                { required: true, message: "请输入验证码" },
                {
                  pattern: /^\d{6}$/,
                  message: "验证码必须是6位数字",
                },
              ]}
              style={{ display: "inline-block", width: "60%" }}
            >
              <Input prefix={<SafetyOutlined />} placeholder="验证码" />
            </Form.Item>
            <Form.Item
              style={{
                display: "inline-block",
                width: "38%",
                marginLeft: "2%",
              }}
            >
              <Button
                onClick={handleGetCode}
                disabled={countdown > 0}
                loading={codeLoading}
                block
              >
                {countdown > 0 ? `${countdown}s` : "获取验证码"}
              </Button>
            </Form.Item>
          </Form.Item> */}

          <Form.Item
            name="password"
            rules={[
              { required: true, message: "请输入密码" },
              { min: 6, message: "密码至少6位" },
            ]}
          >
            <Input.Password prefix={<LockOutlined />} placeholder="密码" />
          </Form.Item>

          <Form.Item
            name="password"
            dependencies={["password"]}
            rules={[
              { required: true, message: "请确认密码" },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue("password") === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error("两次输入的密码不一致"));
                },
              }),
            ]}
          >
            <Input.Password prefix={<LockOutlined />} placeholder="确认密码" />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block loading={loading}>
              注册
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default RegisterForm;
