import React, { useState } from "react";
import { Form, Input, Button, message, Card } from "antd";
import { MailOutlined, LockOutlined, SafetyOutlined } from "@ant-design/icons";

const ResetForm = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [codeLoading, setCodeLoading] = useState(false);
  const [countdown, setCountdown] = useState(0);

  const handleGetCode = async () => {
    try {
      const email = form.getFieldValue("email");
      if (!email) {
        message.warning("请先输入邮箱");
        return;
      }
      form.validateFields(["email"]);
      setCodeLoading(true);
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
      // 验证失败忽略
    } finally {
      setCodeLoading(false);
    }
  };

  const handleResetPassword = async (values) => {
    setLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      message.success("密码重置成功");
    } catch {
      message.error("密码重置失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <Card style={{ width: 400, border: "none" }}>
        <Form form={form} onFinish={handleResetPassword} autoComplete="off">
          <Form.Item
            name="email"
            rules={[
              { required: true, message: "请输入邮箱" },
              { type: "email", message: "请输入有效的邮箱" },
            ]}
          >
            <Input prefix={<MailOutlined />} placeholder="邮箱" />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
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
          </Form.Item>

          <Form.Item
            name="newPassword"
            rules={[
              { required: true, message: "请输入新密码" },
              { min: 6, message: "密码至少6位" },
            ]}
          >
            <Input.Password prefix={<LockOutlined />} placeholder="新密码" />
          </Form.Item>

          <Form.Item
            name="confirmPassword"
            dependencies={["newPassword"]}
            rules={[
              { required: true, message: "请确认新密码" },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue("newPassword") === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error("两次输入的密码不一致"));
                },
              }),
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="确认新密码"
            />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block loading={loading}>
              确认重置密码
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default ResetForm;
