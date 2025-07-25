import React from "react";
import { Navigate } from "react-router-dom";
import { useSelector } from "react-redux"; // 或其他全局状态方式
import { message } from "antd";

const ManagerRouter = ({ children }) => {
  const user = localStorage.getItem("user");

  if (user === null) {
    message.error("请先登录");
    return <Navigate to="/" replace />;
  }

  return children;
};

export default ManagerRouter;
