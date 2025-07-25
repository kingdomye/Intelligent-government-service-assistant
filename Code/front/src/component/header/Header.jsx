import React from "react";
import {
  Layout,
  Avatar,
  Dropdown,
  Menu,
  Typography,
  Space,
  message,
} from "antd";
import { UserOutlined, LogoutOutlined } from "@ant-design/icons";
import qg from "../../assets/qg.png";
const { Header } = Layout;
const { Title } = Typography;
import style from "./Header.module.css";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { logout } from "../../store/slice/userSlice.js";
import { useDispatch } from "react-redux";

const HeaderBar = ({ title, avatar }) => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("user"));
  console.log("user", user);

  const onLogout = () => {
    //退出登录
    message.success("退出登录成功");
    localStorage.removeItem("user");
    dispatch(logout());

    navigate("/");
  };

  const menuItems = [
    {
      key: "profile",
      icon: <UserOutlined />,
      label: user.name,
    },
    {
      key: "logout",
      icon: <LogoutOutlined />,
      label: "退出登录",
      onClick: onLogout,
    },
  ];

  return (
    <Header className={style.header}>
      {/* 左侧 Logo */}
      <div className={style.logoContainer}>
        <img src={avatar} alt="Logo" className={style.logo} />
      </div>

      {/* 中间标题 */}
      <Title level={4} className={style.title}>
        {title}
      </Title>

      {/* 右侧头像 + 下拉菜单 */}
      <Dropdown menu={{ items: menuItems }} placement="bottomRight">
        <Space className={style.userInfo}>
          <Avatar
            size="large"
            icon={user ? <UserOutlined /> : null}
            src={user ? user.avatar : null}
          />
        </Space>
      </Dropdown>
    </Header>
  );
};

export default HeaderBar;
