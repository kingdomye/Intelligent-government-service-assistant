import React from "react";
import { useState } from "react";
import Header from "../component/header/Header.jsx";
import ChatModule from "../component/chat/Chatmodule.jsx";
import style from "./Home.module.css";
import { useSelector } from "react-redux";
import qg from "../assets/qg.png";

export default function Home() {
  const [title, setTitle] = React.useState("");
  const user = useSelector((state) => state.user);
  console.log("user", user);

  return (
    <div className={style.container}>
      <Header title={title} avatar={qg} />
      <ChatModule
        className={style.chatModule}
        initialMessages={
          [
            // { role: "assistant", content: "你好，我是AI助手。" },
            // { role: "user", content: "你好" },
          ]
        }
        onSend={(message) => {
          console.log("用户发送的消息：", message);
          // 可以调用后端接口、更新聊天记录等
        }}
        getTitle={(titleText) => {
          setTitle(titleText); // 设置标题
        }}
      />
    </div>
  );
}
