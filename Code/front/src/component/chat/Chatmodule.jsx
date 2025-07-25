import React, { useState, useRef, useEffect } from "react";
import { Button, Input, message } from "antd";
import {
  SendOutlined,
  AudioOutlined,
  AudioMutedOutlined,
} from "@ant-design/icons";
import styles from "./ChatModule.module.css";
import { changeUserData, getUserInfo } from "../../api/userservice/user.js";
import { useSelector } from "react-redux";
import qg from "../../assets/qg.png";

const { TextArea } = Input;

const ChatModule = ({ onSend, initialMessages = [], getTitle }) => {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [activated, setActivated] = useState(initialMessages.length > 0);
  const messageEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const scrollRef = useRef(null);
  const [isSending, setIsSending] = useState(false);
  const user = JSON.parse(localStorage.getItem("user"));
  const userId = user.id; // 假设用户ID存

  // 语音识别配置
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true; // ✅ 允许持续识别直到用户手动停止
      recognition.interimResults = false; // 如果需要实时中间结果可以设为 true
      recognition.lang = "zh-CN";

      recognition.onstart = () => {
        setIsListening(true);
        message.info("开始语音识别...");
      };

      recognition.onend = () => {
        // 注意：这只会在手动停止或出错时调用
        setIsListening(false);
      };

      recognition.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript;
          }
        }
        setInput((prev) => prev + transcript);
      };

      recognition.onerror = (event) => {
        setIsListening(false);
        message.error("语音识别出错：" + event.error);
      };

      recognitionRef.current = recognition;
    }
  }, []);
  // useEffect(() => {
  //   const SpeechRecognition =
  //     window.SpeechRecognition || window.webkitSpeechRecognition;
  //   if (SpeechRecognition) {
  //     const recognition = new SpeechRecognition();
  //     recognition.continuous = false;
  //     recognition.interimResults = false;
  //     recognition.lang = "zh-CN";

  //     recognition.onstart = () => {
  //       setIsListening(true);
  //       message.info("开始语音识别...");
  //     };

  //     recognition.onend = () => {
  //       setIsListening(false);
  //     };

  //     recognition.onresult = (event) => {
  //       const transcript = event.results[0][0].transcript;
  //       setInput((prev) => prev + transcript);
  //       message.success("语音识别完成");
  //     };

  //     recognition.onerror = (event) => {
  //       setIsListening(false);
  //       message.error("语音识别失败，请重试");
  //     };

  //     recognitionRef.current = recognition;
  //   }
  // }, []);

  // 发送消息

  const handleSend = async () => {
    if (input.trim()) {
      const newMessage = {
        content: input.trim(),
        role: "user",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, newMessage]);

      //如果为第一条信息，则调用getTitle函数
      if (messages.length === 0) {
        getTitle(newMessage.content);
      }

      // 调用外部onSend回调
      if (onSend) {
        onSend(newMessage);
        // pollForAIResponse();
      }
      // try {
      //   await changeUserData({
      //     name: "linlinlin1",
      //   });
      //   console.log("更新用户数据成功");
      // } catch (error) {
      //   console.error("更新用户数据失败", error);
      // }

      // try {
      //   const response = await getUserInfo(userId);
      //   console.log("fasongId", userId);
      //   console.log("获取用户信息成功", response);
      //   message.success("获取用户信息成功");
      // } catch (error) {
      //   console.error("获取用户信息失败", error);
      // }

      setInput("");
      setActivated(true);
      setIsSending(true); // 禁用发送

      //轮询获取回复
      pollForAIResponse();
    }
  };

  const pollForAIResponse = () => {
    let retries = 0;
    const maxRetries = 10;
    const interval = 1500;

    const poller = setInterval(() => {
      retries++;

      // 模拟 AI 返回内容（你可以替换为实际 fetch 请求）
      const mockAIResponse = {
        content: "这是AI的回复内容。",
        role: "assistant",
        timestamp: new Date(),
      };

      // 模拟第3次轮询收到回复
      if (retries === 3) {
        clearInterval(poller);
        setMessages((prev) => [...prev, mockAIResponse]);
        setIsSending(false); // 启用发送按钮

        // ✅ 朗读文字
        speakText(mockAIResponse.content);
      }

      if (retries >= maxRetries) {
        clearInterval(poller);
        setIsSending(false);
        message.error("AI未及时回复，请稍后重试");
      }
    }, interval);
  };

  //将文字转语音
  function speakText(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "zh-CN";
    window.speechSynthesis.speak(utterance);
  }

  // 开始语音识别
  const startListening = () => {
    if (!recognitionRef.current) {
      message.error("您的浏览器不支持语音识别功能");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.start();
      } catch (error) {
        message.error("启动语音识别失败");
      }
    }
  };

  // 滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // 处理键盘事件
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 渲染输入区域
  const renderInputArea = () => (
    <div
      className={activated ? styles.chatFooter : styles.initialInputContainer}
    >
      <div className={styles.inputArea}>
        <TextArea
          className={styles.textArea}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          autoSize={{ minRows: 2, maxRows: 5 }}
          placeholder="请输入内容..."
        />
        <div className={styles.chatActions}>
          <Button
            icon={isListening ? <AudioMutedOutlined /> : <AudioOutlined />}
            onClick={startListening}
            className={`${styles.voiceButton} ${
              isListening ? styles.listening : ""
            }`}
            type={isListening ? "primary" : "default"}
            danger={isListening}
          >
            {isListening ? "停止录音" : "语音输入"}
          </Button>
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            disabled={!input.trim()}
            className={styles.sendButton}
          >
            发送
          </Button>
        </div>
      </div>
    </div>
  );

  // 如果未激活，显示初始居中布局
  if (!activated) {
    return (
      <div className={styles.chatWrapper}>
        <div className={styles.initialState}>
          <h1 className={styles.title}>欢迎来到多智能体协同辅助政务服务助手</h1>
          {renderInputArea()}
        </div>
      </div>
    );
  }

  // 激活后的布局
  return (
    <div className={styles.chatWrapper}>
      <div className={styles.activatedLayout}>
        <div className={styles.chatContent} ref={scrollRef}>
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`${styles.chatMessage} ${
                msg.role === "user" ? styles.messageRight : styles.messageLeft
              }`}
            >
              <div
                className={
                  msg.role === "user"
                    ? styles.chatMessageUser
                    : styles.chatMessageAI
                }
              >
                {msg.content}
              </div>
            </div>
          ))}
          <div ref={messageEndRef} />
        </div>
        {renderInputArea()}
      </div>
    </div>
  );
};

export default ChatModule;
