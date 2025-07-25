import React, { useEffect, useRef, useState } from "react";
import { Button, Progress, message, Alert } from "antd";
import style from "./Collect.module.css";
import api from "../api/index.js";

const Collect = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [collecting, setCollecting] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [progress, setProgress] = useState(0);
  const maxImages = 100;
  const intervalTime = 5000; // 5秒间隔

  // const [message, setMessage] = useState({
  //   userId: 100,
  //   endTime: "2025-09-13 01:42:29",
  //   reason: "reprehenderit Ut aute consequat",
  // });
  // api.get("/applyDevelopers", message)

  useEffect(() => {
    // 打开摄像头
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      })
      .catch(() => {
        message.error("无法访问摄像头");
      });

    return () => {
      // 组件销毁时关闭摄像头
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  //上传所有图片
  const uploadAllImages = async () => {
    const formData = new FormData();
    capturedImages.forEach((blob, index) => {
      formData.append("images", blob, `image_${index}.png`);
    });

    try {
      const response = await api.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log("批量上传成功", response.data);
      message.success("全部图片上传成功！");
    } catch (error) {
      console.error("批量上传失败", error);
      message.error("批量上传失败！");
    }
  };

  useEffect(() => {
    let timer;
    if (collecting) {
      timer = setInterval(() => {
        if (capturedImages.length >= maxImages) {
          clearInterval(timer);
          setCollecting(false);
          message.success("采集完成！");
          return;
        }
        captureImage();
      }, intervalTime);
      if (capturedImages.length >= maxImages) {
        clearInterval(timer);
        setCollecting(false);
        message.success("采集完成，开始上传！");
        uploadAllImages();
        return;
      }
    }
    return () => clearInterval(timer);
  }, [collecting, capturedImages]);

  const captureImage = () => {
    const context = canvasRef.current.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, 320, 240);

    canvasRef.current.toBlob((blob) => {
      if (!blob) {
        message.error("捕获图像失败");
        return;
      }
      setCapturedImages((prev) => {
        const updated = [...prev, blob];
        setProgress((updated.length / maxImages) * 100);
        return updated;
      });
    }, "image/png");
    // canvasRef.current.toBlob(async (blob) => {
    //   if (!blob) return;

    //   const formData = new FormData();
    //   formData.append("image", blob, `image_${Date.now()}.png`); // 设置文件名

    //   try {
    //     await api.post("", formData, {
    //       headers: {
    //         "Content-Type": "multipart/form-data",
    //       },
    //     });
    //     setCapturedImages((prev) => [...prev, URL.createObjectURL(blob)]);
    //     setProgress(((capturedImages.length + 1) / maxImages) * 100);
    //   } catch (error) {
    //     console.error("上传失败", error);
    //     message.error("图片上传失败");
    //   }
    // }, "image/png");
  };

  const handleStart = () => {
    if (!videoRef.current.srcObject) {
      message.warning("摄像头未就绪");
      return;
    }
    setCapturedImages([]);
    setProgress(0);
    setCollecting(true);
  };

  return (
    <div className={style.main}>
      <div className={style.container}>
        <video
          ref={videoRef}
          width="320"
          height="240"
          style={{ border: "1px solid #ccc" }}
        />
        <canvas
          ref={canvasRef}
          width="320"
          height="240"
          style={{ display: "none" }}
        />
        <div>
          <Button type="primary" onClick={handleStart} disabled={collecting}>
            {collecting ? "采集中..." : "开始采集"}
          </Button>
          {/* 增加一个按钮，用于停止采集 */}
          <Button
            type="danger"
            onClick={() => setCollecting(false)}
            disabled={!collecting}
            style={{ marginLeft: 10 }}
          >
            停止采集
          </Button>
        </div>
        <div style={{ width: 320 }}>
          <Progress
            percent={progress}
            status={collecting ? "active" : "normal"}
          />
        </div>
        <Alert message="请缓慢转动头部，让系统采集更多角度" type="info" />
        <div style={{ marginTop: 10 }}>
          已采集：{capturedImages.length} / {maxImages}
        </div>
      </div>
    </div>
  );
};

export default Collect;
