import aiApi from "../aiapi.js";

//一个发送信息的接口
//在home页面的onSend回调中调用
export const sendMessage = async (message) => {
  try {
    const response = await aiApi.post("/chat/send", { message });
    return response.data;
  } catch (error) {
    console.error("发送信息失败：", error);
    throw error;
  }
};

//一个用来轮询接收信息的接口
export const receiveMessages = async () => {
  try {
    const response = await aiApi.get("/chat/receive");
    return response.data;
  } catch (error) {
    console.error("接收信息失败：", error);
    throw error;
  }
};
