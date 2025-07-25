import { message } from "antd";
import api from "../index";
import aiApi from "../aiapi.js";

export const login = async (credentials) => {
  const response = await api.post("/user/loginbypassword", credentials);
  return response.data;
};

export const register = async (userData) => {
  const response = await api.post("/user/userRegister", userData);
  return response.data;
};

export const faceLogin = async (imageData) => {
  // const response = await aiApi.post("/user/faceLogin", formData, {
  //   headers: {
  //     "Content-Type": "multipart/form-data",
  //   },
  // });
  // return response.data;

  await new Promise((resolve) => setTimeout(resolve, 500));
  console.log("模拟接收到 FormData，对应的 Blob 是：", imageData.get("image"));

  return {
    code: 200,
    msg: "识别成功",
  };
};

export const changeUserData = async (userData) => {
  const response = await api.post("/user/updateInfomation", userData);
  return response.data;
};

export const getUserInfo = async (id) => {
  const response = await api.get("/user/GetInformationById", {
    params: { id },
  });
  return response.data;
};

// export const logout = async () => {
//   try {
//     const response = await api.post("/logout");
//     return response.data;
//   } catch (error) {
//     console.error("Logout failed", error);
//     throw error;
//   }
// };
