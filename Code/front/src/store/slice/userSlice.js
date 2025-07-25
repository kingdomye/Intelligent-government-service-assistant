import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { login, register } from "../../api/userservice/user.js";

//导入axios的登录，退出账号等，这里假定导入为axios

//登录
export const loginUser = createAsyncThunk(
  "/user/login",
  async (credentials, { rejectWithValue }) => {
    try {
      const data = await login(credentials);
      console.log("登录成功", data);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || "登录失败");
    }
  }
);

export const registerUser = createAsyncThunk(
  "/user/userRegister", //注册用户API
  async (userData, { rejectWithValue }) => {
    try {
      const data = await register(userData);
      console.log("注册成功", data);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || "注册失败");
    }
  }
);

const userSlice = createSlice({
  name: "user",
  initialState: {
    id: null,
    name: null,
    loading: false,
    error: null,
  },
  reducers: {
    logout: (state) => {
      state.name = null;
      state.id = null;
      localStorage.removeItem("user");
    },
    clearError: (state) => {
      state.error = null;
    },
    //获取用户信息
    getUserInfo: (state, action) => {
      state.user = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(loginUser.pending, (state) => {
        state.loading = true;
        state.error = null;
        state.name = null;
        state.id = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.loading = false;
        state.name = action.payload.data.name;
        state.id = action.payload.data.id;
        state.error = null;
        localStorage.setItem("user", JSON.stringify(action.payload.data));
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      .addCase(registerUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state, action) => {
        state.loading = false;
        state.name = action.payload.data.name;
        state.id = action.payload.data.id;
        state.error = null;
        localStorage.setItem("user", JSON.stringify(action.payload.data));
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const { logout, clearError, loginSuccess } = userSlice.actions;
export default userSlice.reducer;
