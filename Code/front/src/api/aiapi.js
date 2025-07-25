import axios from "axios";

const aiApi = axios.create({
  baseURL: "https://ai-team-backend.com/api",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

export default aiApi;
