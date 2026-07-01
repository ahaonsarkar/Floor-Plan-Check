import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

export const uploadFloorPlan = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload-floor-plan', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const sendChatMessage = async (message, history, floorPlanContext = null) => {
  const response = await api.post('/chat', {
    message,
    history,
    floor_plan_context: floorPlanContext,
  });
  return response.data;
};

export const generateReport = async (floorPlanContext) => {
  const response = await api.post('/generate-report', {
    floor_plan_context: floorPlanContext,
  });
  return response.data;
};

export default api;