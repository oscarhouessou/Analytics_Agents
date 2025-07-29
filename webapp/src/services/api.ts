import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await axios.post(`${API_BASE}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data; // { columns: string[], data: any[][], name: string }
};

export const askAgent = async (question: string, datasetId?: string) => {
  const res = await axios.post(`${API_BASE}/ask`, { question, datasetId });
  return res.data; // { agent: string, response: string, visualizations?: any[] }
};
