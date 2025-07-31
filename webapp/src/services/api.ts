import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const uploadDataset = async (file: File, question: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('question', question);
  const response = await axios.post(`${API_URL}/analyze/`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const sendChatMessage = async (question: string) => {
  const response = await axios.post(`${API_URL}/chat/`, { question });
  return response.data;
};

// Ajoute d'autres endpoints si besoin
