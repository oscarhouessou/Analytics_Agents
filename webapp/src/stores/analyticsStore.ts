import { create } from 'zustand';
import { uploadDataset as apiUploadDataset, sendChatMessage } from '../services/api';
import axios from 'axios';

export interface Dataset {
  name: string
  columns: string[]
  rowCount: number
  sample: Record<string, any>[]
}

export interface ChatMessage {
  agent: string;
  message: string;
  timestamp: string;
  loading?: boolean;
  error?: string;
}

export interface AgentStatus {
  name: string
  color: string
  active: boolean
}

interface AnalyticsStore {
  currentDataset: File | null;
  chatHistory: ChatMessage[];
  isAnalyzing: boolean;
  error: string | null;
  agents: string[];
  datasetSummary: any | null;
  isSummaryLoading: boolean;
  uploadDataset: (file: File, question: string) => Promise<void>;
  fetchDatasetSummary: (file: File) => Promise<void>;
  sendMessage: (message: string) => Promise<void>;
  clearHistory: () => void;
}

export const useAnalyticsStore = create<AnalyticsStore>((set, get) => ({
  currentDataset: null,
  chatHistory: [],
  isAnalyzing: false,
  error: null,
  agents: ['profiling', 'statistical', 'visualization', 'transformation', 'supervisor'],
  datasetSummary: null,
  isSummaryLoading: false,
  uploadDataset: async (file, question) => {
    set({ isAnalyzing: true, error: null });
    // 1. Appel dataset summary
    await get().fetchDatasetSummary(file);
    // 2. Profiling automatique
    try {
      const profilingResult = await apiUploadDataset(file, 'Donne-moi un résumé du dataset');
      set((state) => ({
        chatHistory: [
          ...state.chatHistory,
          { agent: profilingResult.agent || 'profiling', message: profilingResult.response || 'Profiling terminé', timestamp: new Date().toISOString() },
        ],
        currentDataset: file,
      }));
    } catch (e: any) {
      set({ error: e?.message || 'Erreur profiling' });
    }
    // 3. Visualisation par défaut (histogramme sur la première colonne numérique)
    try {
      const summary = get().datasetSummary;
      if (summary) {
        const firstNumCol = summary.columns.find((col: string) => summary.dtypes[col] === 'float64' || summary.dtypes[col] === 'int64');
        if (firstNumCol) {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('column', firstNumCol);
          formData.append('chart_type', 'histogram');
          const res = await axios.post((import.meta.env.VITE_API_URL || 'http://localhost:8000') + '/visualize/', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          set({
            // Ajoute la visualisation dans le store si besoin (ex: datasetSummary.visualization = res.data)
            datasetSummary: { ...summary, defaultViz: res.data }
          });
        }
      }
    } catch (e: any) {
      // Visualisation par défaut non bloquante
    }
    set({ isAnalyzing: false });
  },
  fetchDatasetSummary: async (file: File) => {
    set({ isSummaryLoading: true, error: null });
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post((import.meta.env.VITE_API_URL || 'http://localhost:8000') + '/dataset/summary', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      set({ datasetSummary: res.data, isSummaryLoading: false });
    } catch (e: any) {
      set({ isSummaryLoading: false, error: e?.message || 'Erreur résumé dataset' });
    }
  },
  sendMessage: async (message) => {
    set((state) => ({
      chatHistory: [
        ...state.chatHistory,
        { agent: 'user', message, timestamp: new Date().toISOString(), loading: true },
      ],
      error: null,
    }));
    try {
      const result = await sendChatMessage(message);
      set((state) => ({
        chatHistory: [
          ...state.chatHistory.slice(0, -1),
          { ...state.chatHistory[state.chatHistory.length - 1], loading: false },
          { agent: result.agent || 'supervisor', message: result.answer || '', timestamp: new Date().toISOString() },
        ],
      }));
    } catch (e: any) {
      set((state) => ({
        chatHistory: [
          ...state.chatHistory.slice(0, -1),
          { ...state.chatHistory[state.chatHistory.length - 1], loading: false, error: e?.message || 'Erreur API' },
        ],
        error: e?.message || 'Erreur API',
      }));
    }
  },
  clearHistory: () => set({ chatHistory: [], error: null }),
}))
