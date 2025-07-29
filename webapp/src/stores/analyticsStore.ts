import { create } from 'zustand';
import { uploadFile, askAgent } from '../services/api';

export interface Dataset {
  name: string;
  columns: string[];
  data: any[][];
  id?: string;
}

export interface ChatMessage {
  agent: string;
  message: string;
  timestamp: string;
  loading?: boolean;
}

export interface AgentStatus {
  name: string;
  color: string;
  active: boolean;
}

interface AnalyticsStore {
  currentDataset: Dataset | null;
  chatHistory: ChatMessage[];
  isAnalyzing: boolean;
  agents: AgentStatus[];

  uploadDataset: (file: File) => Promise<void>;
  sendMessage: (message: string) => Promise<void>;
  clearHistory: () => void;
}

export const useAnalyticsStore = create<AnalyticsStore>((set, get) => ({
  currentDataset: null,
  chatHistory: [],
  isAnalyzing: false,
  agents: [
    { name: 'profiling', color: '#3b82f6', active: true },
    { name: 'statistical', color: '#8b5cf6', active: true },
    { name: 'visualization', color: '#10b981', active: true },
    { name: 'transformation', color: '#f59e0b', active: true },
    { name: 'supervisor', color: '#ef4444', active: true },
  ],
  uploadDataset: async (file: File) => {
    set({ isAnalyzing: true });
    try {
      const res = await uploadFile(file);
      set({
        currentDataset: {
          name: res.name,
          columns: res.columns,
          data: res.data,
          id: res.id,
        },
        isAnalyzing: false,
      });
    } catch (e) {
      set({ isAnalyzing: false });
      alert('Erreur lors de l’upload du fichier.');
    }
  },
  sendMessage: async (message: string) => {
    const { currentDataset } = get();
    set(state => ({
      chatHistory: [
        ...state.chatHistory,
        { agent: 'user', message, timestamp: new Date().toLocaleTimeString() },
        { agent: 'profiling', message: 'Analyse en cours...', timestamp: new Date().toLocaleTimeString(), loading: true },
      ],
      isAnalyzing: true,
    }));
    try {
      const res = await askAgent(message, currentDataset?.id);
      set(state => ({
        chatHistory: [
          ...state.chatHistory.slice(0, -1),
          { agent: res.agent, message: res.response, timestamp: new Date().toLocaleTimeString(), loading: false },
        ],
        isAnalyzing: false,
      }));
    } catch (e) {
      set(state => ({
        chatHistory: [
          ...state.chatHistory.slice(0, -1),
          { agent: 'error', message: 'Erreur lors de la requête à l’API.', timestamp: new Date().toLocaleTimeString(), loading: false },
        ],
        isAnalyzing: false,
      }));
    }
  },
  clearHistory: () => set({ chatHistory: [] }),
}));
