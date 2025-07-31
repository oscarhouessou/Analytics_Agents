import { useAnalyticsStore } from '../stores/analyticsStore';

export const useChat = () => {
  const { chatHistory, sendMessage, isAnalyzing, error } = useAnalyticsStore();
  return {
    messages: chatHistory,
    sendMessage,
    isLoading: isAnalyzing,
    error,
  };
};
