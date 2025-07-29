import React, { useState } from 'react';
import { useAnalyticsStore } from '../stores/analyticsStore';

const suggestions = [
  "Montre-moi des graphiques",
  "Analyse les corrélations",
  "Qualité des données",
  "Suggère des transformations",
  "Résumé des données"
];

const agentColors: Record<string, string> = {
  profiling: '#3b82f6',
  statistical: '#8b5cf6',
  visualization: '#10b981',
  transformation: '#f59e0b',
  supervisor: '#ef4444',
  user: '#64748b',
};

const Chat: React.FC = () => {
  const { chatHistory, sendMessage, isAnalyzing } = useAnalyticsStore();
  const [input, setInput] = useState('');

  const handleSend = async (msg: string) => {
    if (!msg.trim()) return;
    setInput('');
    await sendMessage(msg);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-white">
      <h2 className="text-2xl font-bold mb-4">💬 Chat Multi-Agents</h2>
      <div className="w-full max-w-2xl bg-white rounded-xl shadow p-8 flex flex-col gap-6">
        <div className="flex flex-wrap gap-2 mb-2">
          {suggestions.map(s => (
            <button
              key={s}
              className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200 transition"
              onClick={() => handleSend(s)}
              disabled={isAnalyzing}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="flex-1 overflow-y-auto max-h-80 border rounded p-3 bg-blue-50">
          {chatHistory.length === 0 && (
            <div className="text-gray-400 text-center">Aucune conversation pour l’instant.</div>
          )}
          {chatHistory.map((msg, i) => (
            <div key={i} className="mb-3 flex items-start gap-2">
              <span
                className="w-8 h-8 rounded-full flex items-center justify-center font-bold text-white"
                style={{ background: agentColors[msg.agent] || '#64748b' }}
              >
                {msg.agent === 'user' ? '👤' : '🤖'}
              </span>
              <div>
                <div className="text-xs text-gray-400">{msg.agent} • {msg.timestamp}</div>
                <div className="bg-white rounded px-3 py-2 shadow text-gray-700 mt-1">
                  {msg.loading ? <span className="italic text-blue-400">Analyse en cours...</span> : msg.message}
                </div>
              </div>
            </div>
          ))}
        </div>
        <form
          className="flex gap-2 mt-2"
          onSubmit={e => {
            e.preventDefault();
            handleSend(input);
          }}
        >
          <input
            className="flex-1 border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-300"
            placeholder="Posez une question à l’IA..."
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={isAnalyzing}
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            disabled={isAnalyzing || !input.trim()}
          >Envoyer</button>
        </form>
      </div>
    </div>
  );
};

export default Chat;
