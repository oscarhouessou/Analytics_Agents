import { useEffect, useRef, useState } from 'react'
import { useAnalyticsStore } from '../../stores/analyticsStore'
import { toast } from 'sonner'

const SUGGESTIONS = [
  'Qui sont mes meilleurs clients ?',
  'Quels produits cartonnent ?',
  'Comment évolue mon CA ?',
  'Montre-moi la répartition par région',
  'Montre-moi les outliers',
  'Crée un dashboard de cela',
]

const AGENT_STYLES = {
  profiling: 'bg-blue-50 text-blue-700 border-blue-200',
  statistique: 'bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200',
  visualisation: 'bg-orange-50 text-orange-700 border-orange-200',
  transformation: 'bg-green-50 text-green-700 border-green-200',
}

function AgentBubble({ agent, message, timestamp, loading }: any) {
  const style = AGENT_STYLES[agent] || 'bg-gray-50 text-gray-700 border-gray-200';
  return (
    <div className={`rounded-2xl border px-4 py-3 mb-2 shadow-sm flex flex-col ${style}`}>
      <div className="flex items-center gap-2 mb-1">
        <span className="font-bold capitalize">{agent}</span>
        <span className="text-xs text-gray-400">{timestamp}</span>
      </div>
      <div className="text-base whitespace-pre-line">
        {loading ? <span className="animate-pulse">Génération en cours...</span> : message}
      </div>
    </div>
  )
}


export default function ChatPanel() {
  const { chatHistory, error } = useAnalyticsStore();
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState(SUGGESTIONS);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const userName = 'Sarah';

  useEffect(() => {
    if (error) toast.error(error);
  }, [error]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Simule la suggestion contextuelle après chaque réponse
  useEffect(() => {
    if (chatHistory.length > 0) {
      setSuggestions(SUGGESTIONS.sort(() => 0.5 - Math.random()).slice(0, 3));
    }
  }, [chatHistory]);

  // Simule l'envoi d'une question
  const handleSend = (e: any) => {
    e.preventDefault();
    if (!input.trim()) return;
    setIsTyping(true);
    setTimeout(() => {
      setIsTyping(false);
      // Ici, on devrait appeler l'API pour générer la réponse IA
      // et mettre à jour le store (mocké pour la démo)
    }, 1200);
    setInput('');
  };

  return (
    <div className="flex flex-col w-full max-w-3xl mx-auto mt-8 mb-16">
      <div className="font-bold text-blue-700 text-lg mb-2">Votre Assistant Analytics</div>
      <div className="flex flex-col gap-2 min-h-[300px] bg-white/80 rounded-2xl p-6 shadow-lg border border-blue-50">
        {chatHistory.length === 0 && (
          <div className="text-blue-300 text-center py-8">
            <div className="text-lg text-blue-700 font-bold mb-2">🤖 Bonjour {userName} !</div>
            <div className="mb-2">J'ai analysé vos données. Que voulez-vous découvrir ?</div>
            <div className="text-blue-400 text-sm">Essayez une question ou choisissez une suggestion ci-dessous.</div>
          </div>
        )}
        {chatHistory.map((msg: any, i: number) => (
          <AgentBubble key={i} {...msg} />
        ))}
        {isTyping && (
          <div className="flex items-center gap-2 mt-2">
            <span className="animate-bounce text-blue-400 text-lg">🤖</span>
            <span className="italic text-blue-400">L'IA réfléchit...</span>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      {/* Suggestions contextuelles */}
      {suggestions.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {suggestions.map((s, i) => (
            <button
              key={i}
              className="bg-blue-50 text-blue-700 px-4 py-2 rounded-full text-sm font-semibold shadow hover:bg-blue-100 transition"
              onClick={() => setInput(s)}
            >{s}</button>
          ))}
        </div>
      )}
      {/* Zone de saisie */}
      <form className="flex gap-2 mt-4" onSubmit={handleSend}>
        <input
          type="text"
          className="flex-1 border rounded-lg px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-blue-400 shadow"
          placeholder="Posez votre question..."
          value={input}
          onChange={e => setInput(e.target.value)}
        />
        <button type="submit" className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2 text-base">
          Envoyer
        </button>
      </form>
    </div>
  )
}
