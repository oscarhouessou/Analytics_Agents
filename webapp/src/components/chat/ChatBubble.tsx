import { cn } from '../../utils/cn'
import { LucideUser } from 'lucide-react'

interface ChatBubbleProps {
  agent: string
  message: string
  timestamp: string
  loading?: boolean
}

const agentColors: Record<string, string> = {
  profiling: '#3b82f6',
  statistical: '#8b5cf6',
  visualization: '#10b981',
  transformation: '#f59e0b',
  supervisor: '#ef4444',
}

export function ChatBubble({ agent, message, timestamp, loading }: ChatBubbleProps) {
  return (
    <div className={cn('flex items-start gap-2 mb-2')}> 
      <span className="w-8 h-8 rounded-full flex items-center justify-center" style={{ background: agentColors[agent] || '#e5e7eb' }}>
        <LucideUser className="text-white" size={20} />
      </span>
      <div className="flex-1">
        <div className="text-xs text-gray-400 mb-1">{agent} • {timestamp}</div>
        <div className={cn('rounded-lg px-4 py-2 bg-white shadow', loading && 'opacity-60 animate-pulse')}>{message}</div>
      </div>
    </div>
  )
}
