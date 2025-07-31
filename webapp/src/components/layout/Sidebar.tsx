import { LucideBarChart, LucideMessageCircle, LucideUpload, LucideSettings } from 'lucide-react'
import { cn } from '../../utils/cn'

const navItems = [
  { label: 'Dashboard', icon: LucideBarChart },
  { label: 'Chat', icon: LucideMessageCircle },
  { label: 'Upload', icon: LucideUpload },
  { label: 'Paramètres', icon: LucideSettings },
]

export function Sidebar({ current, onNavigate }: { current: string, onNavigate: (label: string) => void }) {
  return (
    <aside className="h-full w-20 bg-white border-r flex flex-col items-center py-6 gap-4 shadow-sm">
      {navItems.map(({ label, icon: Icon }) => (
        <button
          key={label}
          className={cn('flex flex-col items-center gap-1 text-xs text-gray-500 hover:text-blue-600 focus:outline-none', current === label && 'text-blue-600 font-bold')}
          onClick={() => onNavigate(label)}
          aria-label={label}
        >
          <Icon size={24} />
          {label}
        </button>
      ))}
    </aside>
  )
}
