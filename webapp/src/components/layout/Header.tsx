import { Button } from '../ui/Button'

export function Header() {
  return (
    <header className="w-full h-16 flex items-center justify-between px-6 bg-white border-b shadow-sm">
      <h1 className="text-xl font-bold tracking-tight">Analytics Pro</h1>
      <div className="flex items-center gap-2">
        {/* TODO: Theme switcher, user menu, notifications */}
        <Button>Se connecter</Button>
      </div>
    </header>
  )
}
