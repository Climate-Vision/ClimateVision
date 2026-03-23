import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Sun, Moon, Wifi, WifiOff } from 'lucide-react'
import { health } from '../../api'

const PAGE_TITLES: Record<string, string> = {
  '/': 'New Analysis',
  '/upload': 'Upload',
  '/runs': 'Run History',
  '/analytics': 'Analytics',
  '/settings': 'Settings',
}

export function TopBar({ theme, onToggleTheme }: { theme: 'dark' | 'light'; onToggleTheme: () => void }) {
  const location = useLocation()
  const [apiOk, setApiOk] = useState<boolean | null>(null)

  const title = PAGE_TITLES[location.pathname] ?? 'ClimateVision'

  useEffect(() => {
    health()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
    const interval = setInterval(() => {
      health()
        .then(() => setApiOk(true))
        .catch(() => setApiOk(false))
    }, 30_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-cv-border bg-cv-surface/80 backdrop-blur-sm sticky top-0 z-30">
      <h1 className="text-lg font-bold text-cv-text-primary">{title}</h1>

      <div className="flex items-center gap-3">
        {/* API status */}
        <div
          className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${
            apiOk === null
              ? 'border-cv-border text-cv-text-dim'
              : apiOk
              ? 'border-green-700/50 bg-green-950/50 text-green-400'
              : 'border-red-700/50 bg-red-950/50 text-red-400'
          }`}
          title={apiOk === null ? 'Checking API...' : apiOk ? 'API Connected' : 'API Error'}
        >
          {apiOk ? (
            <Wifi className="w-3 h-3" />
          ) : (
            <WifiOff className="w-3 h-3" />
          )}
          <span className="hidden sm:inline">
            {apiOk === null ? 'Checking…' : apiOk ? 'API Connected' : 'API Error'}
          </span>
        </div>

        {/* Theme toggle */}
        <button
          onClick={onToggleTheme}
          className="p-2 rounded-lg text-cv-text-secondary hover:text-cv-text-primary hover:bg-cv-card transition"
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  )
}
