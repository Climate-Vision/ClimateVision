import { createContext, useContext, useState, useCallback } from 'react'
import type { AnalysisType } from '../api'

interface AppContextValue {
  theme: 'dark' | 'light'
  toggleTheme: () => void
  defaultAnalysisType: AnalysisType
  setDefaultAnalysisType: (t: AnalysisType) => void
  googleMapsApiKey: string
  apiBaseUrl: string
}

const AppContext = createContext<AppContextValue | null>(null)

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be inside AppProvider')
  return ctx
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [defaultAnalysisType, setDefaultAnalysisType] = useState<AnalysisType>('deforestation')

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }, [])

  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY ?? ''
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

  return (
    <AppContext.Provider
      value={{ theme, toggleTheme, defaultAnalysisType, setDefaultAnalysisType, googleMapsApiKey, apiBaseUrl }}
    >
      {children}
    </AppContext.Provider>
  )
}
