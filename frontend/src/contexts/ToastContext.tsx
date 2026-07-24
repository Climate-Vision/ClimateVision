import { createContext, useContext, useState, useCallback, useRef } from 'react'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  message: string
  action?: { label: string; onClick: () => void }
}

interface ToastContextValue {
  toasts: Toast[]
  showToast: (type: ToastType, message: string, action?: Toast['action']) => void
  dismissToast: (id: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used inside ToastProvider')
  return ctx
}

const icons: Record<ToastType, typeof CheckCircle> = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
}

const styles: Record<ToastType, string> = {
  success: 'border-green-200 bg-green-50 text-green-800',
  error: 'border-red-200 bg-red-50 text-red-800',
  warning: 'border-amber-200 bg-amber-50 text-amber-800',
  info: 'border-blue-200 bg-blue-50 text-blue-800',
}

const iconStyles: Record<ToastType, string> = {
  success: 'text-green-600',
  error: 'text-red-600',
  warning: 'text-amber-600',
  info: 'text-blue-600',
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const Icon = icons[toast.type]
  return (
    <div
      className={`flex items-start gap-3 rounded-xl border px-4 py-3 shadow-lg backdrop-blur-sm text-sm animate-fade-in ${styles[toast.type]}`}
      role="alert"
      aria-live="polite"
    >
      <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${iconStyles[toast.type]}`} />
      <span className="flex-1">{toast.message}</span>
      {toast.action && (
        <button
          onClick={toast.action.onClick}
          className="underline opacity-80 hover:opacity-100 transition shrink-0"
        >
          {toast.action.label}
        </button>
      )}
      <button onClick={onDismiss} className="shrink-0 opacity-60 hover:opacity-100 transition">
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
    clearTimeout(timers.current[id])
    delete timers.current[id]
  }, [])

  const showToast = useCallback(
    (type: ToastType, message: string, action?: Toast['action']) => {
      const id = Math.random().toString(36).slice(2)
      setToasts((prev) => [...prev, { id, type, message, action }])
      timers.current[id] = setTimeout(() => dismissToast(id), 5000)
    },
    [dismissToast],
  )

  return (
    <ToastContext.Provider value={{ toasts, showToast, dismissToast }}>
      {children}
      <div
        className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none"
        aria-label="Notifications"
      >
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto">
            <ToastItem toast={t} onDismiss={() => dismissToast(t.id)} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
