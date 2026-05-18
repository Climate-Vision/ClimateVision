// frontend/src/components/ui/ApiError.tsx
import { AlertCircle, X } from 'lucide-react'
import { useState } from 'react'

type ApiErrorProps = {
  message: string
}

export function ApiError({ message }: ApiErrorProps) {
  const [visible, setVisible] = useState(true)

  if (!visible) return null

  return (
    <div className="flex bg-red-950/90 items-start justify-between overflow-hidden gap-3 rounded-xl sm:text-xs md:text-base border border-red-500/70 p-3 text-white/80">
      <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />

      <div className="flex-1 text-sm break-words">{message}</div>

      <button type="button" aria-label="Dismiss" onClick={() => setVisible(false)} className="text-white/50 hover:text-red-900">
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
