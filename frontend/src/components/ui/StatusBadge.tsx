import { CheckCircle, XCircle, Clock } from 'lucide-react'

export type RunStatus = 'running' | 'completed' | 'failed' | 'pending'

const config: Record<RunStatus, { label: string; classes: string; icon: React.ReactNode }> = {
  completed: {
    label: 'Completed',
    classes: 'bg-green-950/60 text-green-400 border-green-700/40',
    icon: <CheckCircle className="w-3 h-3" />,
  },
  failed: {
    label: 'Failed',
    classes: 'bg-red-950/60 text-red-400 border-red-700/40',
    icon: <XCircle className="w-3 h-3" />,
  },
  running: {
    label: 'Running',
    classes: 'bg-amber-950/60 text-amber-400 border-amber-700/40',
    icon: <span className="w-2 h-2 rounded-full bg-amber-400 pulse-dot inline-block" />,
  },
  pending: {
    label: 'Pending',
    classes: 'bg-zinc-900/60 text-zinc-400 border-zinc-700/40',
    icon: <Clock className="w-3 h-3" />,
  },
}

export function StatusBadge({ status }: { status: RunStatus | string }) {
  const s = (config[status as RunStatus] ?? config.pending)
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full border ${s.classes}`}
      aria-label={`Status: ${s.label}`}
    >
      {s.icon}
      {s.label}
    </span>
  )
}
