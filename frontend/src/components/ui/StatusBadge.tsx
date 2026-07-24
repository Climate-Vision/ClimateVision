import { CheckCircle, XCircle, Clock } from 'lucide-react'

export type RunStatus = 'running' | 'completed' | 'failed' | 'pending'

const config: Record<RunStatus, { label: string; classes: string; icon: React.ReactNode }> = {
  completed: {
    label: 'Completed',
    classes: 'bg-green-50 text-green-700 border-green-200',
    icon: <CheckCircle className="w-3 h-3" />,
  },
  failed: {
    label: 'Failed',
    classes: 'bg-red-50 text-red-700 border-red-200',
    icon: <XCircle className="w-3 h-3" />,
  },
  running: {
    label: 'Running',
    classes: 'bg-amber-50 text-amber-700 border-amber-200',
    icon: <span className="w-2 h-2 rounded-full bg-amber-500 pulse-dot inline-block" />,
  },
  pending: {
    label: 'Pending',
    classes: 'bg-zinc-100 text-zinc-600 border-zinc-300',
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
