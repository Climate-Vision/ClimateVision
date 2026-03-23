import type { AnalysisType } from '../../api'
import { CheckCircle } from 'lucide-react'

interface AnalysisOption {
  value: AnalysisType
  emoji: string
  label: string
  description: string
  enabled: boolean
}

const OPTIONS: AnalysisOption[] = [
  { value: 'deforestation', emoji: '🌲', label: 'Deforestation Detection', description: 'Track forest cover loss', enabled: true },
  { value: 'ice_melting', emoji: '🧊', label: 'Arctic Ice Melting', description: 'Monitor polar ice extent', enabled: true },
  { value: 'flooding', emoji: '🌊', label: 'Flood Detection', description: 'Identify inundated areas', enabled: true },
  { value: 'drought', emoji: '🏜️', label: 'Drought Monitoring', description: 'Measure vegetation stress', enabled: false },
  { value: 'wildfire', emoji: '🔥', label: 'Wildfire Detection', description: 'Detect active burn zones', enabled: false },
]

export function AnalysisTypeSelector({
  value,
  onChange,
}: {
  value: AnalysisType
  onChange: (v: AnalysisType) => void
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      {OPTIONS.map((opt) => {
        const selected = value === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            disabled={!opt.enabled}
            onClick={() => opt.enabled && onChange(opt.value)}
            className={`relative flex flex-col items-start gap-1 p-3 rounded-xl border text-left transition-all ${
              selected
                ? 'border-cv-primary bg-cv-primary-muted shadow-glow'
                : opt.enabled
                ? 'border-cv-border bg-cv-card hover:border-cv-border-strong hover:bg-cv-card-hover'
                : 'border-cv-border bg-cv-card opacity-40 cursor-not-allowed'
            }`}
            aria-pressed={selected}
            aria-label={opt.label}
          >
            {selected && (
              <CheckCircle className="absolute top-2 right-2 w-4 h-4 text-cv-primary" />
            )}
            {!opt.enabled && (
              <span className="absolute top-1.5 right-1.5 text-xs bg-cv-surface px-1.5 py-0.5 rounded text-cv-text-dim border border-cv-border">
                Soon
              </span>
            )}
            <span className="text-xl">{opt.emoji}</span>
            <span className="text-xs font-semibold text-cv-text-primary leading-tight">{opt.label}</span>
            <span className="text-xs text-cv-text-secondary leading-tight">{opt.description}</span>
          </button>
        )
      })}
    </div>
  )
}
