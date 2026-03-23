import { Map } from 'lucide-react'
import type { Run } from '../../api'
import { StatusBadge } from '../ui/StatusBadge'
import { useGeocoding } from '../../hooks/useGeocoding'
import { useApp } from '../../contexts/AppContext'

const ANALYSIS_EMOJI: Record<string, string> = {
  deforestation: '🌲',
  ice_melting: '🧊',
  flooding: '🌊',
  drought: '🏜️',
  wildfire: '🔥',
}

const ANALYSIS_LABEL: Record<string, string> = {
  deforestation: 'Deforestation Detection',
  ice_melting: 'Arctic Ice Melting',
  flooding: 'Flood Detection',
  drought: 'Drought Monitoring',
  wildfire: 'Wildfire Detection',
}

interface RunCardProps {
  run: Run
  selected?: boolean
  onClick?: () => void
  confidence?: number
}

function StaticMapThumb({ bbox, apiKey }: { bbox: number[]; apiKey: string }) {
  if (!apiKey || apiKey === 'YOUR_GOOGLE_MAPS_API_KEY_HERE') {
    return (
      <div className="w-full h-28 bg-cv-surface flex items-center justify-center border-b border-cv-border">
        <Map className="w-6 h-6 text-cv-text-dim" />
      </div>
    )
  }
  const lat = (bbox[1] + bbox[3]) / 2
  const lon = (bbox[0] + bbox[2]) / 2
  const path = `color:0x22c55ecc|weight:2|${bbox[1]},${bbox[0]}|${bbox[3]},${bbox[0]}|${bbox[3]},${bbox[2]}|${bbox[1]},${bbox[2]}|${bbox[1]},${bbox[0]}`
  const src = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lon}&zoom=7&size=400x150&maptype=satellite&path=${encodeURIComponent(path)}&key=${apiKey}`
  return <img src={src} alt="" className="w-full h-28 object-cover border-b border-cv-border" loading="lazy" />
}

export function RunCard({ run, selected, onClick, confidence }: RunCardProps) {
  const { googleMapsApiKey } = useApp()
  const bbox: number[] | null = run.bbox ? (() => { try { return JSON.parse(run.bbox!) } catch { return null } })() : null
  const regionName = useGeocoding(bbox, googleMapsApiKey)

  const date = new Date(run.created_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  })

  const isRunning = run.status === 'running'

  return (
    <button
      onClick={onClick}
      className={`text-left w-full rounded-xl border overflow-hidden transition-all duration-150 ${
        selected
          ? 'border-cv-primary shadow-glow'
          : 'border-cv-border hover:border-cv-border-strong'
      } bg-cv-card hover:bg-cv-card-hover`}
      aria-label={`Run #${run.id} - ${ANALYSIS_LABEL[run.analysis_type] ?? run.analysis_type}`}
    >
      {/* Map thumbnail */}
      {bbox ? (
        <StaticMapThumb bbox={bbox} apiKey={googleMapsApiKey} />
      ) : (
        <div className="w-full h-28 bg-cv-surface flex items-center justify-center border-b border-cv-border">
          <Map className="w-6 h-6 text-cv-text-dim" />
        </div>
      )}

      {/* Card body */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-cv-text-primary">#{run.id}</span>
          <div className="flex items-center gap-2">
            <StatusBadge status={run.status} />
            <span className="text-xs text-cv-text-dim">{date}</span>
          </div>
        </div>

        <div className="text-xs text-cv-text-secondary">
          <span className="mr-1.5">{ANALYSIS_EMOJI[run.analysis_type] ?? '📡'}</span>
          {ANALYSIS_LABEL[run.analysis_type] ?? run.analysis_type}
        </div>

        {regionName && (
          <div className="text-xs text-cv-text-dim flex items-center gap-1">
            <span>📍</span>
            <span className="truncate">{regionName}</span>
          </div>
        )}

        {/* Confidence bar */}
        {confidence !== undefined && confidence > 0 && (
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-cv-text-dim">Confidence</span>
              <span className="text-cv-text-secondary">{(confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-cv-border rounded-full h-1.5">
              <div
                className="h-1.5 rounded-full bg-cv-primary"
                style={{ width: `${Math.min(confidence * 100, 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* Running indicator */}
        {isRunning && (
          <div className="w-full bg-cv-border rounded-full h-1 overflow-hidden">
            <div className="h-1 bg-amber-400 rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        )}
      </div>
    </button>
  )
}
