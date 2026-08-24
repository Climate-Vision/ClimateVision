import { useEffect, useRef, useState } from 'react'
import { Download, Share2, RotateCcw, Map as MapIcon } from 'lucide-react'
import type { Run } from '../../api'
import { StatusBadge } from '../ui/StatusBadge'
import { ConfidenceGauge } from './ConfidenceGauge'
import { useApp } from '../../contexts/AppContext'

interface ResultPayload {
  inference?: {
    forest_percentage?: number
    ice_percentage?: number
    flooded_percentage?: number
    mean_confidence?: number
    image_size?: [number, number]
    forest_pixels?: number
    non_forest_pixels?: number
    ice_pixels?: number
    water_pixels?: number
    flooded_pixels?: number
    dry_pixels?: number
  }
  ndvi_stats?: { NDVI_min: number; NDVI_mean: number; NDVI_max: number }
  region?: { bbox?: number[]; date_range?: string }
  analysis_type?: string
  error?: string
}

interface ResultsPanelProps {
  run: Run
  payload: ResultPayload | null
  onRunAgain?: () => void
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-cv-card border border-cv-border rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs text-cv-text-secondary">{label}</span>
      <span className="text-xl font-bold text-cv-text-primary">{value}</span>
      {sub && <span className="text-xs text-cv-text-dim">{sub}</span>}
    </div>
  )
}

function StaticMapImage({ bbox, apiKey }: { bbox: number[]; apiKey: string }) {
  if (!apiKey || apiKey === 'YOUR_GOOGLE_MAPS_API_KEY_HERE') {
    return (
      <div className="w-full h-48 bg-cv-card border border-cv-border rounded-xl flex items-center justify-center">
        <MapIcon className="w-8 h-8 text-cv-text-dim" />
      </div>
    )
  }
  const lat = (bbox[1] + bbox[3]) / 2
  const lon = (bbox[0] + bbox[2]) / 2
  const path = `color:0x22c55ecc|weight:2|${bbox[1]},${bbox[0]}|${bbox[3]},${bbox[0]}|${bbox[3]},${bbox[2]}|${bbox[1]},${bbox[2]}|${bbox[1]},${bbox[0]}`
  const src = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lon}&zoom=8&size=600x300&maptype=satellite&path=${encodeURIComponent(path)}&key=${apiKey}`
  return (
    <img
      src={src}
      alt="Analyzed region satellite view"
      className="w-full rounded-xl border border-cv-border object-cover"
    />
  )
}

export function ResultsPanel({ run, payload, onRunAgain }: ResultsPanelProps) {
  const { googleMapsApiKey } = useApp()
  const bbox = run.bbox ? JSON.parse(run.bbox) : payload?.region?.bbox ?? null
  const inf = payload?.inference
  const confidence = (inf?.mean_confidence ?? 0) * 100

  const analysisType = run.analysis_type ?? payload?.analysis_type ?? 'deforestation'

  const mainPct =
    analysisType === 'ice_melting'
      ? inf?.ice_percentage ?? 0
      : analysisType === 'flooding'
      ? inf?.flooded_percentage ?? 0
      : inf?.forest_percentage ?? 0

  const mainLabel =
    analysisType === 'ice_melting' ? 'Ice Extent' : analysisType === 'flooding' ? 'Flooded Area' : 'Forest Coverage'

  const totalPixels = (inf?.forest_pixels ?? 0) + (inf?.non_forest_pixels ?? inf?.water_pixels ?? inf?.dry_pixels ?? 0) + (inf?.ice_pixels ?? 0)

  const copyRunLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}/runs#run-${run.id}`)
  }

  const downloadGeoJSON = () => {
    if (!bbox) return
    const geojson = {
      type: 'Feature',
      properties: { run_id: run.id, analysis_type: analysisType, ...payload },
      geometry: {
        type: 'Polygon',
        coordinates: [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]],
      },
    }
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `run-${run.id}.geojson`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (payload?.error && !inf) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-sm font-semibold text-cv-text-primary">Run #{run.id}</span>
          <StatusBadge status={run.status} />
        </div>
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-sm text-red-700">{payload.error}</p>
        </div>
        {onRunAgain && (
          <button onClick={onRunAgain} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-cv-card border border-cv-border text-sm text-cv-text-secondary hover:text-cv-text-primary transition">
            <RotateCcw className="w-4 h-4" />
            Run Again
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm font-semibold text-cv-text-primary">Run #{run.id}</span>
        <StatusBadge status={run.status} />
        <span className="text-xs text-cv-text-dim ml-auto">
          {new Date(run.created_at).toLocaleDateString()}
        </span>
      </div>

      {/* Satellite image */}
      {bbox && (
        <div>
          <h4 className="text-xs font-medium text-cv-text-secondary mb-2 uppercase tracking-wide">Region</h4>
          <StaticMapImage bbox={bbox} apiKey={googleMapsApiKey} />
        </div>
      )}

      {/* Confidence + main metric */}
      <div className="flex items-center gap-6">
        <ConfidenceGauge value={confidence} size={110} />
        <div className="flex-1 space-y-2">
          <div>
            <div className="text-xs text-cv-text-secondary mb-1">{mainLabel}</div>
            <div className="text-3xl font-bold text-cv-primary">{mainPct.toFixed(1)}%</div>
          </div>
          <div className="w-full bg-cv-border rounded-full h-2">
            <div
              className="bg-cv-primary h-2 rounded-full transition-all duration-700"
              style={{ width: `${Math.min(mainPct, 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Key metrics */}
      {inf && (
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Area Analyzed" value={totalPixels ? `${(totalPixels * 0.09).toFixed(0)} km²` : 'N/A'} sub="estimated" />
          <StatCard label="Change Detected" value={`${mainPct.toFixed(1)}%`} sub={mainLabel} />
          {inf.image_size && (
            <StatCard label="Image Size" value={`${inf.image_size[0]}×${inf.image_size[1]}`} sub="pixels" />
          )}
          <StatCard label="Confidence" value={`${confidence.toFixed(0)}%`} sub="mean score" />
        </div>
      )}

      {/* NDVI */}
      {payload?.ndvi_stats && (
        <div className="bg-cv-card border border-cv-border rounded-xl p-4">
          <h4 className="text-xs font-medium text-cv-text-secondary uppercase tracking-wide mb-3">NDVI Statistics</h4>
          <div className="grid grid-cols-3 gap-3 text-center">
            {[
              { label: 'Min', value: payload.ndvi_stats.NDVI_min },
              { label: 'Mean', value: payload.ndvi_stats.NDVI_mean },
              { label: 'Max', value: payload.ndvi_stats.NDVI_max },
            ].map(({ label, value }) => (
              <div key={label}>
                <div className="text-xs text-cv-text-dim">{label}</div>
                <div className={`text-base font-semibold ${value >= 0.3 ? 'text-cv-primary' : value >= 0 ? 'text-amber-600' : 'text-red-600'}`}>
                  {value.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-wrap gap-2">
        {bbox && (
          <button onClick={downloadGeoJSON} className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-cv-card border border-cv-border text-xs text-cv-text-secondary hover:text-cv-text-primary hover:border-cv-border-strong transition">
            <Download className="w-3.5 h-3.5" />
            GeoJSON
          </button>
        )}
        <button onClick={copyRunLink} className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-cv-card border border-cv-border text-xs text-cv-text-secondary hover:text-cv-text-primary hover:border-cv-border-strong transition">
          <Share2 className="w-3.5 h-3.5" />
          Share
        </button>
        {onRunAgain && (
          <button onClick={onRunAgain} className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-cv-card border border-cv-border text-xs text-cv-text-secondary hover:text-cv-text-primary hover:border-cv-border-strong transition">
            <RotateCcw className="w-3.5 h-3.5" />
            Run Again
          </button>
        )}
      </div>
    </div>
  )
}
