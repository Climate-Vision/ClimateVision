import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import type { AnalysisType } from '../api'
import { predictJson } from '../api'
import { MapBBoxPicker } from '../components/Map/MapBBoxPicker'
import { AnalysisTypeSelector } from '../components/ui/AnalysisTypeSelector'
import { ResultsPanel } from '../components/results/ResultsPanel'
import { ErrorBoundary } from '../components/ui/ErrorBoundary'
import { useToast } from '../contexts/ToastContext'
import { useApp } from '../contexts/AppContext'
import type { Run } from '../api'
import { ApiError } from '../components/ui/ApiError'

const PRESETS = [
  { label: 'Last 30d', days: 30 },
  { label: 'Last 90d', days: 90 },
  { label: 'Last year', days: 365 },
]

function toISO(date: Date) {
  return date.toISOString().split('T')[0]
}

function SectionLabel({ step, label }: { step: number; label: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="w-7 h-7 rounded-full bg-cv-primary-muted flex items-center justify-center text-xs font-bold text-cv-primary shrink-0">
        {step}
      </div>
      <h3 className="text-sm font-semibold text-cv-text-primary uppercase tracking-wide">{label}</h3>
    </div>
  )
}

export default function NewAnalysis() {
  const { showToast } = useToast()
  const { googleMapsApiKey } = useApp()
  const navigate = useNavigate()

  const [analysisType, setAnalysisType] = useState<AnalysisType>('deforestation')
  const [bbox, setBbox] = useState<number[] | null>(null)
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState('2024-12-31')
  const [busy, setBusy] = useState(false)
  const [resultRun, setResultRun] = useState<Run | null>(null)
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)

  const canSubmit = bbox !== null && startDate && endDate && !busy

  const applyPreset = (days: number) => {
    const end = new Date()
    const start = new Date()
    start.setDate(start.getDate() - days)
    setStartDate(toISO(start))
    setEndDate(toISO(end))
  }

  const handleSubmit = async () => {
    if (!canSubmit) return
    if (startDate > endDate) {
      showToast('error', 'Start date must be before end date.')
      return
    }

    
    setResultRun(null)
    setResultPayload(null)

    try {
      setBusy(true);
      setError(null);
      const res = await predictJson({ kind: 'bbox', analysis_type: analysisType, bbox: bbox!, start_date: startDate, end_date: endDate })
      setResultPayload(res.result)
      
      // Construct a minimal Run object for the results panel
      setResultRun({
        id: res.run_id,
        kind: 'bbox',
        status: 'completed',
        analysis_type: analysisType,
        bbox: JSON.stringify(bbox),
        start_date: startDate,
        end_date: endDate,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      showToast('success', `Run #${res.run_id} complete!`, {
        label: 'View in history',
        onClick: () => navigate('/runs'),
      })
    } catch (e:any) {
      const message = e?.response?.data?.detail || e?.message || "Something went wrong";
      setError(message);
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        <ApiError message={error} onDismiss={() => setError(null)} />
        
      {/* Step 1 — Analysis Type */}
      <section>
        <SectionLabel step={1} label="Analysis Type" />
        <AnalysisTypeSelector value={analysisType} onChange={setAnalysisType} />
      </section>

      {/* Step 2 — Region */}
      <section>
        <SectionLabel step={2} label="Select Region" />
        <ErrorBoundary section="Map">
          <MapBBoxPicker value={bbox} onChange={setBbox} apiKey={googleMapsApiKey} />
        </ErrorBoundary>
      </section>

      {/* Step 3 — Date Range */}
      <section>
        <SectionLabel step={3} label="Date Range" />
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-cv-text-secondary mb-1.5">Start date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-cv-card border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary focus:outline-none focus:border-cv-primary transition"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-cv-text-secondary mb-1.5">End date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-cv-card border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary focus:outline-none focus:border-cv-primary transition"
              />
            </div>
          </div>
          <div className="flex gap-2 flex-wrap">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => applyPreset(p.days)}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cv-card border border-cv-border text-cv-text-secondary hover:text-cv-primary hover:border-cv-primary transition"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="w-full h-12 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
          bg-cv-primary text-white hover:bg-cv-primary-hover disabled:opacity-40 disabled:cursor-not-allowed shadow-glow"
      >
        {busy ? (
          <>
            <Loader2 className="w-4 h-4 spinner" />
            Running analysis…
          </>
        ) : (
          'Run Prediction →'
        )}
      </button>

      {/* Inline hint if bbox not set */}
      {!bbox && (
        <p className="text-xs text-cv-text-dim text-center -mt-4">
          Draw a region on the map above to enable prediction
        </p>
      )}

      {/* Results */}
      {resultRun && (
        <section className="border-t border-cv-border pt-8">
          <h3 className="text-sm font-semibold text-cv-text-secondary uppercase tracking-wide mb-4">Results</h3>
          <ErrorBoundary section="Results">
            <ResultsPanel
              run={resultRun}
              payload={resultPayload as Record<string, unknown> & { inference?: Record<string, number> } | null}
              onRunAgain={() => {
                setResultRun(null)
                setResultPayload(null)
              }}
            />
          </ErrorBoundary>
        </section>
      )}
    </div>
  )
}
