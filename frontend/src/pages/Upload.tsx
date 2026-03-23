import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { CloudUpload, FileText, X, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'
import type { AnalysisType } from '../api'
import { predictUpload } from '../api'
import { AnalysisTypeSelector } from '../components/ui/AnalysisTypeSelector'
import { MapBBoxPicker } from '../components/map/MapBBoxPicker'
import { ErrorBoundary } from '../components/ui/ErrorBoundary'
import { useToast } from '../contexts/ToastContext'
import { useApp } from '../contexts/AppContext'

const ACCEPTED = ['.tif', '.tiff', '.geotiff', '.nc', '.hdf5']
const MAX_MB = 500

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default function Upload() {
  const { showToast } = useToast()
  const { googleMapsApiKey } = useApp()
  const navigate = useNavigate()

  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [metaOpen, setMetaOpen] = useState(false)
  const [analysisType, setAnalysisType] = useState<AnalysisType>('deforestation')
  const [bbox, setBbox] = useState<number[] | null>(null)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const validateFile = (f: File): string | null => {
    const ext = '.' + f.name.split('.').pop()?.toLowerCase()
    if (!ACCEPTED.includes(ext)) return `Unsupported format. Accepted: ${ACCEPTED.join(', ')}`
    if (f.size > MAX_MB * 1024 * 1024) return `File too large. Max size: ${MAX_MB} MB`
    return null
  }

  const handleFile = (f: File) => {
    const err = validateFile(f)
    setFileError(err)
    setFile(err ? null : f)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [])

  const handleUpload = async () => {
    if (!file) return
    setBusy(true)
    setUploadProgress(0)

    try {
      const res = await predictUpload({
        file,
        kind: 'upload',
        analysis_type: analysisType,
        bbox: bbox ?? undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      })
      setUploadProgress(100)
      showToast('success', `Upload complete! Run #${res.run_id} created.`, {
        label: 'View run',
        onClick: () => navigate('/runs'),
      })
      setFile(null)
      setUploadProgress(null)
    } catch (e) {
      showToast('error', String(e))
      setUploadProgress(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">

      {/* Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !file && fileInputRef.current?.click()}
        className={`rounded-xl border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center gap-3 py-14 px-8 text-center ${
          dragging
            ? 'border-cv-primary bg-cv-primary-muted/20'
            : file
            ? 'border-cv-border-strong bg-cv-card cursor-default'
            : 'border-cv-border bg-cv-card hover:border-cv-border-strong hover:bg-cv-card-hover'
        }`}
      >
        {!file ? (
          <>
            <CloudUpload className={`w-10 h-10 ${dragging ? 'text-cv-primary' : 'text-cv-text-dim'}`} />
            <div>
              <p className="text-base font-medium text-cv-text-primary">
                Drop satellite imagery here
              </p>
              <p className="text-sm text-cv-text-secondary mt-1">
                or{' '}
                <span className="text-cv-primary underline cursor-pointer">browse files</span>
              </p>
            </div>
            <p className="text-xs text-cv-text-dim">
              Supported: {ACCEPTED.join('  ')} · Max {MAX_MB} MB
            </p>
          </>
        ) : (
          <div className="flex items-center gap-4 w-full max-w-md">
            <FileText className="w-8 h-8 text-cv-primary shrink-0" />
            <div className="flex-1 text-left">
              <p className="text-sm font-medium text-cv-text-primary truncate">{file.name}</p>
              <p className="text-xs text-cv-text-secondary">{formatBytes(file.size)} · ✓ Valid format</p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setFile(null) }}
              className="text-cv-text-dim hover:text-red-400 transition"
              aria-label="Remove file"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
      />

      {fileError && (
        <div className="rounded-lg bg-red-950/40 border border-red-700/40 px-4 py-3 text-sm text-red-300">
          {fileError}
        </div>
      )}

      {/* Optional metadata */}
      <div className="border border-cv-border rounded-xl overflow-hidden">
        <button
          onClick={() => setMetaOpen((o) => !o)}
          className="flex items-center justify-between w-full px-4 py-3 text-sm font-medium text-cv-text-secondary hover:text-cv-text-primary hover:bg-cv-card transition"
        >
          <span>Add metadata (optional)</span>
          {metaOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        {metaOpen && (
          <div className="px-4 pb-4 border-t border-cv-border space-y-5 pt-4">
            <div>
              <p className="text-xs font-medium text-cv-text-secondary mb-3 uppercase tracking-wide">Analysis Type</p>
              <AnalysisTypeSelector value={analysisType} onChange={setAnalysisType} />
            </div>
            <div>
              <p className="text-xs font-medium text-cv-text-secondary mb-3 uppercase tracking-wide">Region</p>
              <ErrorBoundary section="Map">
                <MapBBoxPicker value={bbox} onChange={setBbox} apiKey={googleMapsApiKey} />
              </ErrorBoundary>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-cv-text-secondary mb-1.5">Start date</label>
                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
                  className="w-full bg-cv-card border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary focus:outline-none focus:border-cv-primary transition" />
              </div>
              <div>
                <label className="block text-xs font-medium text-cv-text-secondary mb-1.5">End date</label>
                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
                  className="w-full bg-cv-card border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary focus:outline-none focus:border-cv-primary transition" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!file || busy}
        className="w-full h-12 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
          bg-cv-primary text-white hover:bg-cv-primary-hover disabled:opacity-40 disabled:cursor-not-allowed shadow-glow"
      >
        {busy ? (
          <>
            <Loader2 className="w-4 h-4 spinner" />
            Uploading…
          </>
        ) : (
          'Upload + Run →'
        )}
      </button>

      {/* Progress bar */}
      {uploadProgress !== null && (
        <div className="w-full bg-cv-border rounded-full h-1.5 overflow-hidden">
          <div
            className="h-1.5 bg-cv-primary rounded-full transition-all duration-300"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}
    </div>
  )
}
