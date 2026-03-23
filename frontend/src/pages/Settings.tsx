import { useState } from 'react'
import { Eye, EyeOff, ExternalLink } from 'lucide-react'
import type { AnalysisType } from '../api'
import { useApp } from '../contexts/AppContext'
import { AnalysisTypeSelector } from '../components/ui/AnalysisTypeSelector'
import { useToast } from '../contexts/ToastContext'

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-cv-card border border-cv-border rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-cv-border">
        <h3 className="text-sm font-semibold text-cv-text-primary">{title}</h3>
      </div>
      <div className="p-5 space-y-4">{children}</div>
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-1.5">
        <label className="text-xs font-medium text-cv-text-secondary">{label}</label>
        {hint && <span className="text-xs text-cv-text-dim">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

export default function Settings() {
  const { showToast } = useToast()
  const { theme, toggleTheme, defaultAnalysisType, setDefaultAnalysisType, googleMapsApiKey, apiBaseUrl } = useApp()

  const [showKey, setShowKey] = useState(false)
  const [testingApi, setTestingApi] = useState(false)
  const [localApiKey, setLocalApiKey] = useState(googleMapsApiKey)
  const [localApiUrl, setLocalApiUrl] = useState(apiBaseUrl)
  const [exportFormat, setExportFormat] = useState<'geojson' | 'csv' | 'shapefile'>('geojson')
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [mapStyle, setMapStyle] = useState<'satellite' | 'dark' | 'terrain'>('satellite')

  const testConnection = async () => {
    setTestingApi(true)
    try {
      const res = await fetch(`${localApiUrl}/api/health`)
      if (res.ok) {
        showToast('success', 'API connection successful!')
      } else {
        showToast('error', `API returned ${res.status}`)
      }
    } catch {
      showToast('error', 'Could not reach the API')
    } finally {
      setTestingApi(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-8 space-y-5">

      <Section title="API Configuration">
        <Field label="Google Maps API Key" hint="Used for maps and geocoding">
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={localApiKey}
              onChange={(e) => setLocalApiKey(e.target.value)}
              placeholder="AIza..."
              className="w-full bg-cv-surface border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary pr-10 focus:outline-none focus:border-cv-primary transition"
            />
            <button
              onClick={() => setShowKey((v) => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-cv-text-dim hover:text-cv-text-secondary transition"
            >
              {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
          <p className="text-xs text-cv-text-dim mt-1">Set via <code className="text-cv-primary">VITE_GOOGLE_MAPS_API_KEY</code> in .env to persist</p>
        </Field>

        <Field label="Backend API URL">
          <div className="flex gap-2">
            <input
              type="url"
              value={localApiUrl}
              onChange={(e) => setLocalApiUrl(e.target.value)}
              className="flex-1 bg-cv-surface border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary focus:outline-none focus:border-cv-primary transition"
            />
            <button
              onClick={testConnection}
              disabled={testingApi}
              className="px-4 py-2 rounded-lg bg-cv-primary-muted text-cv-primary text-sm font-medium hover:bg-green-800/40 transition disabled:opacity-50"
            >
              {testingApi ? 'Testing…' : 'Test'}
            </button>
          </div>
        </Field>
      </Section>

      <Section title="Default Analysis Preferences">
        <Field label="Default Analysis Type">
          <AnalysisTypeSelector value={defaultAnalysisType} onChange={setDefaultAnalysisType} />
        </Field>
      </Section>

      <Section title="Export Settings">
        <Field label="Preferred Format">
          <div className="flex gap-3">
            {(['geojson', 'csv', 'shapefile'] as const).map((fmt) => (
              <label key={fmt} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="format"
                  value={fmt}
                  checked={exportFormat === fmt}
                  onChange={() => setExportFormat(fmt)}
                  className="accent-cv-primary"
                />
                <span className="text-sm text-cv-text-secondary capitalize">{fmt}</span>
              </label>
            ))}
          </div>
        </Field>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={includeMetadata}
            onChange={(e) => setIncludeMetadata(e.target.checked)}
            className="accent-cv-primary w-4 h-4"
          />
          <span className="text-sm text-cv-text-secondary">Include metadata in exports</span>
        </label>
      </Section>

      <Section title="Appearance">
        <Field label="Theme">
          <div className="flex gap-3">
            {(['dark', 'light'] as const).map((t) => (
              <label key={t} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="theme"
                  value={t}
                  checked={theme === t}
                  onChange={toggleTheme}
                  className="accent-cv-primary"
                />
                <span className="text-sm text-cv-text-secondary capitalize">{t}</span>
              </label>
            ))}
          </div>
        </Field>

        <Field label="Map Style">
          <div className="flex gap-3">
            {(['satellite', 'dark', 'terrain'] as const).map((s) => (
              <label key={s} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="mapStyle"
                  value={s}
                  checked={mapStyle === s}
                  onChange={() => setMapStyle(s)}
                  className="accent-cv-primary"
                />
                <span className="text-sm text-cv-text-secondary capitalize">{s}</span>
              </label>
            ))}
          </div>
        </Field>
      </Section>

      <Section title="About">
        <div className="space-y-2 text-sm text-cv-text-secondary">
          <div className="flex justify-between">
            <span>Version</span>
            <span className="text-cv-text-primary font-mono">0.2.0</span>
          </div>
          <div className="flex justify-between">
            <span>License</span>
            <span className="text-cv-text-primary">MIT</span>
          </div>
          <a
            href="https://github.com/yourusername/ClimateVision"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-cv-primary hover:underline"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            View on GitHub
          </a>
        </div>
      </Section>
    </div>
  )
}
