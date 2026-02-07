import { useEffect, useMemo, useState } from 'react'
import { getRun, health, listRuns, predictJson, predictUpload } from './api'

type Tab = 'bbox' | 'upload' | 'runs'

type Toast = { type: 'success' | 'error'; message: string }

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

function Card(props: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-base-800 bg-base-900/70 shadow-soft backdrop-blur px-5 py-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base-100 font-semibold tracking-tight">{props.title}</h2>
        {props.right}
      </div>
      <div className="mt-4">{props.children}</div>
    </div>
  )
}

function Field(props: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-sm font-medium text-base-200">{props.label}</div>
        {props.hint ? <div className="text-xs text-base-200/60">{props.hint}</div> : null}
      </div>
      <div className="mt-2">{props.children}</div>
    </label>
  )
}

function Button(props: {
  children: React.ReactNode
  onClick?: () => void
  type?: 'button' | 'submit'
  disabled?: boolean
  variant?: 'primary' | 'ghost'
}) {
  const variant = props.variant ?? 'primary'
  return (
    <button
      type={props.type ?? 'button'}
      disabled={props.disabled}
      onClick={props.onClick}
      className={cx(
        'inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-semibold transition',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        variant === 'primary' && 'bg-brand-600 hover:bg-brand-500 text-base-100',
        variant === 'ghost' && 'bg-base-800/60 hover:bg-base-800 text-base-100 border border-base-800',
      )}
    >
      {props.children}
    </button>
  )
}

function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cx(
        'w-full rounded-xl border border-base-800 bg-base-950/40 px-3 py-2 text-sm text-base-100',
        'placeholder:text-base-200/50 focus:outline-none focus:ring-2 focus:ring-ocean-500/40',
        props.className,
      )}
    />
  )
}

function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cx(
        'w-full rounded-xl border border-base-800 bg-base-950/40 px-3 py-2 text-sm text-base-100',
        'placeholder:text-base-200/50 focus:outline-none focus:ring-2 focus:ring-ocean-500/40',
        props.className,
      )}
    />
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>('bbox')
  const [apiOk, setApiOk] = useState<boolean | null>(null)

  const [toast, setToast] = useState<Toast | null>(null)
  const showToast = (t: Toast) => {
    setToast(t)
    window.setTimeout(() => setToast(null), 3500)
  }

  const [bboxText, setBboxText] = useState('[-62.0, -3.1, -61.8, -2.9]')
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState('2024-12-31')

  const [uploadFile, setUploadFile] = useState<File | null>(null)

  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<any | null>(null)

  const [runs, setRuns] = useState<any[]>([])
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const [selectedRun, setSelectedRun] = useState<any | null>(null)

  const parsedBBox = useMemo(() => {
    try {
      const v = JSON.parse(bboxText)
      if (Array.isArray(v) && v.length === 4 && v.every((n) => typeof n === 'number')) return v as number[]
      return null
    } catch {
      return null
    }
  }, [bboxText])

  useEffect(() => {
    health()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
  }, [])

  useEffect(() => {
    if (tab !== 'runs') return
    listRuns()
      .then(setRuns)
      .catch((e) => showToast({ type: 'error', message: String(e) }))
  }, [tab])

  useEffect(() => {
    if (selectedRunId == null) return
    getRun(selectedRunId)
      .then(setSelectedRun)
      .catch((e) => showToast({ type: 'error', message: String(e) }))
  }, [selectedRunId])

  const runBBoxPredict = async () => {
    if (!parsedBBox) {
      showToast({ type: 'error', message: 'BBox must be valid JSON: [minLon, minLat, maxLon, maxLat]' })
      return
    }

    setBusy(true)
    setResult(null)
    try {
      const res = await predictJson({
        kind: 'bbox',
        bbox: parsedBBox,
        start_date: startDate,
        end_date: endDate,
      })
      setResult(res)
      showToast({ type: 'success', message: `Run created (#${res.run_id})` })
    } catch (e) {
      showToast({ type: 'error', message: String(e) })
    } finally {
      setBusy(false)
    }
  }

  const runUploadPredict = async () => {
    if (!uploadFile) {
      showToast({ type: 'error', message: 'Choose a file first.' })
      return
    }

    setBusy(true)
    setResult(null)
    try {
      const res = await predictUpload({
        file: uploadFile,
        kind: 'upload',
        bbox: parsedBBox ?? undefined,
        start_date: startDate,
        end_date: endDate,
      })
      setResult(res)
      showToast({ type: 'success', message: `Upload processed (#${res.run_id})` })
    } catch (e) {
      showToast({ type: 'error', message: String(e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="text-sm text-base-200/70">ClimateVision</div>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight text-base-100">
              Deforestation Monitoring Dashboard
            </h1>
            <div className="mt-2 text-sm text-base-200/70">
              API:{' '}
              {apiOk === null ? (
                <span className="text-base-200">checking…</span>
              ) : apiOk ? (
                <span className="text-brand-500">connected</span>
              ) : (
                <span className="text-danger-500">offline</span>
              )}
            </div>
          </div>

          <div className="flex gap-2">
            <Button variant={tab === 'bbox' ? 'primary' : 'ghost'} onClick={() => setTab('bbox')}>
              BBox + Dates
            </Button>
            <Button variant={tab === 'upload' ? 'primary' : 'ghost'} onClick={() => setTab('upload')}>
              Upload
            </Button>
            <Button variant={tab === 'runs' ? 'primary' : 'ghost'} onClick={() => setTab('runs')}>
              Runs
            </Button>
          </div>
        </header>

        <main className="mt-8 grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            {tab === 'bbox' ? (
              <Card
                title="Create run (BBox + date range)"
                right={<span className="text-xs text-base-200/60">POST /api/predict</span>}
              >
                <div className="grid gap-4">
                  <Field label="Bounding box" hint="JSON: [minLon, minLat, maxLon, maxLat]">
                    <Textarea
                      rows={3}
                      value={bboxText}
                      onChange={(e) => setBboxText(e.target.value)}
                    />
                    {!parsedBBox ? (
                      <div className="mt-2 text-xs text-amber-500">BBox is not valid JSON array of 4 numbers.</div>
                    ) : null}
                  </Field>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <Field label="Start date">
                      <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                    </Field>
                    <Field label="End date">
                      <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                    </Field>
                  </div>

                  <div className="flex items-center justify-end gap-2">
                    <Button disabled={busy} onClick={runBBoxPredict}>
                      {busy ? 'Running…' : 'Run prediction'}
                    </Button>
                  </div>
                </div>
              </Card>
            ) : null}

            {tab === 'upload' ? (
              <Card
                title="Create run (Upload + optional bbox/dates)"
                right={<span className="text-xs text-base-200/60">POST /api/predict/upload</span>}
              >
                <div className="grid gap-4">
                  <Field label="File">
                    <Input
                      type="file"
                      onChange={(e) => {
                        const f = e.target.files?.[0] ?? null
                        setUploadFile(f)
                      }}
                    />
                    {uploadFile ? (
                      <div className="mt-2 text-xs text-base-200/60">
                        Selected: <span className="text-base-200">{uploadFile.name}</span>
                      </div>
                    ) : null}
                  </Field>

                  <Field label="Bounding box (optional)" hint="JSON used to annotate the run">
                    <Textarea
                      rows={3}
                      value={bboxText}
                      onChange={(e) => setBboxText(e.target.value)}
                    />
                  </Field>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <Field label="Start date (optional)">
                      <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                    </Field>
                    <Field label="End date (optional)">
                      <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                    </Field>
                  </div>

                  <div className="flex items-center justify-end gap-2">
                    <Button disabled={busy} onClick={runUploadPredict}>
                      {busy ? 'Uploading…' : 'Upload + run'}
                    </Button>
                  </div>
                </div>
              </Card>
            ) : null}

            {tab === 'runs' ? (
              <Card title="Run history">
                <div className="grid gap-3">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-base-200/70">Latest runs</div>
                    <Button
                      variant="ghost"
                      onClick={() =>
                        listRuns()
                          .then(setRuns)
                          .catch((e) => showToast({ type: 'error', message: String(e) }))
                      }
                    >
                      Refresh
                    </Button>
                  </div>

                  <div className="grid gap-2">
                    {runs.length === 0 ? (
                      <div className="text-sm text-base-200/60">No runs yet.</div>
                    ) : (
                      runs.map((r) => (
                        <button
                          key={r.id}
                          onClick={() => setSelectedRunId(r.id)}
                          className={cx(
                            'text-left rounded-xl border border-base-800 bg-base-950/30 px-4 py-3 transition',
                            'hover:bg-base-950/50',
                          )}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-sm font-semibold text-base-100">Run #{r.id}</div>
                            <div className="text-xs text-base-200/60">{r.status}</div>
                          </div>
                          <div className="mt-1 text-xs text-base-200/60">kind: {r.kind}</div>
                          <div className="mt-1 text-xs text-base-200/60">created: {r.created_at}</div>
                        </button>
                      ))
                    )}
                  </div>

                  {selectedRun ? (
                    <div className="mt-4 rounded-xl border border-base-800 bg-base-950/30 p-4">
                      <div className="text-sm font-semibold text-base-100">Run details</div>
                      <pre className="mt-3 overflow-auto text-xs text-base-200/80">
                        {JSON.stringify(selectedRun, null, 2)}
                      </pre>
                    </div>
                  ) : null}
                </div>
              </Card>
            ) : null}
          </div>

          <div className="lg:col-span-1">
            <Card title="Latest response">
              {result ? (
                <pre className="max-h-[520px] overflow-auto text-xs text-base-200/80">
                  {JSON.stringify(result, null, 2)}
                </pre>
              ) : (
                <div className="text-sm text-base-200/60">Run a prediction to see the response here.</div>
              )}
            </Card>

            <div className="mt-6 rounded-2xl border border-base-800 bg-base-900/50 p-4">
              <div className="text-sm font-semibold text-base-100">Notes</div>
              <div className="mt-2 text-sm text-base-200/70">
                This UI calls:
                <div className="mt-2 grid gap-1 text-xs text-base-200/60">
                  <div>GET /api/health</div>
                  <div>POST /api/predict</div>
                  <div>POST /api/predict/upload</div>
                  <div>GET /api/runs</div>
                  <div>GET /api/runs/:id</div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

      {toast ? (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2">
          <div
            className={cx(
              'rounded-xl px-4 py-3 text-sm font-semibold shadow-soft border',
              toast.type === 'success' && 'bg-brand-600/20 border-brand-600/40 text-base-100',
              toast.type === 'error' && 'bg-danger-500/20 border-danger-500/40 text-base-100',
            )}
          >
            {toast.message}
          </div>
        </div>
      ) : null}
    </div>
  )
}
