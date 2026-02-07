export type PredictJsonRequest = {
  kind?: string
  bbox?: number[]
  start_date?: string
  end_date?: string
}

const DEFAULT_BASE_URL = ''

export function getApiBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? DEFAULT_BASE_URL
}

export async function health(): Promise<{ status: string }> {
  const res = await fetch(`${getApiBaseUrl()}/api/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function listRuns(): Promise<any[]> {
  const res = await fetch(`${getApiBaseUrl()}/api/runs`)
  if (!res.ok) throw new Error('Failed to load runs')
  return res.json()
}

export async function getRun(runId: number): Promise<any> {
  const res = await fetch(`${getApiBaseUrl()}/api/runs/${runId}`)
  if (!res.ok) throw new Error('Failed to load run')
  return res.json()
}

export async function predictJson(payload: PredictJsonRequest): Promise<any> {
  const res = await fetch(`${getApiBaseUrl()}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Prediction failed')
  return res.json()
}

export async function predictUpload(args: {
  file: File
  kind?: string
  bbox?: number[]
  start_date?: string
  end_date?: string
}): Promise<any> {
  const form = new FormData()
  form.set('kind', args.kind ?? 'upload')
  if (args.bbox) form.set('bbox', JSON.stringify(args.bbox))
  if (args.start_date) form.set('start_date', args.start_date)
  if (args.end_date) form.set('end_date', args.end_date)
  form.set('file', args.file)

  const res = await fetch(`${getApiBaseUrl()}/api/predict/upload`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error('Upload prediction failed')
  return res.json()
}
