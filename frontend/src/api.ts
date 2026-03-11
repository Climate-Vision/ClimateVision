/**
 * ClimateVision API Client
 * 
 * TypeScript client for interacting with the ClimateVision REST API.
 */

// ===== Types =====

export type AnalysisType = 'deforestation' | 'ice_melting' | 'flooding' | 'drought' | 'wildfire'

export type RunStatus = 'pending' | 'running' | 'completed' | 'failed'

export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical'

export interface HealthResponse {
  status: string
  version?: string
  analysis_types?: AnalysisType[]
}

export interface PredictJsonRequest {
  kind?: string
  analysis_type?: AnalysisType
  bbox?: number[]
  start_date?: string
  end_date?: string
}

export interface PredictUploadRequest {
  file: File
  kind?: string
  analysis_type?: AnalysisType
  bbox?: number[]
  start_date?: string
  end_date?: string
}

export interface Run {
  id: number
  kind: string
  status: RunStatus
  analysis_type: AnalysisType
  bbox?: string
  start_date?: string
  end_date?: string
  created_at: string
  updated_at: string
}

export interface RunResult {
  id: number
  run_id: number
  payload: Record<string, unknown>
  mask_path?: string
  created_at: string
}

export interface RunWithResult {
  run: Run
  result: RunResult | null
}

export interface Organization {
  id: number
  name: string
  type: string
  description?: string
  contact_email?: string
  website_url?: string
  active: boolean
  created_at: string
}

export interface OrganizationWithKey extends Organization {
  api_key: string
}

export interface CreateOrganizationRequest {
  name: string
  type?: string
  description?: string
  contact_email?: string
  website_url?: string
  regions_of_interest?: string[]
}

export interface Subscription {
  id: number
  organization_id: number
  name?: string
  bbox: number[]
  analysis_types: AnalysisType[]
  alert_threshold: number
  notification_channel: string
  active: boolean
  created_at: string
}

export interface CreateSubscriptionRequest {
  name?: string
  description?: string
  bbox: number[]
  analysis_types?: AnalysisType[]
  alert_threshold?: number
  notification_channel?: string
  webhook_url?: string
}

export interface Alert {
  id: number
  organization_id: number
  alert_type: string
  severity: AlertSeverity
  title: string
  message: string
  delivered: boolean
  acknowledged: boolean
  created_at: string
}

export interface AnalysisTypeInfo {
  name: AnalysisType
  display_name: string
  description: string
  enabled: boolean
  bands: string[]
  classes: string[]
}

// ===== Configuration =====

const DEFAULT_BASE_URL = ''

export function getApiBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? DEFAULT_BASE_URL
}

// ===== Core Endpoints =====

export async function health(): Promise<HealthResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function listAnalysisTypes(enabledOnly = true): Promise<AnalysisTypeInfo[]> {
  const res = await fetch(`${getApiBaseUrl()}/api/analysis-types?enabled_only=${enabledOnly}`)
  if (!res.ok) throw new Error('Failed to load analysis types')
  return res.json()
}

export async function getAnalysisType(name: string): Promise<AnalysisTypeInfo> {
  const res = await fetch(`${getApiBaseUrl()}/api/analysis-types/${name}`)
  if (!res.ok) throw new Error('Analysis type not found')
  return res.json()
}

// ===== Run Endpoints =====

export async function listRuns(options?: {
  limit?: number
  status?: RunStatus
  analysis_type?: AnalysisType
}): Promise<Run[]> {
  const params = new URLSearchParams()
  if (options?.limit) params.set('limit', String(options.limit))
  if (options?.status) params.set('status', options.status)
  if (options?.analysis_type) params.set('analysis_type', options.analysis_type)
  
  const url = `${getApiBaseUrl()}/api/runs${params.toString() ? `?${params}` : ''}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to load runs')
  return res.json()
}

export async function getRun(runId: number): Promise<RunWithResult> {
  const res = await fetch(`${getApiBaseUrl()}/api/runs/${runId}`)
  if (!res.ok) throw new Error('Failed to load run')
  return res.json()
}

// ===== Prediction Endpoints =====

export async function predictJson(payload: PredictJsonRequest): Promise<{
  run_id: number
  result: Record<string, unknown>
}> {
  const res = await fetch(`${getApiBaseUrl()}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Prediction failed')
  return res.json()
}

export async function predictUpload(args: PredictUploadRequest): Promise<{
  run_id: number
  result: Record<string, unknown>
}> {
  const form = new FormData()
  form.set('kind', args.kind ?? 'upload')
  form.set('analysis_type', args.analysis_type ?? 'deforestation')
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

// ===== Organization Endpoints =====

export async function createOrganization(
  data: CreateOrganizationRequest
): Promise<OrganizationWithKey> {
  const res = await fetch(`${getApiBaseUrl()}/api/organizations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to create organization')
  return res.json()
}

export async function listOrganizations(options?: {
  type?: string
  limit?: number
}): Promise<Organization[]> {
  const params = new URLSearchParams()
  if (options?.type) params.set('type', options.type)
  if (options?.limit) params.set('limit', String(options.limit))
  
  const url = `${getApiBaseUrl()}/api/organizations${params.toString() ? `?${params}` : ''}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to load organizations')
  return res.json()
}

export async function getOrganization(orgId: number): Promise<Organization> {
  const res = await fetch(`${getApiBaseUrl()}/api/organizations/${orgId}`)
  if (!res.ok) throw new Error('Organization not found')
  return res.json()
}

// ===== Subscription Endpoints =====

export async function createSubscription(
  orgId: number,
  data: CreateSubscriptionRequest
): Promise<Subscription> {
  const res = await fetch(`${getApiBaseUrl()}/api/organizations/${orgId}/subscriptions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to create subscription')
  return res.json()
}

export async function listSubscriptions(orgId: number): Promise<Subscription[]> {
  const res = await fetch(`${getApiBaseUrl()}/api/organizations/${orgId}/subscriptions`)
  if (!res.ok) throw new Error('Failed to load subscriptions')
  return res.json()
}

// ===== Alert Endpoints =====

export async function listAlerts(
  orgId: number,
  options?: {
    undelivered_only?: boolean
    unacknowledged_only?: boolean
    limit?: number
  }
): Promise<Alert[]> {
  const params = new URLSearchParams()
  if (options?.undelivered_only) params.set('undelivered_only', 'true')
  if (options?.unacknowledged_only) params.set('unacknowledged_only', 'true')
  if (options?.limit) params.set('limit', String(options.limit))
  
  const url = `${getApiBaseUrl()}/api/organizations/${orgId}/alerts${params.toString() ? `?${params}` : ''}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to load alerts')
  return res.json()
}

export async function acknowledgeAlert(
  alertId: number,
  acknowledgedBy?: string
): Promise<{ success: boolean; alert_id: number }> {
  const res = await fetch(`${getApiBaseUrl()}/api/alerts/${alertId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ acknowledged_by: acknowledgedBy }),
  })
  if (!res.ok) throw new Error('Failed to acknowledge alert')
  return res.json()
}
