// Shared TypeScript types for ClimateVision frontend

// Analysis types supported by the platform
export type AnalysisType = 'deforestation' | 'ice_melting' | 'flooding' | 'drought' | 'wildfire'

// Run status
export type RunStatus = 'pending' | 'running' | 'completed' | 'failed'

// Alert severity levels
export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical'

// Organization types
export type OrganizationType = 'ngo' | 'government' | 'research' | 'corporate'

// Notification channels
export type NotificationChannel = 'email' | 'webhook' | 'api' | 'sms'

// Region information from analysis
export interface RegionInfo {
  bbox?: number[]
  date_range?: string
  images_available?: number
}

// NDVI statistics
export interface NDVIStats {
  NDVI_min: number
  NDVI_mean: number
  NDVI_max: number
}

// Inference result structure
export interface InferenceResult {
  image_size?: [number, number]
  // Deforestation
  forest_pixels?: number
  non_forest_pixels?: number
  forest_percentage?: number
  // Ice melting
  ice_pixels?: number
  water_pixels?: number
  land_pixels?: number
  ice_percentage?: number
  ice_extent_km2?: number
  melt_rate?: number
  // Flooding
  flooded_pixels?: number
  dry_pixels?: number
  flooded_percentage?: number
  flooded_area_km2?: number
  // Common
  mean_confidence?: number
}

// Complete analysis result
export interface AnalysisResult {
  region?: RegionInfo
  ndvi_stats?: NDVIStats
  inference?: InferenceResult
  error?: string
  input?: {
    file?: string
  }
}

// Run record from API
export interface Run {
  id: number
  kind: string
  status: RunStatus
  bbox?: string
  start_date?: string
  end_date?: string
  analysis_type?: AnalysisType
  created_at: string
  updated_at: string
}

// Result record from API
export interface Result {
  id: number
  run_id: number
  payload: AnalysisResult
  mask_path?: string
  created_at: string
}

// Run with result
export interface RunWithResult {
  run: Run
  result: Result | null
}

// Organization
export interface Organization {
  id: number
  name: string
  type: OrganizationType
  logo_url?: string
  contact_email?: string
  regions_of_interest?: string[]
  alert_preferences?: AlertPreferences
  api_key?: string
  created_at: string
}

// Alert preferences configuration
export interface AlertPreferences {
  enabled: boolean
  channels: NotificationChannel[]
  min_severity: AlertSeverity
  analysis_types: AnalysisType[]
  quiet_hours?: {
    start: string
    end: string
  }
}

// Organization subscription to a region
export interface Subscription {
  id: number
  organization_id: number
  bbox: number[]
  analysis_types: AnalysisType[]
  alert_threshold: number
  notification_channel: NotificationChannel
  active: boolean
  created_at: string
}

// Alert record
export interface Alert {
  id: number
  organization_id: number
  run_id: number
  alert_type: string
  severity: AlertSeverity
  message: string
  delivered: boolean
  delivered_at?: string
  created_at: string
}

// Prediction request
export interface PredictRequest {
  kind?: string
  analysis_type?: AnalysisType
  bbox?: number[]
  start_date?: string
  end_date?: string
}

// Prediction response
export interface PredictResponse {
  run_id: number
  result: AnalysisResult
}

// Analysis type configuration
export interface AnalysisTypeConfig {
  name: AnalysisType
  display_name: string
  description: string
  icon: string
  color: string
  enabled: boolean
  bands: string[]
  classes: string[]
  thresholds: Record<string, number>
}

// API health response
export interface HealthResponse {
  status: string
  version?: string
  analysis_types?: AnalysisType[]
}
