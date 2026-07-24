import { Card } from './ui/Card'
import { Badge, StatusBadge, AnalysisTypeBadge } from './ui/Badge'
import type { RunStatus, AnalysisType } from './ui/Badge'
import { ConfidenceBar, CoverageBar } from './ui/ProgressBar'
import { GaugeChart, getGaugeVariant, MiniGauge } from './charts/GaugeChart'
import { StackedBar } from './charts/BarChart'
import { InfoTooltip } from './ui/Tooltip'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

// Types for inference results
export interface RegionInfo {
  bbox?: number[]
  date_range?: string
  images_available?: number
}

export interface NDVIStats {
  NDVI_min: number
  NDVI_mean: number
  NDVI_max: number
}

export interface InferenceResult {
  image_size?: [number, number]
  forest_pixels?: number
  non_forest_pixels?: number
  forest_percentage?: number
  mean_confidence?: number
  // For ice melting
  ice_pixels?: number
  water_pixels?: number
  ice_percentage?: number
  // For flooding
  flooded_pixels?: number
  dry_pixels?: number
  flooded_percentage?: number
}

export interface AnalysisResult {
  region?: RegionInfo
  ndvi_stats?: NDVIStats
  inference?: InferenceResult
  error?: string
}

export interface ResultCardProps {
  result: AnalysisResult
  runId?: number
  status?: RunStatus
  analysisType?: AnalysisType
  createdAt?: string
  showDetails?: boolean
  onClick?: () => void
  className?: string
}

// Format bbox for display
function formatBBox(bbox?: number[]): string {
  if (!bbox || bbox.length !== 4) return 'N/A'
  return `[${bbox.map((v) => v.toFixed(2)).join(', ')}]`
}

// Format date for display
function formatDate(dateStr?: string): string {
  if (!dateStr) return 'N/A'
  try {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return dateStr
  }
}

// Get NDVI color based on value
function getNDVIColor(value: number): string {
  if (value >= 0.6) return 'text-brand-600'
  if (value >= 0.3) return 'text-amber-600'
  if (value >= 0) return 'text-orange-600'
  return 'text-danger-500'
}

// NDVI indicator component
function NDVIIndicator({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center">
      <div className="text-xs text-base-400 mb-1">{label}</div>
      <div className={cx('text-lg font-semibold', getNDVIColor(value))}>
        {value.toFixed(3)}
      </div>
    </div>
  )
}

// Stat item component
function StatItem({
  label,
  value,
  subvalue,
  tooltip,
}: {
  label: string
  value: string | number
  subvalue?: string
  tooltip?: string
}) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-1">
        <span className="text-xs text-base-400">{label}</span>
        {tooltip && <InfoTooltip content={tooltip} />}
      </div>
      <span className="text-sm font-medium text-base-100">{value}</span>
      {subvalue && <span className="text-xs text-base-400">{subvalue}</span>}
    </div>
  )
}

// Main ResultCard component
export function ResultCard({
  result,
  runId,
  status,
  analysisType = 'deforestation',
  createdAt,
  showDetails = true,
  onClick,
  className,
}: ResultCardProps) {
  const { region, ndvi_stats, inference, error } = result
  
  // Determine main metric based on analysis type
  const getMainPercentage = (): number => {
    if (!inference) return 0
    switch (analysisType) {
      case 'deforestation':
        return inference.forest_percentage ?? 0
      case 'ice_melting':
        return inference.ice_percentage ?? 0
      case 'flooding':
        return inference.flooded_percentage ?? 0
      default:
        return inference.forest_percentage ?? 0
    }
  }
  
  const mainPercentage = getMainPercentage()
  const gaugeType = analysisType === 'ice_melting' ? 'ice' : analysisType === 'flooding' ? 'flood' : 'forest'
  const gaugeVariant = getGaugeVariant(mainPercentage, gaugeType)
  
  // Get pixel data for stacked bar
  const getPixelData = () => {
    if (!inference) return []
    
    if (analysisType === 'deforestation') {
      return [
        { label: 'Forest', value: inference.forest_pixels ?? 0, color: 'bg-brand-500' },
        { label: 'Non-Forest', value: inference.non_forest_pixels ?? 0, color: 'bg-amber-600' },
      ]
    }
    
    if (analysisType === 'ice_melting') {
      return [
        { label: 'Ice', value: inference.ice_pixels ?? 0, color: 'bg-ocean-500' },
        { label: 'Water', value: inference.water_pixels ?? 0, color: 'bg-blue-600' },
      ]
    }
    
    if (analysisType === 'flooding') {
      return [
        { label: 'Flooded', value: inference.flooded_pixels ?? 0, color: 'bg-blue-500' },
        { label: 'Dry', value: inference.dry_pixels ?? 0, color: 'bg-amber-600' },
      ]
    }
    
    return []
  }
  
  const coverageLabel = analysisType === 'ice_melting' ? 'Ice Extent' : analysisType === 'flooding' ? 'Flooded Area' : 'Forest Coverage'
  
  if (error) {
    return (
      <Card
        variant="danger"
        className={className}
        onClick={onClick}
        hoverable={!!onClick}
      >
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-10 h-10 rounded-full bg-danger-500/20 flex items-center justify-center">
            <svg className="w-5 h-5 text-danger-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              {runId && <span className="text-sm font-medium text-base-100">Run #{runId}</span>}
              <StatusBadge status="failed" />
            </div>
            <p className="mt-1 text-sm text-danger-600 break-words">{error}</p>
          </div>
        </div>
      </Card>
    )
  }
  
  return (
    <Card
      className={className}
      onClick={onClick}
      hoverable={!!onClick}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          {runId && (
            <span className="text-sm font-semibold text-base-100">Run #{runId}</span>
          )}
          {status && <StatusBadge status={status} />}
          <AnalysisTypeBadge type={analysisType} />
        </div>
        {createdAt && (
          <span className="text-xs text-base-400">{formatDate(createdAt)}</span>
        )}
      </div>
      
      {/* Main content - Gauge and Stats */}
      <div className="flex gap-6">
        {/* Gauge */}
        <div className="flex-shrink-0">
          <GaugeChart
            value={mainPercentage}
            variant={gaugeVariant}
            size="md"
            label={coverageLabel}
          />
        </div>
        
        {/* Stats grid */}
        <div className="flex-1 grid grid-cols-2 gap-3">
          {inference?.mean_confidence !== undefined && (
            <div className="col-span-2">
              <ConfidenceBar value={inference.mean_confidence} size="sm" />
            </div>
          )}
          
          {inference?.image_size && (
            <StatItem
              label="Image Size"
              value={`${inference.image_size[0]} x ${inference.image_size[1]}`}
              tooltip="Dimensions of the analyzed image in pixels"
            />
          )}
          
          {region?.images_available !== undefined && (
            <StatItem
              label="Images"
              value={region.images_available}
              tooltip="Number of satellite images available for this region"
            />
          )}
        </div>
      </div>
      
      {/* Pixel distribution */}
      {showDetails && inference && (
        <div className="mt-4">
          <StackedBar
            data={getPixelData()}
            title="Pixel Distribution"
          />
        </div>
      )}
      
      {/* NDVI Stats */}
      {showDetails && ndvi_stats && (
        <div className="mt-4 pt-4 border-t border-base-800">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-medium text-base-200">NDVI Statistics</span>
            <InfoTooltip content="Normalized Difference Vegetation Index - measures vegetation health (-1 to 1)" />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <NDVIIndicator label="Min" value={ndvi_stats.NDVI_min} />
            <NDVIIndicator label="Mean" value={ndvi_stats.NDVI_mean} />
            <NDVIIndicator label="Max" value={ndvi_stats.NDVI_max} />
          </div>
        </div>
      )}
      
      {/* Region Info */}
      {showDetails && region && (
        <div className="mt-4 pt-4 border-t border-base-800">
          <div className="text-sm font-medium text-base-200 mb-2">Region</div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-base-400">BBox: </span>
              <span className="text-base-200 font-mono text-xs">{formatBBox(region.bbox)}</span>
            </div>
            {region.date_range && (
              <div>
                <span className="text-base-400">Date Range: </span>
                <span className="text-base-200">{region.date_range}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}

// Compact version for grid displays
export function CompactResultCard({
  result,
  runId,
  status,
  analysisType = 'deforestation',
  createdAt,
  onClick,
  selected,
  className,
}: ResultCardProps & { selected?: boolean }) {
  const { inference, error } = result
  
  const getMainPercentage = (): number => {
    if (!inference) return 0
    switch (analysisType) {
      case 'deforestation':
        return inference.forest_percentage ?? 0
      case 'ice_melting':
        return inference.ice_percentage ?? 0
      case 'flooding':
        return inference.flooded_percentage ?? 0
      default:
        return inference.forest_percentage ?? 0
    }
  }
  
  const mainPercentage = getMainPercentage()
  const gaugeType = analysisType === 'ice_melting' ? 'ice' : analysisType === 'flooding' ? 'flood' : 'forest'
  const gaugeVariant = getGaugeVariant(mainPercentage, gaugeType)
  
  return (
    <button
      onClick={onClick}
      className={cx(
        'text-left w-full rounded-xl border px-4 py-3 transition',
        'hover:bg-base-950/50',
        selected ? 'border-brand-500 bg-brand-900/20' : 'border-base-800 bg-base-950/30',
        error && 'border-danger-500/40 bg-danger-900/10',
        className,
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {runId && (
            <span className="text-sm font-semibold text-base-100">#{runId}</span>
          )}
          {status && <StatusBadge status={status} size="sm" />}
        </div>
        {!error && <MiniGauge value={mainPercentage} variant={gaugeVariant} />}
      </div>
      
      <div className="mt-2 flex items-center gap-2">
        <AnalysisTypeBadge type={analysisType} size="sm" />
        {createdAt && (
          <span className="text-xs text-base-400">{formatDate(createdAt)}</span>
        )}
      </div>
      
      {error && (
        <p className="mt-2 text-xs text-danger-600 truncate">{error}</p>
      )}
    </button>
  )
}
