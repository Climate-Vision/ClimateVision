import { Card } from '../ui/Card'
import { Badge, StatusBadge, SeverityBadge, AnalysisTypeBadge } from '../ui/Badge'
import type { RunStatus, AlertSeverity, AnalysisType } from '../ui/Badge'
import { GaugeChart, getGaugeVariant } from '../charts/GaugeChart'
import { ComparisonBar } from '../charts/BarChart'
import { InfoTooltip } from '../ui/Tooltip'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

// Organization interface
export interface Organization {
  id: number
  name: string
  type: string
  logo_url?: string
  contact_email?: string
}

// Alert data for NGO context
export interface NGOAlert {
  id: number
  alert_type: string
  severity: AlertSeverity
  title: string
  message: string
  created_at: string
  acknowledged: boolean
}

// Region subscription
export interface Subscription {
  id: number
  name?: string
  bbox: number[]
  analysis_types: AnalysisType[]
  alert_threshold: number
  active: boolean
}

// Analysis result with comparison data
export interface NGOAnalysisResult {
  current: {
    forest_percentage?: number
    ice_percentage?: number
    flooded_percentage?: number
    mean_confidence?: number
  }
  previous?: {
    forest_percentage?: number
    ice_percentage?: number
    flooded_percentage?: number
  }
  change_detected: boolean
  change_percentage?: number
  region: {
    bbox?: number[]
    date_range?: string
  }
}

export interface NGOResultCardProps {
  organization: Organization
  result: NGOAnalysisResult
  subscription?: Subscription
  alert?: NGOAlert
  runId?: number
  status?: RunStatus
  analysisType?: AnalysisType
  createdAt?: string
  onAcknowledge?: () => void
  onInvestigate?: () => void
  onExport?: () => void
  className?: string
}

// Format bbox for display
function formatBBox(bbox?: number[]): string {
  if (!bbox || bbox.length !== 4) return 'N/A'
  return `${bbox[0].toFixed(2)}°, ${bbox[1].toFixed(2)}° to ${bbox[2].toFixed(2)}°, ${bbox[3].toFixed(2)}°`
}

export function NGOResultCard({
  organization,
  result,
  subscription,
  alert,
  runId,
  status = 'completed',
  analysisType = 'deforestation',
  createdAt,
  onAcknowledge,
  onInvestigate,
  onExport,
  className,
}: NGOResultCardProps) {
  // Get the main percentage based on analysis type
  const getCurrentPercentage = (): number => {
    const { current } = result
    switch (analysisType) {
      case 'deforestation':
        return current.forest_percentage ?? 0
      case 'ice_melting':
        return current.ice_percentage ?? 0
      case 'flooding':
        return current.flooded_percentage ?? 0
      default:
        return current.forest_percentage ?? 0
    }
  }

  const getPreviousPercentage = (): number | undefined => {
    const { previous } = result
    if (!previous) return undefined
    switch (analysisType) {
      case 'deforestation':
        return previous.forest_percentage
      case 'ice_melting':
        return previous.ice_percentage
      case 'flooding':
        return previous.flooded_percentage
      default:
        return previous.forest_percentage
    }
  }

  const currentPercentage = getCurrentPercentage()
  const previousPercentage = getPreviousPercentage()
  const gaugeType = analysisType === 'ice_melting' ? 'ice' : analysisType === 'flooding' ? 'flood' : 'forest'
  const gaugeVariant = getGaugeVariant(currentPercentage, gaugeType)

  const coverageLabel = 
    analysisType === 'ice_melting' ? 'Ice Extent' : 
    analysisType === 'flooding' ? 'Flooded Area' : 
    'Forest Coverage'

  const cardVariant = alert ? (
    alert.severity === 'critical' || alert.severity === 'high' ? 'danger' :
    alert.severity === 'medium' ? 'warning' : 'default'
  ) : 'default'

  return (
    <Card variant={cardVariant} className={className}>
      {/* Organization Header */}
      <div className="flex items-start justify-between gap-4 pb-4 border-b border-base-800">
        <div className="flex items-center gap-3">
          {organization.logo_url ? (
            <img
              src={organization.logo_url}
              alt={organization.name}
              className="w-10 h-10 rounded-lg object-cover"
            />
          ) : (
            <div className="w-10 h-10 rounded-lg bg-brand-600/20 flex items-center justify-center">
              <span className="text-brand-400 font-bold text-sm">
                {organization.name.slice(0, 2).toUpperCase()}
              </span>
            </div>
          )}
          <div>
            <h3 className="text-base-100 font-semibold">{organization.name}</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <Badge variant="neutral" size="sm">{organization.type.toUpperCase()}</Badge>
              {subscription?.name && (
                <span className="text-xs text-base-400">{subscription.name}</span>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-1">
          {runId && <span className="text-xs text-base-400">Run #{runId}</span>}
          <StatusBadge status={status} />
        </div>
      </div>

      {/* Alert Banner */}
      {alert && (
        <div className={cx(
          'mt-4 p-3 rounded-lg border',
          alert.severity === 'critical' && 'bg-danger-500/10 border-danger-500/30',
          alert.severity === 'high' && 'bg-danger-500/10 border-danger-500/30',
          alert.severity === 'medium' && 'bg-amber-500/10 border-amber-500/30',
          alert.severity === 'low' && 'bg-base-800/50 border-base-700',
        )}>
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2">
              <svg className={cx(
                'w-5 h-5 flex-shrink-0 mt-0.5',
                alert.severity === 'critical' || alert.severity === 'high' ? 'text-danger-400' :
                alert.severity === 'medium' ? 'text-amber-400' : 'text-base-400'
              )} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-base-100">{alert.title}</span>
                  <SeverityBadge severity={alert.severity} size="sm" />
                </div>
                <p className="text-sm text-base-300 mt-1">{alert.message}</p>
              </div>
            </div>
            {!alert.acknowledged && onAcknowledge && (
              <button
                onClick={onAcknowledge}
                className="px-2 py-1 text-xs font-medium text-base-200 hover:text-base-100 bg-base-800/50 hover:bg-base-800 rounded transition"
              >
                Acknowledge
              </button>
            )}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="mt-4 flex gap-6">
        {/* Gauge */}
        <div className="flex-shrink-0">
          <GaugeChart
            value={currentPercentage}
            variant={gaugeVariant}
            size="lg"
            label={coverageLabel}
          />
        </div>

        {/* Stats and Change Detection */}
        <div className="flex-1 space-y-4">
          {/* Analysis Type */}
          <div className="flex items-center gap-2">
            <AnalysisTypeBadge type={analysisType} />
            {result.change_detected && (
              <Badge variant="warning" size="sm" dot>
                Change Detected
              </Badge>
            )}
          </div>

          {/* Change Comparison */}
          {previousPercentage !== undefined && (
            <ComparisonBar
              label={coverageLabel}
              value1={previousPercentage}
              value2={currentPercentage}
              label1="Previous"
              label2="Current"
              showDiff
            />
          )}

          {/* Confidence */}
          {result.current.mean_confidence !== undefined && (
            <div>
              <div className="flex items-center gap-1 mb-1">
                <span className="text-xs text-base-400">Model Confidence</span>
                <InfoTooltip content="How confident the model is in this prediction" />
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-base-800 rounded-full overflow-hidden">
                  <div
                    className={cx(
                      'h-full rounded-full transition-all',
                      result.current.mean_confidence >= 0.8 ? 'bg-brand-500' :
                      result.current.mean_confidence >= 0.6 ? 'bg-ocean-500' :
                      result.current.mean_confidence >= 0.4 ? 'bg-amber-500' : 'bg-danger-500'
                    )}
                    style={{ width: `${result.current.mean_confidence * 100}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-base-200">
                  {(result.current.mean_confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Region Info */}
      <div className="mt-4 pt-4 border-t border-base-800">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-base-400">Region: </span>
            <span className="text-base-200">{formatBBox(result.region.bbox)}</span>
          </div>
          {result.region.date_range && (
            <div>
              <span className="text-base-400">Period: </span>
              <span className="text-base-200">{result.region.date_range}</span>
            </div>
          )}
          {subscription?.alert_threshold && (
            <div>
              <span className="text-base-400">Alert Threshold: </span>
              <span className="text-base-200">{subscription.alert_threshold}% change</span>
            </div>
          )}
          {createdAt && (
            <div>
              <span className="text-base-400">Analyzed: </span>
              <span className="text-base-200">
                {new Date(createdAt).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      {(onInvestigate || onExport) && (
        <div className="mt-4 pt-4 border-t border-base-800 flex items-center justify-end gap-2">
          {onInvestigate && (
            <button
              onClick={onInvestigate}
              className="px-3 py-1.5 text-sm font-medium text-base-200 hover:text-base-100 bg-base-800/60 hover:bg-base-800 border border-base-700 rounded-lg transition"
            >
              Investigate
            </button>
          )}
          {onExport && (
            <button
              onClick={onExport}
              className="px-3 py-1.5 text-sm font-medium text-base-100 bg-brand-600 hover:bg-brand-500 rounded-lg transition"
            >
              Export Report
            </button>
          )}
        </div>
      )}
    </Card>
  )
}
