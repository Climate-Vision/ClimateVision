import { useState } from 'react'
import { Card } from '../ui/Card'
import { Badge, SeverityBadge } from '../ui/Badge'
import type { AlertSeverity } from '../ui/Badge'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
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
  subscription_id?: number
  run_id?: number
}

export interface AlertsPanelProps {
  alerts: Alert[]
  onAcknowledge?: (alertId: number) => void
  onViewRun?: (runId: number) => void
  onDismiss?: (alertId: number) => void
  loading?: boolean
  className?: string
}

type FilterType = 'all' | 'unacknowledged' | AlertSeverity

export function AlertsPanel({
  alerts,
  onAcknowledge,
  onViewRun,
  onDismiss,
  loading = false,
  className,
}: AlertsPanelProps) {
  const [filter, setFilter] = useState<FilterType>('all')
  const [expandedId, setExpandedId] = useState<number | null>(null)

  // Filter alerts
  const filteredAlerts = alerts.filter((alert) => {
    if (filter === 'all') return true
    if (filter === 'unacknowledged') return !alert.acknowledged
    return alert.severity === filter
  })

  // Count by severity
  const counts = {
    all: alerts.length,
    unacknowledged: alerts.filter((a) => !a.acknowledged).length,
    critical: alerts.filter((a) => a.severity === 'critical').length,
    high: alerts.filter((a) => a.severity === 'high').length,
    medium: alerts.filter((a) => a.severity === 'medium').length,
    low: alerts.filter((a) => a.severity === 'low').length,
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const getSeverityIcon = (severity: AlertSeverity) => {
    switch (severity) {
      case 'critical':
        return (
          <svg className="w-5 h-5 text-danger-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        )
      case 'high':
        return (
          <svg className="w-5 h-5 text-danger-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'medium':
        return (
          <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      default:
        return (
          <svg className="w-5 h-5 text-base-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
    }
  }

  return (
    <Card title="Alerts" className={className}>
      {/* Filter tabs */}
      <div className="flex flex-wrap gap-2 pb-4 border-b border-base-800 mb-4">
        <button
          onClick={() => setFilter('all')}
          className={cx(
            'px-3 py-1.5 text-xs font-medium rounded-lg transition',
            filter === 'all'
              ? 'bg-base-700 text-base-100'
              : 'text-base-400 hover:text-base-200 hover:bg-base-800',
          )}
        >
          All ({counts.all})
        </button>
        <button
          onClick={() => setFilter('unacknowledged')}
          className={cx(
            'px-3 py-1.5 text-xs font-medium rounded-lg transition',
            filter === 'unacknowledged'
              ? 'bg-ocean-600/20 text-ocean-400 border border-ocean-600/30'
              : 'text-base-400 hover:text-base-200 hover:bg-base-800',
          )}
        >
          Unacknowledged ({counts.unacknowledged})
        </button>
        {counts.critical > 0 && (
          <button
            onClick={() => setFilter('critical')}
            className={cx(
              'px-3 py-1.5 text-xs font-medium rounded-lg transition',
              filter === 'critical'
                ? 'bg-danger-600/20 text-danger-400 border border-danger-600/30'
                : 'text-base-400 hover:text-base-200 hover:bg-base-800',
            )}
          >
            Critical ({counts.critical})
          </button>
        )}
        {counts.high > 0 && (
          <button
            onClick={() => setFilter('high')}
            className={cx(
              'px-3 py-1.5 text-xs font-medium rounded-lg transition',
              filter === 'high'
                ? 'bg-danger-600/20 text-danger-400 border border-danger-600/30'
                : 'text-base-400 hover:text-base-200 hover:bg-base-800',
            )}
          >
            High ({counts.high})
          </button>
        )}
      </div>

      {/* Loading state */}
      {loading && (
        <div className="text-center py-8 text-base-400">
          <div className="animate-spin w-6 h-6 border-2 border-base-600 border-t-brand-500 rounded-full mx-auto" />
          <p className="mt-2 text-sm">Loading alerts...</p>
        </div>
      )}

      {/* Empty state */}
      {!loading && filteredAlerts.length === 0 && (
        <div className="text-center py-8 text-base-400">
          <svg className="w-12 h-12 mx-auto mb-3 text-base-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm">
            {filter === 'all' ? 'No alerts yet' : `No ${filter} alerts`}
          </p>
        </div>
      )}

      {/* Alerts list */}
      {!loading && filteredAlerts.length > 0 && (
        <div className="space-y-3 max-h-[500px] overflow-y-auto">
          {filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={cx(
                'rounded-lg border transition',
                alert.acknowledged
                  ? 'bg-base-900/30 border-base-800'
                  : alert.severity === 'critical' || alert.severity === 'high'
                  ? 'bg-danger-500/5 border-danger-500/20'
                  : alert.severity === 'medium'
                  ? 'bg-amber-500/5 border-amber-500/20'
                  : 'bg-base-900/50 border-base-700',
              )}
            >
              <button
                onClick={() => setExpandedId(expandedId === alert.id ? null : alert.id)}
                className="w-full text-left p-3"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getSeverityIcon(alert.severity)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={cx(
                        'text-sm font-medium',
                        alert.acknowledged ? 'text-base-300' : 'text-base-100',
                      )}>
                        {alert.title}
                      </span>
                      <SeverityBadge severity={alert.severity} size="sm" />
                      {alert.acknowledged && (
                        <Badge variant="neutral" size="sm">Acknowledged</Badge>
                      )}
                    </div>
                    <p className="text-sm text-base-400 mt-0.5 line-clamp-1">
                      {alert.message}
                    </p>
                  </div>
                  <div className="flex-shrink-0 text-right">
                    <span className="text-xs text-base-500">{formatDate(alert.created_at)}</span>
                    <svg
                      className={cx(
                        'w-4 h-4 text-base-500 mx-auto mt-1 transition-transform',
                        expandedId === alert.id && 'rotate-180',
                      )}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </button>

              {/* Expanded content */}
              {expandedId === alert.id && (
                <div className="px-3 pb-3 border-t border-base-800 pt-3 mt-1">
                  <p className="text-sm text-base-300">{alert.message}</p>
                  
                  <div className="flex items-center justify-between mt-3">
                    <div className="flex items-center gap-4 text-xs text-base-500">
                      <span>Type: {alert.alert_type}</span>
                      {alert.run_id && <span>Run: #{alert.run_id}</span>}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {alert.run_id && onViewRun && (
                        <button
                          onClick={() => onViewRun(alert.run_id!)}
                          className="px-2 py-1 text-xs font-medium text-base-300 hover:text-base-100 bg-base-800/50 hover:bg-base-800 rounded transition"
                        >
                          View Run
                        </button>
                      )}
                      {!alert.acknowledged && onAcknowledge && (
                        <button
                          onClick={() => onAcknowledge(alert.id)}
                          className="px-2 py-1 text-xs font-medium text-brand-400 hover:text-brand-300 bg-brand-600/10 hover:bg-brand-600/20 rounded transition"
                        >
                          Acknowledge
                        </button>
                      )}
                      {onDismiss && (
                        <button
                          onClick={() => onDismiss(alert.id)}
                          className="px-2 py-1 text-xs font-medium text-base-400 hover:text-base-200 rounded transition"
                        >
                          Dismiss
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

// Summary component for dashboard
export interface AlertsSummaryProps {
  alerts: Alert[]
  className?: string
}

export function AlertsSummary({ alerts, className }: AlertsSummaryProps) {
  const unacknowledged = alerts.filter((a) => !a.acknowledged)
  const critical = unacknowledged.filter((a) => a.severity === 'critical' || a.severity === 'high')

  if (unacknowledged.length === 0) {
    return (
      <div className={cx('flex items-center gap-2 text-brand-400', className)}>
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-sm">All clear</span>
      </div>
    )
  }

  return (
    <div className={cx('flex items-center gap-3', className)}>
      {critical.length > 0 && (
        <div className="flex items-center gap-1.5 text-danger-400">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-sm font-medium">{critical.length} critical</span>
        </div>
      )}
      <div className="flex items-center gap-1.5 text-amber-400">
        <span className="text-sm">{unacknowledged.length} unacknowledged</span>
      </div>
    </div>
  )
}
