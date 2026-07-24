import { ReactNode } from 'react'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

export type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'neutral'
export type BadgeSize = 'sm' | 'md' | 'lg'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  size?: BadgeSize
  dot?: boolean
  className?: string
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-base-800 text-base-200',
  success: 'bg-brand-600/15 text-brand-600 border-brand-600/30',
  warning: 'bg-amber-500/15 text-amber-600 border-amber-500/30',
  danger: 'bg-danger-500/15 text-danger-500 border-danger-500/30',
  info: 'bg-ocean-500/15 text-ocean-600 border-ocean-500/30',
  neutral: 'bg-base-700/50 text-base-300 border-base-600/30',
}

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'text-xs px-1.5 py-0.5',
  md: 'text-xs px-2 py-1',
  lg: 'text-sm px-2.5 py-1',
}

const dotColors: Record<BadgeVariant, string> = {
  default: 'bg-base-400',
  success: 'bg-brand-400',
  warning: 'bg-amber-400',
  danger: 'bg-danger-400',
  info: 'bg-ocean-400',
  neutral: 'bg-base-400',
}

export function Badge({
  children,
  variant = 'default',
  size = 'md',
  dot = false,
  className,
}: BadgeProps) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1.5 rounded-full font-medium border',
        variantStyles[variant],
        sizeStyles[size],
        className,
      )}
    >
      {dot && (
        <span className={cx('w-1.5 h-1.5 rounded-full', dotColors[variant])} />
      )}
      {children}
    </span>
  )
}

// Status badge specifically for run status
export type RunStatus = 'running' | 'completed' | 'failed' | 'pending'

interface StatusBadgeProps {
  status: RunStatus
  size?: BadgeSize
}

const statusConfig: Record<RunStatus, { variant: BadgeVariant; label: string }> = {
  running: { variant: 'info', label: 'Running' },
  completed: { variant: 'success', label: 'Completed' },
  failed: { variant: 'danger', label: 'Failed' },
  pending: { variant: 'neutral', label: 'Pending' },
}

export function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const config = statusConfig[status] || statusConfig.pending
  return (
    <Badge variant={config.variant} size={size} dot>
      {config.label}
    </Badge>
  )
}

// Severity badge for alerts
export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical'

interface SeverityBadgeProps {
  severity: AlertSeverity
  size?: BadgeSize
}

const severityConfig: Record<AlertSeverity, { variant: BadgeVariant; label: string }> = {
  low: { variant: 'neutral', label: 'Low' },
  medium: { variant: 'warning', label: 'Medium' },
  high: { variant: 'danger', label: 'High' },
  critical: { variant: 'danger', label: 'Critical' },
}

export function SeverityBadge({ severity, size = 'sm' }: SeverityBadgeProps) {
  const config = severityConfig[severity] || severityConfig.low
  return (
    <Badge variant={config.variant} size={size} dot>
      {config.label}
    </Badge>
  )
}

// Analysis type badge
export type AnalysisType = 'deforestation' | 'ice_melting' | 'flooding' | 'drought' | 'wildfire'

interface AnalysisTypeBadgeProps {
  type: AnalysisType
  size?: BadgeSize
}

const analysisTypeConfig: Record<AnalysisType, { variant: BadgeVariant; label: string }> = {
  deforestation: { variant: 'success', label: 'Deforestation' },
  ice_melting: { variant: 'info', label: 'Ice Melting' },
  flooding: { variant: 'info', label: 'Flooding' },
  drought: { variant: 'warning', label: 'Drought' },
  wildfire: { variant: 'danger', label: 'Wildfire' },
}

export function AnalysisTypeBadge({ type, size = 'sm' }: AnalysisTypeBadgeProps) {
  const config = analysisTypeConfig[type] || { variant: 'default' as BadgeVariant, label: type }
  return (
    <Badge variant={config.variant} size={size}>
      {config.label}
    </Badge>
  )
}
