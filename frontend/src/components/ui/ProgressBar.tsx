function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

export type ProgressVariant = 'default' | 'success' | 'warning' | 'danger' | 'info'
export type ProgressSize = 'sm' | 'md' | 'lg'

interface ProgressBarProps {
  value: number // 0-100
  max?: number
  variant?: ProgressVariant
  size?: ProgressSize
  showLabel?: boolean
  label?: string
  className?: string
  animated?: boolean
}

const variantStyles: Record<ProgressVariant, string> = {
  default: 'bg-base-500',
  success: 'bg-brand-500',
  warning: 'bg-amber-500',
  danger: 'bg-danger-500',
  info: 'bg-ocean-500',
}

const sizeStyles: Record<ProgressSize, string> = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-3',
}

export function ProgressBar({
  value,
  max = 100,
  variant = 'default',
  size = 'md',
  showLabel = false,
  label,
  className,
  animated = false,
}: ProgressBarProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  
  return (
    <div className={cx('w-full', className)}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-1">
          {label && <span className="text-xs text-base-200/70">{label}</span>}
          {showLabel && (
            <span className="text-xs font-medium text-base-200">
              {percentage.toFixed(1)}%
            </span>
          )}
        </div>
      )}
      <div className={cx('w-full bg-base-800 rounded-full overflow-hidden', sizeStyles[size])}>
        <div
          className={cx(
            'h-full rounded-full transition-all duration-500',
            variantStyles[variant],
            animated && 'animate-pulse',
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

// Confidence bar with color gradient based on value
interface ConfidenceBarProps {
  value: number // 0-1
  size?: ProgressSize
  showLabel?: boolean
  className?: string
}

export function ConfidenceBar({
  value,
  size = 'md',
  showLabel = true,
  className,
}: ConfidenceBarProps) {
  const percentage = value * 100
  
  // Determine variant based on confidence level
  let variant: ProgressVariant = 'danger'
  if (percentage >= 80) variant = 'success'
  else if (percentage >= 60) variant = 'info'
  else if (percentage >= 40) variant = 'warning'
  
  return (
    <ProgressBar
      value={percentage}
      variant={variant}
      size={size}
      showLabel={showLabel}
      label="Confidence"
      className={className}
    />
  )
}

// Forest coverage bar with specific styling
interface CoverageBarProps {
  value: number // 0-100
  type?: 'forest' | 'ice' | 'water' | 'flood'
  size?: ProgressSize
  showLabel?: boolean
  className?: string
}

const coverageVariants: Record<string, ProgressVariant> = {
  forest: 'success',
  ice: 'info',
  water: 'info',
  flood: 'warning',
}

const coverageLabels: Record<string, string> = {
  forest: 'Forest Coverage',
  ice: 'Ice Extent',
  water: 'Water Coverage',
  flood: 'Flooded Area',
}

export function CoverageBar({
  value,
  type = 'forest',
  size = 'md',
  showLabel = true,
  className,
}: CoverageBarProps) {
  return (
    <ProgressBar
      value={value}
      variant={coverageVariants[type] || 'default'}
      size={size}
      showLabel={showLabel}
      label={coverageLabels[type] || 'Coverage'}
      className={className}
    />
  )
}
