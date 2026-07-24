function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

export type GaugeVariant = 'forest' | 'ice' | 'water' | 'danger' | 'neutral'
export type GaugeSize = 'sm' | 'md' | 'lg' | 'xl'

interface GaugeChartProps {
  value: number // 0-100
  label?: string
  sublabel?: string
  variant?: GaugeVariant
  size?: GaugeSize
  showValue?: boolean
  thickness?: number
  className?: string
}

const variantColors: Record<GaugeVariant, { stroke: string; bg: string; text: string }> = {
  forest: {
    stroke: 'stroke-brand-500',
    bg: 'stroke-brand-500/20',
    text: 'text-brand-600',
  },
  ice: {
    stroke: 'stroke-ocean-500',
    bg: 'stroke-ocean-500/20',
    text: 'text-ocean-600',
  },
  water: {
    stroke: 'stroke-blue-500',
    bg: 'stroke-blue-500/20',
    text: 'text-blue-600',
  },
  danger: {
    stroke: 'stroke-danger-500',
    bg: 'stroke-danger-500/20',
    text: 'text-danger-500',
  },
  neutral: {
    stroke: 'stroke-base-400',
    bg: 'stroke-base-700',
    text: 'text-base-300',
  },
}

const sizeConfig: Record<GaugeSize, { size: number; strokeWidth: number; fontSize: string; subFontSize: string }> = {
  sm: { size: 80, strokeWidth: 6, fontSize: 'text-lg', subFontSize: 'text-xs' },
  md: { size: 120, strokeWidth: 8, fontSize: 'text-2xl', subFontSize: 'text-sm' },
  lg: { size: 160, strokeWidth: 10, fontSize: 'text-3xl', subFontSize: 'text-sm' },
  xl: { size: 200, strokeWidth: 12, fontSize: 'text-4xl', subFontSize: 'text-base' },
}

export function GaugeChart({
  value,
  label,
  sublabel,
  variant = 'neutral',
  size = 'md',
  showValue = true,
  thickness,
  className,
}: GaugeChartProps) {
  const config = sizeConfig[size]
  const colors = variantColors[variant]
  
  const strokeWidth = thickness || config.strokeWidth
  const radius = (config.size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const percentage = Math.min(100, Math.max(0, value))
  const offset = circumference - (percentage / 100) * circumference
  
  return (
    <div className={cx('flex flex-col items-center', className)}>
      <div className="relative" style={{ width: config.size, height: config.size }}>
        <svg
          width={config.size}
          height={config.size}
          className="transform -rotate-90"
        >
          {/* Background circle */}
          <circle
            cx={config.size / 2}
            cy={config.size / 2}
            r={radius}
            fill="none"
            strokeWidth={strokeWidth}
            className={colors.bg}
          />
          {/* Progress circle */}
          <circle
            cx={config.size / 2}
            cy={config.size / 2}
            r={radius}
            fill="none"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            className={cx(colors.stroke, 'transition-all duration-700 ease-out')}
            style={{
              strokeDasharray: circumference,
              strokeDashoffset: offset,
            }}
          />
        </svg>
        
        {/* Center content */}
        {showValue && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={cx('font-bold', config.fontSize, colors.text)}>
              {percentage.toFixed(1)}%
            </span>
            {sublabel && (
              <span className={cx('text-base-400', config.subFontSize)}>
                {sublabel}
              </span>
            )}
          </div>
        )}
      </div>
      
      {label && (
        <span className="mt-2 text-sm font-medium text-base-200">{label}</span>
      )}
    </div>
  )
}

// Mini gauge for inline display
interface MiniGaugeProps {
  value: number
  variant?: GaugeVariant
  className?: string
}

export function MiniGauge({ value, variant = 'neutral', className }: MiniGaugeProps) {
  const colors = variantColors[variant]
  const percentage = Math.min(100, Math.max(0, value))
  const radius = 14
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percentage / 100) * circumference
  
  return (
    <div className={cx('inline-flex items-center gap-2', className)}>
      <svg width={36} height={36} className="transform -rotate-90">
        <circle
          cx={18}
          cy={18}
          r={radius}
          fill="none"
          strokeWidth={4}
          className={colors.bg}
        />
        <circle
          cx={18}
          cy={18}
          r={radius}
          fill="none"
          strokeWidth={4}
          strokeLinecap="round"
          className={colors.stroke}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset,
          }}
        />
      </svg>
      <span className={cx('text-sm font-medium', colors.text)}>
        {percentage.toFixed(1)}%
      </span>
    </div>
  )
}

// Determine variant based on value thresholds for different analysis types
export function getGaugeVariant(
  value: number,
  type: 'forest' | 'ice' | 'flood' = 'forest'
): GaugeVariant {
  if (type === 'forest') {
    if (value >= 70) return 'forest'
    if (value >= 40) return 'neutral'
    return 'danger'
  }
  
  if (type === 'ice') {
    if (value >= 70) return 'ice'
    if (value >= 40) return 'neutral'
    return 'danger'
  }
  
  if (type === 'flood') {
    if (value >= 50) return 'danger'
    if (value >= 20) return 'water'
    return 'neutral'
  }
  
  return 'neutral'
}
