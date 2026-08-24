function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

interface BarData {
  label: string
  value: number
  color?: string
}

interface BarChartProps {
  data: BarData[]
  title?: string
  maxValue?: number
  showValues?: boolean
  horizontal?: boolean
  height?: number
  className?: string
}

const defaultColors = [
  'bg-brand-500',
  'bg-ocean-500',
  'bg-amber-500',
  'bg-danger-500',
  'bg-purple-500',
  'bg-pink-500',
]

export function BarChart({
  data,
  title,
  maxValue,
  showValues = true,
  horizontal = false,
  height = 200,
  className,
}: BarChartProps) {
  const max = maxValue || Math.max(...data.map((d) => d.value), 1)
  
  if (horizontal) {
    return (
      <div className={cx('w-full', className)}>
        {title && (
          <h3 className="text-sm font-medium text-base-200 mb-3">{title}</h3>
        )}
        <div className="space-y-3">
          {data.map((item, index) => {
            const percentage = (item.value / max) * 100
            const color = item.color || defaultColors[index % defaultColors.length]
            
            return (
              <div key={item.label} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-base-300">{item.label}</span>
                  {showValues && (
                    <span className="text-base-400 font-medium">
                      {item.value.toLocaleString()}
                    </span>
                  )}
                </div>
                <div className="h-2 bg-base-800 rounded-full overflow-hidden">
                  <div
                    className={cx('h-full rounded-full transition-all duration-500', color)}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }
  
  // Vertical bar chart
  return (
    <div className={cx('w-full', className)}>
      {title && (
        <h3 className="text-sm font-medium text-base-200 mb-3">{title}</h3>
      )}
      <div
        className="flex items-end justify-around gap-2"
        style={{ height }}
      >
        {data.map((item, index) => {
          const percentage = (item.value / max) * 100
          const color = item.color || defaultColors[index % defaultColors.length]
          
          return (
            <div
              key={item.label}
              className="flex flex-col items-center gap-1 flex-1"
            >
              <div
                className="w-full relative"
                style={{ height: height - 40 }}
              >
                <div
                  className={cx(
                    'absolute bottom-0 left-0 right-0 rounded-t-md transition-all duration-500',
                    color,
                  )}
                  style={{ height: `${percentage}%` }}
                />
              </div>
              {showValues && (
                <span className="text-xs text-base-400 font-medium">
                  {item.value.toLocaleString()}
                </span>
              )}
              <span className="text-xs text-base-300 truncate max-w-full">
                {item.label}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Comparison bar for before/after or two values
interface ComparisonBarProps {
  label: string
  value1: number
  value2: number
  label1?: string
  label2?: string
  showDiff?: boolean
  className?: string
}

export function ComparisonBar({
  label,
  value1,
  value2,
  label1 = 'Before',
  label2 = 'After',
  showDiff = true,
  className,
}: ComparisonBarProps) {
  const max = Math.max(value1, value2, 1)
  const diff = value2 - value1
  const diffPercent = value1 > 0 ? ((diff / value1) * 100).toFixed(1) : '0'
  
  return (
    <div className={cx('space-y-2', className)}>
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-base-200">{label}</span>
        {showDiff && (
          <span
            className={cx(
              'text-xs font-medium',
              diff > 0 ? 'text-brand-600' : diff < 0 ? 'text-danger-600' : 'text-base-400',
            )}
          >
            {diff > 0 ? '+' : ''}{diffPercent}%
          </span>
        )}
      </div>
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-xs text-base-400 w-12">{label1}</span>
          <div className="flex-1 h-2 bg-base-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-base-500 rounded-full transition-all duration-500"
              style={{ width: `${(value1 / max) * 100}%` }}
            />
          </div>
          <span className="text-xs text-base-400 w-16 text-right">
            {value1.toLocaleString()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-base-400 w-12">{label2}</span>
          <div className="flex-1 h-2 bg-base-800 rounded-full overflow-hidden">
            <div
              className={cx(
                'h-full rounded-full transition-all duration-500',
                diff >= 0 ? 'bg-brand-500' : 'bg-danger-500',
              )}
              style={{ width: `${(value2 / max) * 100}%` }}
            />
          </div>
          <span className="text-xs text-base-400 w-16 text-right">
            {value2.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  )
}

// Stacked bar for composition display (e.g., forest vs non-forest)
interface StackedBarData {
  label: string
  value: number
  color: string
}

interface StackedBarProps {
  data: StackedBarData[]
  title?: string
  showLegend?: boolean
  className?: string
}

export function StackedBar({
  data,
  title,
  showLegend = true,
  className,
}: StackedBarProps) {
  const total = data.reduce((sum, item) => sum + item.value, 0)
  
  return (
    <div className={cx('w-full', className)}>
      {title && (
        <h3 className="text-sm font-medium text-base-200 mb-2">{title}</h3>
      )}
      <div className="h-4 bg-base-800 rounded-full overflow-hidden flex">
        {data.map((item, index) => {
          const percentage = total > 0 ? (item.value / total) * 100 : 0
          return (
            <div
              key={item.label}
              className={cx('h-full transition-all duration-500', item.color)}
              style={{ width: `${percentage}%` }}
              title={`${item.label}: ${percentage.toFixed(1)}%`}
            />
          )
        })}
      </div>
      {showLegend && (
        <div className="flex flex-wrap gap-3 mt-2">
          {data.map((item) => {
            const percentage = total > 0 ? (item.value / total) * 100 : 0
            return (
              <div key={item.label} className="flex items-center gap-1.5">
                <span className={cx('w-2 h-2 rounded-full', item.color)} />
                <span className="text-xs text-base-300">
                  {item.label}: {percentage.toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
