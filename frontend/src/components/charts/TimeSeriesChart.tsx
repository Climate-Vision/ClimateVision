function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

interface DataPoint {
  date: string
  value: number
  label?: string
}

interface TimeSeriesChartProps {
  data: DataPoint[]
  title?: string
  yAxisLabel?: string
  height?: number
  showPoints?: boolean
  showArea?: boolean
  color?: string
  className?: string
}

export function TimeSeriesChart({
  data,
  title,
  yAxisLabel,
  height = 200,
  showPoints = true,
  showArea = true,
  color = 'brand',
  className,
}: TimeSeriesChartProps) {
  if (data.length === 0) {
    return (
      <div className={cx('w-full', className)}>
        {title && (
          <h3 className="text-sm font-medium text-base-200 mb-3">{title}</h3>
        )}
        <div
          className="flex items-center justify-center text-base-400 text-sm bg-base-900/50 rounded-lg"
          style={{ height }}
        >
          No data available
        </div>
      </div>
    )
  }
  
  const values = data.map((d) => d.value)
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const range = maxValue - minValue || 1
  
  // Calculate SVG path
  const chartWidth = 100 // percentage
  const chartHeight = height - 40 // Leave room for labels
  const padding = { top: 10, right: 10, bottom: 30, left: 40 }
  
  const getX = (index: number) => {
    const availableWidth = chartWidth - padding.left - padding.right
    return padding.left + (index / (data.length - 1 || 1)) * availableWidth
  }
  
  const getY = (value: number) => {
    const availableHeight = chartHeight - padding.top - padding.bottom
    const normalized = (value - minValue) / range
    return padding.top + availableHeight - normalized * availableHeight
  }
  
  // Generate path
  const linePath = data
    .map((point, index) => {
      const x = getX(index)
      const y = getY(point.value)
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`
    })
    .join(' ')
  
  // Generate area path
  const areaPath = showArea
    ? `${linePath} L ${getX(data.length - 1)} ${chartHeight - padding.bottom} L ${padding.left} ${chartHeight - padding.bottom} Z`
    : ''
  
  const colorClasses: Record<string, { stroke: string; fill: string; dot: string }> = {
    brand: {
      stroke: 'stroke-brand-500',
      fill: 'fill-brand-500/20',
      dot: 'fill-brand-500',
    },
    ocean: {
      stroke: 'stroke-ocean-500',
      fill: 'fill-ocean-500/20',
      dot: 'fill-ocean-500',
    },
    danger: {
      stroke: 'stroke-danger-500',
      fill: 'fill-danger-500/20',
      dot: 'fill-danger-500',
    },
    amber: {
      stroke: 'stroke-amber-500',
      fill: 'fill-amber-500/20',
      dot: 'fill-amber-500',
    },
  }
  
  const colors = colorClasses[color] || colorClasses.brand
  
  return (
    <div className={cx('w-full', className)}>
      {title && (
        <h3 className="text-sm font-medium text-base-200 mb-3">{title}</h3>
      )}
      <div className="relative" style={{ height }}>
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          preserveAspectRatio="none"
          className="w-full h-full"
        >
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding.top + (chartHeight - padding.top - padding.bottom) * (1 - ratio)
            return (
              <line
                key={ratio}
                x1={padding.left}
                y1={y}
                x2={chartWidth - padding.right}
                y2={y}
                className="stroke-base-800"
                strokeWidth="0.5"
              />
            )
          })}
          
          {/* Area fill */}
          {showArea && (
            <path
              d={areaPath}
              className={colors.fill}
            />
          )}
          
          {/* Line */}
          <path
            d={linePath}
            fill="none"
            className={colors.stroke}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          
          {/* Points */}
          {showPoints &&
            data.map((point, index) => (
              <circle
                key={index}
                cx={getX(index)}
                cy={getY(point.value)}
                r="3"
                className={colors.dot}
              />
            ))}
        </svg>
        
        {/* Y-axis labels */}
        <div
          className="absolute left-0 top-0 flex flex-col justify-between text-xs text-base-400"
          style={{ height: chartHeight - padding.bottom, paddingTop: padding.top }}
        >
          <span>{maxValue.toFixed(1)}</span>
          <span>{((maxValue + minValue) / 2).toFixed(1)}</span>
          <span>{minValue.toFixed(1)}</span>
        </div>
        
        {/* X-axis labels */}
        <div
          className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-base-400 px-10"
          style={{ height: padding.bottom }}
        >
          {data.length > 0 && (
            <>
              <span>{data[0].date}</span>
              {data.length > 2 && (
                <span>{data[Math.floor(data.length / 2)].date}</span>
              )}
              <span>{data[data.length - 1].date}</span>
            </>
          )}
        </div>
        
        {/* Y-axis label */}
        {yAxisLabel && (
          <div
            className="absolute -left-1 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-base-400 whitespace-nowrap"
            style={{ transformOrigin: 'center' }}
          >
            {yAxisLabel}
          </div>
        )}
      </div>
    </div>
  )
}

// Sparkline for compact inline trends
interface SparklineProps {
  data: number[]
  color?: string
  width?: number
  height?: number
  className?: string
}

export function Sparkline({
  data,
  color = 'brand',
  width = 60,
  height = 20,
  className,
}: SparklineProps) {
  if (data.length < 2) return null
  
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  
  const points = data
    .map((value, index) => {
      const x = (index / (data.length - 1)) * width
      const y = height - ((value - min) / range) * height
      return `${x},${y}`
    })
    .join(' ')
  
  const trend = data[data.length - 1] - data[0]
  
  const colorClasses: Record<string, string> = {
    brand: 'stroke-brand-500',
    ocean: 'stroke-ocean-500',
    danger: 'stroke-danger-500',
    auto: trend >= 0 ? 'stroke-brand-500' : 'stroke-danger-500',
  }
  
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox={`0 0 ${width} ${height}`}
    >
      <polyline
        points={points}
        fill="none"
        className={colorClasses[color] || colorClasses.brand}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
