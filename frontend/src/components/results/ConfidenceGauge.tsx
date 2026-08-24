import { useEffect, useState } from 'react'

interface ConfidenceGaugeProps {
  value: number // 0-100
  size?: number
}

export function ConfidenceGauge({ value, size = 120 }: ConfidenceGaugeProps) {
  const [animated, setAnimated] = useState(0)
  const r = (size / 2) * 0.75
  const cx = size / 2
  const circumference = 2 * Math.PI * r
  const arc = (animated / 100) * circumference * 0.75 // 270 degree arc

  const color = animated >= 70 ? '#22c55e' : animated >= 40 ? '#f59e0b' : '#ef4444'

  useEffect(() => {
    const timer = setTimeout(() => {
      let start = 0
      const step = () => {
        start += 2
        setAnimated(Math.min(start, value))
        if (start < value) requestAnimationFrame(step)
      }
      requestAnimationFrame(step)
    }, 200)
    return () => clearTimeout(timer)
  }, [value])

  const dashArray = `${arc} ${circumference}`
  const rotation = -135 // start from bottom-left

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Track */}
        <circle
          cx={cx}
          cy={cx}
          r={r}
          fill="none"
          stroke="#e4e9e6"
          strokeWidth={8}
          strokeDasharray={`${circumference * 0.75} ${circumference}`}
          strokeDashoffset={0}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cx})`}
        />
        {/* Progress */}
        <circle
          cx={cx}
          cy={cx}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeDasharray={dashArray}
          strokeDashoffset={0}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cx})`}
          style={{ transition: 'stroke 0.3s ease' }}
        />
        {/* Center text */}
        <text x={cx} y={cx - 2} textAnchor="middle" dominantBaseline="middle" fill={color} fontSize={size * 0.2} fontWeight="700">
          {Math.round(animated)}%
        </text>
      </svg>
      <span className="text-xs text-cv-text-secondary">Detection Confidence</span>
    </div>
  )
}
