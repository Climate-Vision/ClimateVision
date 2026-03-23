import { ReactNode } from 'react'

interface EmptyStateProps {
  icon?: ReactNode
  heading: string
  subtext?: string
  action?: ReactNode
}

export function EmptyState({ icon, heading, subtext, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      {icon && <div className="mb-4 text-cv-text-dim opacity-60">{icon}</div>}
      <h3 className="text-lg font-semibold text-cv-text-primary mb-2">{heading}</h3>
      {subtext && <p className="text-sm text-cv-text-secondary max-w-xs mb-6">{subtext}</p>}
      {action && <div>{action}</div>}
    </div>
  )
}

// Satellite SVG illustration
export function SatelliteIllustration() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="30" y="30" width="20" height="20" rx="3" fill="#1c2a1f" stroke="#2d4a33" strokeWidth="2"/>
      <rect x="8" y="36" width="20" height="8" rx="2" fill="#14532d" stroke="#22c55e" strokeWidth="1.5"/>
      <rect x="52" y="36" width="20" height="8" rx="2" fill="#14532d" stroke="#22c55e" strokeWidth="1.5"/>
      <circle cx="40" cy="40" r="4" fill="#22c55e" opacity="0.6"/>
      <line x1="40" y1="50" x2="55" y2="65" stroke="#22c55e" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.5"/>
      <circle cx="58" cy="68" r="5" fill="#14532d" stroke="#22c55e" strokeWidth="1.5" opacity="0.7"/>
    </svg>
  )
}
