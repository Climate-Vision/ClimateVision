import { ReactNode } from 'react'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

export type CardVariant = 'default' | 'elevated' | 'outlined' | 'success' | 'warning' | 'danger'

interface CardProps {
  title?: string
  subtitle?: string
  children: ReactNode
  right?: ReactNode
  footer?: ReactNode
  variant?: CardVariant
  className?: string
  onClick?: () => void
  hoverable?: boolean
}

const variantStyles: Record<CardVariant, string> = {
  default: 'border-base-800 bg-base-900/70',
  elevated: 'border-base-700 bg-base-900/90 shadow-lg',
  outlined: 'border-base-700 bg-transparent',
  success: 'border-brand-600/40 bg-brand-900/20',
  warning: 'border-amber-500/40 bg-amber-900/20',
  danger: 'border-danger-500/40 bg-danger-900/20',
}

export function Card({
  title,
  subtitle,
  children,
  right,
  footer,
  variant = 'default',
  className,
  onClick,
  hoverable = false,
}: CardProps) {
  const Component = onClick ? 'button' : 'div'
  
  return (
    <Component
      onClick={onClick}
      className={cx(
        'rounded-2xl border shadow-soft backdrop-blur px-5 py-4 text-left w-full',
        variantStyles[variant],
        hoverable && 'transition-all hover:bg-base-800/50 hover:border-base-700 cursor-pointer',
        onClick && 'cursor-pointer',
        className,
      )}
    >
      {(title || right) && (
        <div className="flex items-start justify-between gap-3">
          <div>
            {title && (
              <h2 className="text-base-100 font-semibold tracking-tight">{title}</h2>
            )}
            {subtitle && (
              <p className="mt-1 text-sm text-base-200/70">{subtitle}</p>
            )}
          </div>
          {right}
        </div>
      )}
      <div className={cx(title && 'mt-4')}>{children}</div>
      {footer && (
        <div className="mt-4 pt-4 border-t border-base-800">{footer}</div>
      )}
    </Component>
  )
}

// Compact card for grid displays
interface CompactCardProps {
  children: ReactNode
  className?: string
  onClick?: () => void
  selected?: boolean
}

export function CompactCard({ children, className, onClick, selected }: CompactCardProps) {
  return (
    <button
      onClick={onClick}
      className={cx(
        'text-left rounded-xl border bg-base-950/30 px-4 py-3 transition w-full',
        'hover:bg-base-950/50',
        selected ? 'border-brand-500 bg-brand-900/20' : 'border-base-800',
        className,
      )}
    >
      {children}
    </button>
  )
}
