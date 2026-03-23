import { ReactNode, useState } from 'react'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right'

interface TooltipProps {
  content: ReactNode
  children: ReactNode
  position?: TooltipPosition
  className?: string
  delay?: number
}

const positionStyles: Record<TooltipPosition, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
}

const arrowStyles: Record<TooltipPosition, string> = {
  top: 'top-full left-1/2 -translate-x-1/2 border-t-base-700 border-x-transparent border-b-transparent',
  bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-base-700 border-x-transparent border-t-transparent',
  left: 'left-full top-1/2 -translate-y-1/2 border-l-base-700 border-y-transparent border-r-transparent',
  right: 'right-full top-1/2 -translate-y-1/2 border-r-base-700 border-y-transparent border-l-transparent',
}

export function Tooltip({
  content,
  children,
  position = 'top',
  className,
  delay = 200,
}: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null)

  const showTooltip = () => {
    const id = setTimeout(() => setVisible(true), delay)
    setTimeoutId(id)
  }

  const hideTooltip = () => {
    if (timeoutId) {
      clearTimeout(timeoutId)
      setTimeoutId(null)
    }
    setVisible(false)
  }

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      {children}
      {visible && (
        <div
          className={cx(
            'absolute z-50 px-2 py-1 text-xs font-medium text-base-100',
            'bg-base-800 border border-base-700 rounded-lg shadow-lg',
            'whitespace-nowrap animate-in fade-in duration-150',
            positionStyles[position],
            className,
          )}
          role="tooltip"
        >
          {content}
          <span
            className={cx(
              'absolute border-4',
              arrowStyles[position],
            )}
          />
        </div>
      )}
    </div>
  )
}

// Info icon with tooltip for explanations
interface InfoTooltipProps {
  content: ReactNode
  position?: TooltipPosition
}

export function InfoTooltip({ content, position = 'top' }: InfoTooltipProps) {
  return (
    <Tooltip content={content} position={position}>
      <button
        type="button"
        className="inline-flex items-center justify-center w-4 h-4 text-base-400 hover:text-base-200 transition"
        aria-label="More information"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className="w-4 h-4"
        >
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </Tooltip>
  )
}
