import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Map, Upload, Clock, BarChart2, Settings, ChevronLeft, ChevronRight, Leaf } from 'lucide-react'

const NAV_ITEMS = [
  { icon: Map, label: 'New Analysis', to: '/app' },
  { icon: Upload, label: 'Upload', to: '/app/upload' },
  { icon: Clock, label: 'Run History', to: '/app/runs' },
  { icon: BarChart2, label: 'Analytics', to: '/app/analytics' },
  { icon: Settings, label: 'Settings', to: '/app/settings' },
]

export function Sidebar() {
  const [expanded, setExpanded] = useState(true)
  const location = useLocation()

  return (
    <>
      {/* Desktop sidebar */}
      <aside
        className={`hidden md:flex flex-col fixed left-0 top-0 h-full z-40 border-r border-cv-border bg-cv-surface transition-all duration-200 ${
          expanded ? 'w-[220px]' : 'w-16'
        }`}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-cv-border min-h-[64px]">
          <div className="shrink-0 w-8 h-8 rounded-lg bg-cv-primary-muted flex items-center justify-center">
            <Leaf className="w-4 h-4 text-cv-primary" />
          </div>
          {expanded && (
            <span className="font-bold text-sm text-cv-text-primary whitespace-nowrap">
              ClimateVision
            </span>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex-1 py-4 overflow-y-auto">
          {NAV_ITEMS.map(({ icon: Icon, label, to }) => {
            const isActive =
              to === '/app' ? location.pathname === '/app' : location.pathname.startsWith(to)
            return (
              <NavLink
                key={to}
                to={to}
                title={!expanded ? label : undefined}
                className={`flex items-center gap-3 px-4 py-3 mx-2 rounded-lg mb-1 transition-all text-sm font-medium ${
                  isActive
                    ? 'bg-cv-primary-muted text-cv-primary border-l-2 border-cv-primary'
                    : 'text-cv-text-secondary hover:bg-cv-card hover:text-cv-text-primary'
                }`}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {expanded && <span>{label}</span>}
              </NavLink>
            )
          })}
        </nav>

        {/* Collapse toggle */}
        <button
          onClick={() => setExpanded((e) => !e)}
          className="flex items-center justify-center p-4 border-t border-cv-border text-cv-text-secondary hover:text-cv-text-primary transition"
          aria-label={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          {expanded ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </aside>

      {/* Mobile bottom tab bar */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 flex border-t border-cv-border bg-cv-surface">
        {NAV_ITEMS.map(({ icon: Icon, label, to }) => {
          const isActive =
            to === '/app' ? location.pathname === '/app' : location.pathname.startsWith(to)
          return (
            <NavLink
              key={to}
              to={to}
              className={`flex-1 flex flex-col items-center gap-1 py-3 text-xs transition ${
                isActive ? 'text-cv-primary' : 'text-cv-text-secondary'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="hidden sm:block">{label}</span>
            </NavLink>
          )
        })}
      </nav>

      {/* Sidebar spacer for desktop */}
      <div className={`hidden md:block shrink-0 transition-all duration-200 ${expanded ? 'w-[220px]' : 'w-16'}`} />
    </>
  )
}
