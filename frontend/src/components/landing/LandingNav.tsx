import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Leaf, ArrowRight } from 'lucide-react'

const LINKS = [
  { label: 'Platform', href: '#platform' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Impact', href: '#impact' },
]

export function LandingNav() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'border-b border-cv-border bg-cv-bg/80 backdrop-blur-xl'
          : 'border-b border-transparent bg-transparent'
      }`}
    >
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-cv-primary-muted">
            <Leaf className="h-5 w-5 text-cv-primary" />
          </span>
          <span className={`text-lg font-semibold tracking-tight transition-colors ${
            scrolled ? 'text-cv-text-primary' : 'text-white drop-shadow-[0_1px_6px_rgba(0,0,0,0.5)]'
          }`}>
            ClimateVision
          </span>
        </Link>

        <div className="hidden items-center gap-8 md:flex">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className={`text-sm font-medium transition ${
                scrolled
                  ? 'text-cv-text-secondary hover:text-cv-text-primary'
                  : 'text-white/80 hover:text-white'
              }`}
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <Link
            to="/signin"
            className={`hidden text-sm font-medium transition sm:block ${
              scrolled
                ? 'text-cv-text-secondary hover:text-cv-text-primary'
                : 'text-white/80 hover:text-white'
            }`}
          >
            Sign in
          </Link>
          <Link
            to="/signup"
            className="group flex items-center gap-1.5 rounded-full bg-cv-primary px-4 py-2 text-sm font-semibold text-cv-bg transition hover:bg-cv-primary-hover"
          >
            Get started
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
        </div>
      </nav>
    </header>
  )
}
