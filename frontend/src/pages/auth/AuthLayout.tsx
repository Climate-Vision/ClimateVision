import { type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Leaf } from 'lucide-react'
import { ParticleGlobe } from '../../components/landing/ParticleGlobe'
import { VideoSlot } from '../../components/landing/VideoSlot'

export function AuthLayout({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle: ReactNode
  children: ReactNode
}) {
  return (
    <div className="flex min-h-screen bg-cv-bg">
      {/* Brand panel */}
      <div className="relative hidden w-1/2 overflow-hidden border-r border-cv-border lg:block">
        <VideoSlot
          src="/videos/auth-ambient.mp4"
          fallback={
            <div className="relative h-full w-full">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(20,83,45,0.4),transparent_65%)]" />
              <div className="absolute inset-0 opacity-70">
                <ParticleGlobe />
              </div>
            </div>
          }
        />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-cv-bg via-transparent to-cv-bg/60" />
        <div className="absolute inset-x-0 bottom-0 p-10">
          <blockquote className="max-w-md text-xl font-medium leading-relaxed text-cv-text-primary">
            &ldquo;The planet is changing faster than we can watch it. So we taught machines
            to watch with us.&rdquo;
          </blockquote>
          <p className="mt-4 text-sm text-cv-text-muted">The ClimateVision project</p>
        </div>
      </div>

      {/* Form panel */}
      <div className="flex w-full items-center justify-center px-4 py-12 lg:w-1/2">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md"
        >
          <Link to="/" className="mb-10 flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-cv-primary-muted">
              <Leaf className="h-5 w-5 text-cv-primary" />
            </span>
            <span className="text-lg font-semibold text-cv-text-primary">ClimateVision</span>
          </Link>
          <h1 className="text-3xl font-bold tracking-tight text-cv-text-primary">{title}</h1>
          <p className="mt-2 text-sm text-cv-text-secondary">{subtitle}</p>
          <div className="mt-8">{children}</div>
        </motion.div>
      </div>
    </div>
  )
}
