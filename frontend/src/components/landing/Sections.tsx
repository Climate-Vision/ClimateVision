import { useEffect, useRef, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion, useInView, useMotionValue, useTransform, animate } from 'framer-motion'
import {
  TreePine,
  Waves,
  Snowflake,
  MapPin,
  Satellite,
  Cpu,
  Bell,
  ArrowRight,
  Leaf,
} from 'lucide-react'

function GithubIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.55v-2.17c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.75 2.69 1.25 3.34.95.1-.74.4-1.25.72-1.53-2.55-.29-5.23-1.28-5.23-5.68 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.05 0 0 .96-.31 3.15 1.18a11 11 0 0 1 5.74 0c2.19-1.49 3.15-1.18 3.15-1.18.62 1.59.23 2.76.11 3.05.74.81 1.18 1.83 1.18 3.09 0 4.41-2.69 5.38-5.25 5.67.41.35.77 1.05.77 2.12v3.14c0 .3.21.66.8.55A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z" />
    </svg>
  )
}
import { VideoSlot } from './VideoSlot'
import { ForestScene, FloodScene, IceScene, SCENE_IMAGES, KenBurns } from './AnimatedScenes'

/**
 * Full-bleed animated section background (Ken Burns motion). The image stays
 * clearly visible; white gradient scrims top+bottom keep the section's dark
 * body text readable where it overlaps the photo.
 */
function SectionBackdrop({
  src,
  variant = 'a',
  opacity = 0.55,
}: {
  src: string
  variant?: 'a' | 'b' | 'c'
  opacity?: number
}) {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <KenBurns src={src} variant={variant} imgClassName="opacity-[var(--bd-op)]" />
      <div
        className="absolute inset-0"
        style={{ ['--bd-op' as string]: opacity } as React.CSSProperties}
      />
      {/* scrims: readable center, image bleeds at the edges */}
      <div className="absolute inset-0 bg-gradient-to-b from-cv-bg via-cv-bg/40 to-cv-bg" />
    </div>
  )
}

/* ---------------------------------- utils --------------------------------- */

function Reveal({
  children,
  delay = 0,
  className = '',
}: {
  children: ReactNode
  delay?: number
  className?: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay, ease: 'easeOut' }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

function Counter({ to, suffix = '', decimals = 0 }: { to: number; suffix?: string; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: '-40px' })
  const value = useMotionValue(0)
  const rounded = useTransform(value, (v) => v.toFixed(decimals))

  useEffect(() => {
    if (inView) {
      const controls = animate(value, to, { duration: 1.8, ease: 'easeOut' })
      return controls.stop
    }
  }, [inView, to, value])

  return (
    <span ref={ref}>
      <motion.span>{rounded}</motion.span>
      {suffix}
    </span>
  )
}

/* ---------------------------------- stats --------------------------------- */

const STATS = [
  { value: 2.4, suffix: 'M km²', decimals: 1, label: 'Area analyzable per day' },
  { value: 90, suffix: '%+', decimals: 0, label: 'Flood detection accuracy' },
  { value: 3, suffix: '', decimals: 0, label: 'Analysis types, one API' },
  { value: 5, suffix: ' min', decimals: 0, label: 'From imagery to alert' },
]

export function Stats() {
  return (
    <section className="relative border-y border-cv-border bg-cv-surface/50">
      <div className="mx-auto grid max-w-7xl grid-cols-2 gap-8 px-4 py-14 sm:px-6 lg:grid-cols-4 lg:px-8">
        {STATS.map((s, i) => (
          <Reveal key={s.label} delay={i * 0.08} className="text-center">
            <div className="text-3xl font-bold text-cv-text-primary sm:text-4xl">
              <Counter to={s.value} suffix={s.suffix} decimals={s.decimals} />
            </div>
            <div className="mt-2 text-sm text-cv-text-secondary">{s.label}</div>
          </Reveal>
        ))}
      </div>
    </section>
  )
}

/* -------------------------------- features -------------------------------- */

const FEATURES = [
  {
    icon: TreePine,
    title: 'Deforestation',
    video: '/videos/feature-deforestation.mp4',
    scene: ForestScene,
    description:
      'Siamese change detection compares imagery across time to flag forest loss in the Amazon, Congo Basin, and beyond — down to individual clearings.',
    points: ['Before/after change maps', 'NDVI trend analysis', 'Protected-area watchlists'],
  },
  {
    icon: Waves,
    title: 'Flooding',
    video: '/videos/feature-flooding.mp4',
    scene: FloodScene,
    description:
      'U-Net segmentation on Sentinel-1 SAR and Sentinel-2 optical imagery maps flood extent even through cloud cover, when responders need it most.',
    points: ['SAR sees through clouds', 'Pixel-level water masks', 'Rapid disaster response'],
  },
  {
    icon: Snowflake,
    title: 'Ice melt',
    video: '/videos/feature-ice.mp4',
    scene: IceScene,
    description:
      'Track glacial retreat and arctic sea-ice decline season over season with automated surface classification and long-term trend reporting.',
    points: ['Glacier extent tracking', 'Seasonal comparisons', 'Long-term trend charts'],
  },
]

export function Features() {
  return (
    <section id="platform" className="relative py-24">
      <SectionBackdrop src={SCENE_IMAGES.forest} />
      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      <Reveal className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-bold tracking-tight text-cv-text-primary sm:text-4xl">
          One platform. Three planetary signals.
        </h2>
        <p className="mt-4 text-cv-text-secondary">
          Every analysis runs through the same pipeline: pick a region, pick a date range,
          and let the models do the reading.
        </p>
      </Reveal>

      <div className="mt-16 grid gap-6 lg:grid-cols-3">
        {FEATURES.map((f, i) => (
          <Reveal key={f.title} delay={i * 0.1}>
            <div className="group relative h-full overflow-hidden rounded-2xl border border-cv-border bg-cv-card transition-all duration-300 hover:border-cv-border-strong hover:shadow-glow">
              <div className="relative h-44 overflow-hidden">
                <VideoSlot src={f.video} fallback={<f.scene />} />
                <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-cv-card to-transparent" />
              </div>
              <div className="p-6">
                <div className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-cv-primary-muted">
                    <f.icon className="h-5 w-5 text-cv-primary" />
                  </span>
                  <h3 className="text-xl font-semibold text-cv-text-primary">{f.title}</h3>
                </div>
                <p className="mt-4 text-sm leading-relaxed text-cv-text-secondary">{f.description}</p>
                <ul className="mt-4 space-y-2">
                  {f.points.map((p) => (
                    <li key={p} className="flex items-center gap-2 text-sm text-cv-text-muted">
                      <span className="h-1 w-1 rounded-full bg-cv-primary" />
                      {p}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
      </div>
    </section>
  )
}

/* ------------------------------- how it works ------------------------------ */

const STEPS = [
  {
    icon: MapPin,
    title: 'Define your region',
    text: 'Draw a bounding box anywhere on Earth, or search a place. Set your date range.',
  },
  {
    icon: Satellite,
    title: 'Satellites deliver',
    text: 'Sentinel-2, Sentinel-1 SAR, and Landsat imagery is fetched via Google Earth Engine.',
  },
  {
    icon: Cpu,
    title: 'AI reads the pixels',
    text: 'U-Net and Siamese networks segment water, forest loss, and ice with per-pixel precision.',
  },
  {
    icon: Bell,
    title: 'You get answers',
    text: 'Change maps, area statistics, confidence scores, and alerts — in minutes, not months.',
  },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="relative border-y border-cv-border bg-cv-surface/40 py-24">
      <SectionBackdrop src={SCENE_IMAGES.ice} opacity={0.14} />
      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-cv-text-primary sm:text-4xl">
            From orbit to insight in four steps
          </h2>
          <p className="mt-4 text-cv-text-secondary">
            No GIS degree required. If you can draw a rectangle on a map, you can run a
            planetary-scale analysis.
          </p>
        </Reveal>

        <div className="relative mt-16 grid gap-10 md:grid-cols-4">
          {/* connector line */}
          <div className="absolute left-0 right-0 top-7 hidden h-px bg-gradient-to-r from-transparent via-cv-border-strong to-transparent md:block" />
          {STEPS.map((s, i) => (
            <Reveal key={s.title} delay={i * 0.12} className="relative text-center md:text-left">
              <div className="relative z-10 mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-cv-border-strong bg-cv-card shadow-card md:mx-0">
                <s.icon className="h-6 w-6 text-cv-primary" />
              </div>
              <div className="mt-5 text-sm font-semibold uppercase tracking-wider text-cv-text-dim">
                Step {i + 1}
              </div>
              <h3 className="mt-1 text-lg font-semibold text-cv-text-primary">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-cv-text-secondary">{s.text}</p>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2} className="mt-16 flex flex-wrap items-center justify-center gap-3">
          {['Sentinel-2', 'Sentinel-1 SAR', 'Landsat 8/9', 'Google Earth Engine', 'PyTorch U-Net', 'FastAPI'].map(
            (t) => (
              <span
                key={t}
                className="rounded-full border border-cv-border bg-cv-card px-4 py-1.5 text-xs font-medium text-cv-text-secondary"
              >
                {t}
              </span>
            ),
          )}
        </Reveal>
      </div>
    </section>
  )
}

/* ---------------------------------- impact --------------------------------- */

export function Impact() {
  return (
    <section id="impact" className="relative py-24">
      <SectionBackdrop src={SCENE_IMAGES.flood} opacity={0.15} />
      <div className="relative mx-auto max-w-5xl px-4 text-center sm:px-6">
      <Reveal>
        <Leaf className="mx-auto h-8 w-8 text-cv-primary" />
        <blockquote className="mt-8 text-2xl font-medium leading-relaxed text-cv-text-primary sm:text-3xl">
          &ldquo;Manual satellite analysis takes weeks and specialist staff. Automated
          monitoring means small conservation teams can respond to deforestation and floods
          in days — while intervention is still possible.&rdquo;
        </blockquote>
        <p className="mt-6 text-sm text-cv-text-muted">
          Why we built ClimateVision — open source, for NGOs and researchers everywhere
        </p>
      </Reveal>
      </div>
    </section>
  )
}

/* ----------------------------------- CTA ----------------------------------- */

export function CTA() {
  return (
    <section className="mx-auto max-w-7xl px-4 pb-24 sm:px-6 lg:px-8">
      <Reveal>
        <div className="relative overflow-hidden rounded-3xl border border-cv-border-strong bg-gradient-to-br from-cv-primary-muted via-cv-card to-cv-bg p-10 text-center sm:p-16">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(34,197,94,0.12),transparent_65%)]" />
          <div className="relative">
            <h2 className="text-3xl font-bold tracking-tight text-cv-text-primary sm:text-4xl">
              Start monitoring in minutes
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-cv-text-secondary">
              Create a free account, draw your first region, and get your first change-detection
              report today.
            </p>
            <div className="mt-8 flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Link
                to="/signup"
                className="group flex items-center gap-2 rounded-full bg-cv-primary px-7 py-3.5 text-base font-semibold text-cv-bg transition hover:bg-cv-primary-hover"
              >
                Create free account
                <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
              </Link>
              <a
                href="https://github.com/Climate-Vision/ClimateVision"
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-2 rounded-full border border-cv-border-strong px-7 py-3.5 text-base font-semibold text-cv-text-primary transition hover:border-cv-primary"
              >
                <GithubIcon className="h-5 w-5" />
                Star on GitHub
              </a>
            </div>
          </div>
        </div>
      </Reveal>
    </section>
  )
}

/* ---------------------------------- footer --------------------------------- */

export function Footer() {
  return (
    <footer className="border-t border-cv-border bg-cv-surface/60">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-6 px-4 py-10 sm:flex-row sm:px-6 lg:px-8">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-cv-primary-muted">
            <Leaf className="h-4 w-4 text-cv-primary" />
          </span>
          <span className="text-sm font-semibold text-cv-text-primary">ClimateVision</span>
          <span className="text-xs text-cv-text-dim">© {new Date().getFullYear()} · MIT License</span>
        </div>
        <div className="flex items-center gap-6 text-sm text-cv-text-secondary">
          <Link to="/app" className="transition hover:text-cv-text-primary">
            Dashboard
          </Link>
          <a
            href="https://climatevision.green/docs"
            target="_blank"
            rel="noreferrer"
            className="transition hover:text-cv-text-primary"
          >
            API Docs
          </a>
          <a
            href="https://github.com/Climate-Vision/ClimateVision"
            target="_blank"
            rel="noreferrer"
            className="transition hover:text-cv-text-primary"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  )
}
