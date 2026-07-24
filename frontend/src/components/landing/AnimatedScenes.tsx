/**
 * Ambient, code-driven motion scenes (MotionSites-style).
 * Used as living fallbacks inside VideoSlot until Higgsfield clips are added —
 * pure CSS/SVG animation, no assets, GPU-friendly transforms only.
 * Keyframes live in styles.css (cv-drift, cv-sway, cv-wave, cv-fall, cv-bob).
 */

/* ------------------------------ scene photos ------------------------------ */
/**
 * Local photography — drop the four images into frontend/public/scenes/.
 * See public/scenes/README.md for which uploaded image maps to which file.
 *   hero   → grid-earth from space   (public/scenes/hero-earth.jpg)
 *   forest → rainforest from space   (public/scenes/forest.jpg)
 *   ice    → arctic sea ice          (public/scenes/ice.jpg)
 *   flood  → river delta from above  (public/scenes/river.jpg)
 */
export const SCENE_IMAGES = {
  hero: '/scenes/hero-earth.jpg',
  forest: '/scenes/forest.jpg',
  flood: '/scenes/river.jpg',
  ice: '/scenes/ice.jpg',
}

/**
 * Full-bleed photographic backdrop with a slow Ken Burns zoom/pan.
 * `variant` picks one of three motion paths so stacked scenes don't sync up.
 * `overlay` is a Tailwind gradient/overlay class layered on top for legibility.
 */
export function KenBurns({
  src,
  variant = 'a',
  className = '',
  imgClassName = '',
  overlay,
}: {
  src: string
  variant?: 'a' | 'b' | 'c'
  className?: string
  imgClassName?: string
  overlay?: string
}) {
  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`} aria-hidden="true">
      <img
        src={src}
        alt=""
        loading="lazy"
        className={`absolute inset-0 h-full w-full object-cover cv-kenburns-${variant} ${imgClassName}`}
      />
      {overlay && <div className={`absolute inset-0 ${overlay}`} />}
    </div>
  )
}

function SceneImage({ src, variant = 'a' }: { src: string; variant?: 'a' | 'b' | 'c' }) {
  return (
    <>
      <img
        src={src}
        alt=""
        loading="lazy"
        className={`absolute inset-0 h-full w-full object-cover cv-kenburns-${variant}`}
      />
      {/* soft top fade so overlaid card text stays legible */}
      <div className="absolute inset-0 bg-gradient-to-t from-cv-card/80 via-transparent to-transparent" />
    </>
  )
}

/* ------------------------------- clouds ---------------------------------- */

const CLOUDS = [
  { top: '12%', scale: 1.0, dur: 95, delay: -20, opacity: 0.16 },
  { top: '28%', scale: 0.7, dur: 70, delay: -55, opacity: 0.12 },
  { top: '6%', scale: 1.4, dur: 130, delay: -80, opacity: 0.10 },
  { top: '40%', scale: 0.9, dur: 110, delay: -35, opacity: 0.08 },
]

export function DriftingClouds({ className = '' }: { className?: string }) {
  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`} aria-hidden="true">
      {CLOUDS.map((c, i) => (
        <div
          key={i}
          className="absolute inset-x-0"
          style={{
            top: c.top,
            animation: `cv-drift ${c.dur}s linear infinite`,
            animationDelay: `${c.delay}s`,
            opacity: c.opacity,
          }}
        >
          <div
            className="h-16 w-64 rounded-full bg-cv-text-primary blur-2xl"
            style={{ transform: `scale(${c.scale})` }}
          />
        </div>
      ))}
    </div>
  )
}

/* ------------------------------- forest ---------------------------------- */

function Tree({ x, h, delay, opacity }: { x: number; h: number; delay: number; opacity: number }) {
  const w = h * 0.6
  return (
    <g
      style={{
        animation: `cv-sway 6s ease-in-out infinite`,
        animationDelay: `${delay}s`,
        transformOrigin: `${x}px 100px`,
        transformBox: 'view-box',
      }}
      opacity={opacity}
    >
      <polygon
        points={`${x},${100 - h} ${x - w / 2},100 ${x + w / 2},100`}
        fill="currentColor"
      />
      <rect x={x - 1} y={97} width={2} height={4} fill="currentColor" />
    </g>
  )
}

export function ForestScene() {
  // deterministic pseudo-random layout — dark silhouettes over the photo
  const layers = [
    { count: 11, minH: 18, maxH: 28, opacity: 0.45, y: 4 },
    { count: 8, minH: 26, maxH: 40, opacity: 0.85, y: 8 },
  ]
  return (
    <div className="relative h-full w-full overflow-hidden bg-cv-bg">
      <SceneImage src={SCENE_IMAGES.forest} variant="a" />
      <DriftingClouds className="opacity-60" />
      {layers.map((l, li) => (
        <svg
          key={li}
          viewBox="0 0 200 100"
          preserveAspectRatio="xMidYMax slice"
          className="absolute inset-x-0 bottom-0 h-full w-full text-[#08130c]"
          style={{ transform: `translateY(${l.y}px)` }}
        >
          {Array.from({ length: l.count }, (_, i) => {
            const seed = Math.sin((li + 1) * (i + 1) * 7.13) * 0.5 + 0.5
            const x = (i + 0.5) * (200 / l.count) + (seed - 0.5) * 8
            const h = l.minH + seed * (l.maxH - l.minH)
            return <Tree key={i} x={x} h={h} delay={-seed * 6} opacity={l.opacity} />
          })}
        </svg>
      ))}
      {/* satellite scan sweep */}
      <div className="absolute inset-0" style={{ animation: 'cv-drift 9s linear infinite' }}>
        <div className="absolute inset-y-0 left-0 w-16 bg-gradient-to-r from-transparent via-brand-400/25 to-transparent" />
      </div>
    </div>
  )
}

/* ------------------------------- flooding -------------------------------- */

const WAVE_PATH =
  'M0,20 C20,12 40,28 60,20 C80,12 100,28 120,20 C140,12 160,28 180,20 C200,12 220,28 240,20 L240,60 L0,60 Z'

export function FloodScene() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-cv-bg">
      <SceneImage src={SCENE_IMAGES.flood} variant="b" />
      <DriftingClouds className="opacity-50" />
      {/* rain */}
      {Array.from({ length: 18 }, (_, i) => {
        const seed = Math.sin((i + 1) * 12.9898) * 0.5 + 0.5
        return (
          <div
            key={i}
            className="absolute inset-y-0"
            style={{
              left: `${(i / 18) * 100 + seed * 4}%`,
              animation: `cv-fall ${0.9 + seed * 0.8}s linear infinite`,
              animationDelay: `${-seed * 2}s`,
            }}
          >
            <div
              className="w-px bg-gradient-to-b from-transparent via-ocean-400/50 to-ocean-400/20"
              style={{ height: `${18 + seed * 14}%` }}
            />
          </div>
        )
      })}
      {/* rising water: two drifting wave layers */}
      <div
        className="absolute inset-x-0 bottom-0 h-1/2"
        style={{ animation: 'cv-bob 7s ease-in-out infinite' }}
      >
        <svg viewBox="0 0 240 60" preserveAspectRatio="none" className="absolute bottom-0 h-full w-[200%]" style={{ animation: 'cv-wave 11s linear infinite' }}>
          <path d={WAVE_PATH} fill="rgba(6,182,212,0.25)" />
        </svg>
        <svg viewBox="0 0 240 60" preserveAspectRatio="none" className="absolute -bottom-2 h-full w-[200%]" style={{ animation: 'cv-wave 7s linear infinite reverse' }}>
          <path d={WAVE_PATH} fill="rgba(6,182,212,0.35)" />
        </svg>
      </div>
    </div>
  )
}

/* --------------------------------- ice ----------------------------------- */

export function IceScene() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-cv-bg">
      <SceneImage src={SCENE_IMAGES.ice} variant="c" />
      <DriftingClouds className="opacity-40" />
      {/* snowfall */}
      {Array.from({ length: 22 }, (_, i) => {
        const seed = Math.sin((i + 1) * 78.233) * 0.5 + 0.5
        return (
          <div
            key={i}
            className="absolute inset-y-0"
            style={{
              left: `${(i / 22) * 100 + seed * 4}%`,
              animation: `cv-fall ${4 + seed * 5}s linear infinite`,
              animationDelay: `${-seed * 8}s`,
            }}
          >
            <div
              className="rounded-full bg-cv-text-primary/60"
              style={{
                width: `${2 + seed * 2}px`,
                height: `${2 + seed * 2}px`,
                opacity: 0.3 + seed * 0.5,
              }}
            />
          </div>
        )
      })}
      {/* drifting ice floes */}
      {[
        { left: '8%', w: 70, delay: 0, dur: 9 },
        { left: '45%', w: 100, delay: -3, dur: 11 },
        { left: '74%', w: 55, delay: -6, dur: 8 },
      ].map((f, i) => (
        <div
          key={i}
          className="absolute bottom-4 h-4 rounded-[40%] bg-cv-text-primary/25 blur-[1px]"
          style={{ left: f.left, width: f.w, animation: `cv-bob ${f.dur}s ease-in-out infinite`, animationDelay: `${f.delay}s` }}
        />
      ))}
      {/* water line */}
      <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-ocean-600/30 to-transparent" />
    </div>
  )
}
