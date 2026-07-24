import { Link } from 'react-router-dom'
import { motion, useScroll, useTransform } from 'framer-motion'
import { ArrowRight, Play, Satellite } from 'lucide-react'
import { VideoSlot } from './VideoSlot'
import { ParticleGlobe } from './ParticleGlobe'
import { DriftingClouds, KenBurns, SCENE_IMAGES } from './AnimatedScenes'

export function Hero() {
  const { scrollY } = useScroll()
  const contentY = useTransform(scrollY, [0, 600], [0, 120])
  const contentOpacity = useTransform(scrollY, [0, 500], [1, 0])

  return (
    <section className="relative flex min-h-screen items-center justify-center overflow-hidden">
      {/* Background: optional Higgsfield hero clip, falls back to the animated grid-earth photo */}
      <VideoSlot
        src="/videos/hero-earth.mp4"
        fallback={
          <div className="relative h-full w-full bg-[#04120c]">
            {/* the uploaded grid-earth image with a slow Ken Burns motion */}
            <KenBurns src={SCENE_IMAGES.hero} variant="a" />
            {/* particle globe adds depth on top of the photo */}
            <div className="absolute inset-x-0 top-[8%] mx-auto h-[85vh] max-w-4xl opacity-40 mix-blend-screen">
              <ParticleGlobe />
            </div>
            <DriftingClouds className="opacity-40" />
          </div>
        }
      />
      {/* Readability scrim — dark so the light hero text stays legible over the photo */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[#04120c]/70 via-[#04120c]/30 to-[#04120c]/85" />

      <motion.div
        style={{ y: contentY, opacity: contentOpacity }}
        className="relative z-10 mx-auto max-w-5xl px-4 pt-24 text-center sm:px-6"
      >
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto mb-6 flex w-fit items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-1.5 backdrop-blur"
        >
          <Satellite className="h-3.5 w-3.5 text-brand-400" />
          <span className="text-xs font-medium tracking-wide text-white/85">
            Satellite intelligence for a changing planet
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-4xl font-bold leading-[1.08] tracking-tight text-white drop-shadow-[0_2px_12px_rgba(0,0,0,0.5)] sm:text-6xl lg:text-7xl"
        >
          See environmental change
          <span className="block bg-gradient-to-r from-brand-400 via-ocean-400 to-brand-400 bg-clip-text text-transparent">
            before it&apos;s irreversible
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-white/80 drop-shadow-[0_1px_8px_rgba(0,0,0,0.5)] sm:text-lg"
        >
          ClimateVision turns Sentinel-2 and Landsat imagery into automated alerts for
          deforestation, flooding, and ice melt — powered by deep learning, built for
          conservation teams and researchers.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row"
        >
          <Link
            to="/signup"
            className="group flex items-center gap-2 rounded-full bg-cv-primary px-7 py-3.5 text-base font-semibold text-cv-bg shadow-glow transition hover:bg-cv-primary-hover"
          >
            Start monitoring free
            <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
          </Link>
          <Link
            to="/app"
            className="flex items-center gap-2 rounded-full border border-white/25 bg-white/10 px-7 py-3.5 text-base font-semibold text-white backdrop-blur transition hover:border-brand-400 hover:bg-white/20"
          >
            <Play className="h-4 w-4 text-brand-400" />
            Explore the live demo
          </Link>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.6 }}
          className="mt-8 text-xs uppercase tracking-[0.2em] text-white/60"
        >
          Open source · MIT licensed · Built on Google Earth Engine
        </motion.p>
      </motion.div>

      {/* Scroll cue */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
          className="flex h-9 w-6 items-start justify-center rounded-full border border-white/40 p-1.5"
        >
          <div className="h-2 w-1 rounded-full bg-brand-400" />
        </motion.div>
      </motion.div>
    </section>
  )
}
