import { useEffect, useRef } from 'react'

/**
 * Lightweight rotating particle Earth rendered on <canvas>.
 * Zero dependencies — used as the cinematic fallback wherever a
 * Higgsfield video has not been dropped in yet.
 */
export function ParticleGlobe({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const DPR = Math.min(window.devicePixelRatio || 1, 2)
    const N = 750
    const pts: { x: number; y: number; z: number }[] = []
    // Fibonacci sphere for even distribution
    for (let i = 0; i < N; i++) {
      const theta = Math.acos(1 - (2 * (i + 0.5)) / N)
      const phi = Math.PI * (1 + Math.sqrt(5)) * i
      pts.push({
        x: Math.sin(theta) * Math.cos(phi),
        y: Math.cos(theta),
        z: Math.sin(theta) * Math.sin(phi),
      })
    }

    let raf = 0
    let angle = 0

    const resize = () => {
      canvas.width = canvas.clientWidth * DPR
      canvas.height = canvas.clientHeight * DPR
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = () => {
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)
      const R = Math.min(w, h) * 0.36
      const cx = w / 2
      const cy = h / 2
      angle += 0.0016

      // soft atmosphere glow
      const glow = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 1.45)
      glow.addColorStop(0, 'rgba(34, 197, 94, 0.10)')
      glow.addColorStop(1, 'rgba(34, 197, 94, 0)')
      ctx.fillStyle = glow
      ctx.fillRect(0, 0, w, h)

      const cos = Math.cos(angle)
      const sin = Math.sin(angle)
      for (const p of pts) {
        const x = p.x * cos + p.z * sin
        const z = -p.x * sin + p.z * cos
        const depth = (z + 1) / 2 // 0 back, 1 front
        const sx = cx + x * R
        const sy = cy + p.y * R * 0.98
        const alpha = 0.10 + depth * 0.6
        const size = (0.5 + depth * 1.5) * DPR
        ctx.beginPath()
        ctx.fillStyle = `rgba(74, 222, 128, ${alpha.toFixed(3)})`
        ctx.arc(sx, sy, size, 0, Math.PI * 2)
        ctx.fill()
      }

      // orbiting "satellite"
      const orbT = angle * 6
      const ox = cx + Math.cos(orbT) * R * 1.25
      const oy = cy + Math.sin(orbT) * R * 0.45
      ctx.beginPath()
      ctx.fillStyle = 'rgba(240, 253, 244, 0.95)'
      ctx.arc(ox, oy, 2.2 * DPR, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.15)'
      ctx.lineWidth = 1 * DPR
      ctx.ellipse(cx, cy, R * 1.25, R * 0.45, 0, 0, Math.PI * 2)
      ctx.stroke()

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={canvasRef} className={`h-full w-full ${className}`} aria-hidden="true" />
}
