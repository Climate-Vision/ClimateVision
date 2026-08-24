import { useState, type ReactNode } from 'react'

/**
 * Renders a looping background video if the file exists (drop Higgsfield
 * exports into frontend/public/videos/), otherwise renders the code-driven
 * fallback. See public/videos/README.md for the expected filenames.
 */
export function VideoSlot({
  src,
  poster,
  fallback,
  className = '',
}: {
  src: string
  poster?: string
  fallback: ReactNode
  className?: string
}) {
  const [failed, setFailed] = useState(false)
  const [loaded, setLoaded] = useState(false)

  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`} aria-hidden="true">
      {(failed || !loaded) && <div className="absolute inset-0">{fallback}</div>}
      {!failed && (
        <video
          src={src}
          poster={poster}
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
          onError={() => setFailed(true)}
          onLoadedData={() => setLoaded(true)}
          className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-700 ${
            loaded ? 'opacity-100' : 'opacity-0'
          }`}
        />
      )}
    </div>
  )
}
