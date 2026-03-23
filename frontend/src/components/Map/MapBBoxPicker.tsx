import { useEffect, useRef, useState, useCallback } from 'react'
import { MapPin, Trash2, Maximize2, Pencil, Hand } from 'lucide-react'

interface MapBBoxPickerProps {
  value: number[] | null
  onChange: (bbox: number[] | null) => void
  apiKey: string
}

declare global {
  interface Window {
    google: typeof google
    initGoogleMaps?: () => void
    _googleMapsLoaded?: boolean
  }
}

function loadGoogleMapsScript(apiKey: string): Promise<void> {
  if (window._googleMapsLoaded) return Promise.resolve()
  if (!apiKey || apiKey === 'YOUR_GOOGLE_MAPS_API_KEY_HERE') return Promise.resolve()
  return new Promise((resolve, reject) => {
    if (document.querySelector('script[data-gmaps]')) {
      const check = setInterval(() => {
        if (window.google?.maps) {
          window._googleMapsLoaded = true
          clearInterval(check)
          resolve()
        }
      }, 100)
      return
    }
    window.initGoogleMaps = () => {
      window._googleMapsLoaded = true
      resolve()
    }
    const script = document.createElement('script')
    script.setAttribute('data-gmaps', '1')
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places&callback=initGoogleMaps&loading=async`
    script.async = true
    script.defer = true
    script.onerror = () => {
      window._googleMapsLoaded = false
      reject(new Error('Failed to load Google Maps script'))
    }
    document.head.appendChild(script)
  })
}

export function MapBBoxPicker({ value, onChange, apiKey }: MapBBoxPickerProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const mapInstance = useRef<google.maps.Map | null>(null)
  const rectangle = useRef<google.maps.Rectangle | null>(null)

  const [mapType, setMapType] = useState<'satellite' | 'roadmap' | 'hybrid'>('satellite')
  const [mapsReady, setMapsReady] = useState(false)
  const [noKey, setNoKey] = useState(false)
  const [drawMode, setDrawMode] = useState(false)

  const updateBBoxFromRectangle = useCallback((rect: google.maps.Rectangle) => {
    const bounds = rect.getBounds()
    if (!bounds) return
    const sw = bounds.getSouthWest()
    const ne = bounds.getNorthEast()
    onChange([sw.lng(), sw.lat(), ne.lng(), ne.lat()])
  }, [onChange])

  const clearRectangle = useCallback(() => {
    rectangle.current?.setMap(null)
    rectangle.current = null
    onChange(null)
  }, [onChange])

  const useCurrentView = useCallback(() => {
    if (!mapInstance.current) return
    const bounds = mapInstance.current.getBounds()
    if (!bounds) return
    const sw = bounds.getSouthWest()
    const ne = bounds.getNorthEast()
    onChange([sw.lng(), sw.lat(), ne.lng(), ne.lat()])

    if (rectangle.current) {
      rectangle.current.setBounds(bounds)
    } else {
      rectangle.current = new google.maps.Rectangle({
        map: mapInstance.current,
        bounds,
        strokeColor: '#22c55e',
        strokeOpacity: 0.9,
        strokeWeight: 2,
        fillColor: '#22c55e',
        fillOpacity: 0.12,
        editable: true,
        draggable: true,
      })
      rectangle.current.addListener('bounds_changed', () => {
        if (rectangle.current) updateBBoxFromRectangle(rectangle.current)
      })
    }
  }, [onChange, updateBBoxFromRectangle])

  // Apply draw mode to map (cursor + draggability)
  useEffect(() => {
    if (!mapInstance.current) return
    if (drawMode) {
      mapInstance.current.setOptions({ draggable: false, gestureHandling: 'none' })
      if (mapRef.current) mapRef.current.style.cursor = 'crosshair'
    } else {
      mapInstance.current.setOptions({ draggable: true, gestureHandling: 'auto' })
      if (mapRef.current) mapRef.current.style.cursor = ''
    }
  }, [drawMode])

  useEffect(() => {
    if (!apiKey || apiKey === 'YOUR_GOOGLE_MAPS_API_KEY_HERE') {
      setNoKey(true)
      return
    }
    loadGoogleMapsScript(apiKey)
      .then(() => setMapsReady(true))
      .catch(() => setNoKey(true))
  }, [apiKey])

  useEffect(() => {
    if (!mapsReady || !mapRef.current) return

    const map = new google.maps.Map(mapRef.current, {
      center: { lat: 20, lng: 0 },
      zoom: 2,
      mapTypeId: mapType,
      disableDefaultUI: false,
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
      zoomControl: true,
      styles: [
        { elementType: 'geometry', stylers: [{ color: '#1a2e20' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#86efac' }] },
        { elementType: 'labels.text.stroke', stylers: [{ color: '#0a0f0d' }] },
        { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0c2340' }] },
        { featureType: 'road', stylers: [{ visibility: 'simplified' }] },
      ],
    })
    mapInstance.current = map

    // Places autocomplete
    if (searchRef.current) {
      const autocomplete = new google.maps.places.Autocomplete(searchRef.current)
      autocomplete.addListener('place_changed', () => {
        const place = autocomplete.getPlace()
        if (place.geometry?.viewport) {
          map.fitBounds(place.geometry.viewport)
        } else if (place.geometry?.location) {
          map.setCenter(place.geometry.location)
          map.setZoom(10)
        }
      })
    }

    // Drawing via mouse — only active in draw mode
    // We use a ref-accessed flag so the listener always sees the latest value
    const drawModeRef = { current: false }

    let startPoint: google.maps.LatLng | null = null
    let isDrawing = false

    const mousedownListener = map.addListener('mousedown', (e: google.maps.MapMouseEvent) => {
      if (!drawModeRef.current || !e.latLng) return
      startPoint = e.latLng
      isDrawing = true

      // Clear existing rectangle when starting a new draw
      if (rectangle.current) {
        rectangle.current.setMap(null)
        rectangle.current = null
      }
    })

    const mousemoveListener = map.addListener('mousemove', (e: google.maps.MapMouseEvent) => {
      if (!isDrawing || !startPoint || !e.latLng) return
      const bounds = new google.maps.LatLngBounds(startPoint, e.latLng)

      if (!rectangle.current) {
        rectangle.current = new google.maps.Rectangle({
          map,
          bounds,
          strokeColor: '#22c55e',
          strokeOpacity: 0.9,
          strokeWeight: 2,
          fillColor: '#22c55e',
          fillOpacity: 0.12,
          editable: false,
          draggable: false,
        })
      } else {
        rectangle.current.setBounds(bounds)
      }
    })

    const mouseupListener = map.addListener('mouseup', () => {
      if (!isDrawing) return
      if (rectangle.current) {
        // Make editable/draggable now that drawing is done
        rectangle.current.setOptions({ editable: true, draggable: true })
        rectangle.current.addListener('bounds_changed', () => {
          if (rectangle.current) updateBBoxFromRectangle(rectangle.current)
        })
        updateBBoxFromRectangle(rectangle.current)
      }
      isDrawing = false
      startPoint = null
      // Exit draw mode after completing a box
      drawModeRef.current = false
      setDrawMode(false)
    })

    // Keep drawModeRef in sync with React state
    const syncDrawMode = (active: boolean) => {
      drawModeRef.current = active
    }

    // Expose sync function via map data so the drawMode effect can call it
    ;(map as unknown as { _syncDrawMode?: (v: boolean) => void })._syncDrawMode = syncDrawMode

    return () => {
      google.maps.event.removeListener(mousedownListener)
      google.maps.event.removeListener(mousemoveListener)
      google.maps.event.removeListener(mouseupListener)
    }
  }, [mapsReady, updateBBoxFromRectangle])

  // Keep the map's internal drawModeRef in sync with React drawMode state
  useEffect(() => {
    if (!mapInstance.current) return
    const map = mapInstance.current as unknown as { _syncDrawMode?: (v: boolean) => void }
    map._syncDrawMode?.(drawMode)
  }, [drawMode])

  // Sync map type
  useEffect(() => {
    if (mapInstance.current) {
      mapInstance.current.setMapTypeId(mapType)
    }
  }, [mapType])

  if (noKey) {
    return (
      <div className="rounded-xl border border-dashed border-cv-border bg-cv-card flex flex-col items-center justify-center h-[260px] gap-3">
        <MapPin className="w-8 h-8 text-cv-text-dim" />
        <p className="text-sm text-cv-text-secondary text-center max-w-xs">
          Add your Google Maps API key in <code className="text-cv-primary">.env</code> to enable
          the interactive map picker.
        </p>
        <p className="text-xs text-cv-text-dim">VITE_GOOGLE_MAPS_API_KEY=your_key_here</p>
      </div>
    )
  }

  if (!mapsReady) {
    return (
      <div className="rounded-xl border border-cv-border bg-cv-card h-[260px] md:h-[380px] skeleton" />
    )
  }

  return (
    <div className="rounded-xl overflow-hidden border border-cv-border">
      {/* Search bar */}
      <div className="px-3 py-2 bg-cv-card border-b border-cv-border">
        <input
          ref={searchRef}
          type="text"
          placeholder="Search for a region…"
          className="w-full bg-cv-surface border border-cv-border rounded-lg px-3 py-2 text-sm text-cv-text-primary placeholder:text-cv-text-dim focus:outline-none focus:border-cv-primary transition"
        />
      </div>

      {/* Map */}
      <div className="relative">
        <div ref={mapRef} className="w-full h-[260px] md:h-[380px]" />

        {/* Draw / Pan mode toggle — top left */}
        <div className="absolute top-3 left-3 flex rounded-lg overflow-hidden border border-cv-border shadow-card">
          <button
            onClick={() => setDrawMode(false)}
            title="Pan mode"
            className={`px-2.5 py-1.5 text-xs font-medium flex items-center gap-1.5 transition ${
              !drawMode
                ? 'bg-cv-primary-muted text-cv-primary'
                : 'bg-cv-card text-cv-text-secondary hover:bg-cv-card-hover'
            }`}
          >
            <Hand className="w-3.5 h-3.5" />
            Pan
          </button>
          <button
            onClick={() => setDrawMode(true)}
            title="Draw bounding box"
            className={`px-2.5 py-1.5 text-xs font-medium flex items-center gap-1.5 transition ${
              drawMode
                ? 'bg-cv-primary-muted text-cv-primary'
                : 'bg-cv-card text-cv-text-secondary hover:bg-cv-card-hover'
            }`}
          >
            <Pencil className="w-3.5 h-3.5" />
            Draw
          </button>
        </div>

        {/* Map type toggle — top right */}
        <div className="absolute top-3 right-3 flex rounded-lg overflow-hidden border border-cv-border shadow-card">
          {(['satellite', 'roadmap', 'hybrid'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setMapType(t)}
              className={`px-2.5 py-1.5 text-xs font-medium capitalize transition ${
                mapType === t
                  ? 'bg-cv-primary-muted text-cv-primary'
                  : 'bg-cv-card text-cv-text-secondary hover:bg-cv-card-hover'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Instruction overlay */}
        {drawMode && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-cv-surface/90 border border-cv-primary/40 rounded-lg px-3 py-1.5 text-xs text-cv-primary backdrop-blur-sm pointer-events-none">
            Click and drag to draw a bounding box
          </div>
        )}
        {!drawMode && !value && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-cv-surface/90 border border-cv-border rounded-lg px-3 py-1.5 text-xs text-cv-text-secondary backdrop-blur-sm pointer-events-none">
            Switch to Draw mode to select a region
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between gap-3 px-3 py-2 bg-cv-card border-t border-cv-border flex-wrap">
        {/* Coordinate chips */}
        {value ? (
          <div className="flex flex-wrap gap-1.5 text-xs font-mono">
            {['minLon', 'minLat', 'maxLon', 'maxLat'].map((label, i) => (
              <span
                key={label}
                className="px-2 py-1 rounded bg-cv-surface border border-cv-border text-cv-text-secondary"
                title={label}
              >
                {value[i]?.toFixed(4)}
              </span>
            ))}
          </div>
        ) : (
          <span className="text-xs text-cv-text-dim">No region selected</span>
        )}

        <div className="flex gap-2 ml-auto">
          <button
            onClick={useCurrentView}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-cv-surface border border-cv-border text-cv-text-secondary hover:text-cv-text-primary hover:border-cv-border-strong transition"
          >
            <Maximize2 className="w-3 h-3" />
            Use current view
          </button>
          {value && (
            <button
              onClick={clearRectangle}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-cv-surface border border-red-800/40 text-red-400 hover:bg-red-950/40 transition"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
