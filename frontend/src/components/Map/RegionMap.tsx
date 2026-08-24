/**
 * RegionMap Component
 * 
 * Interactive map for displaying and selecting geographic regions.
 * Uses a simple SVG-based world map for lightweight implementation.
 * Can be upgraded to Leaflet for more advanced features.
 */

import { useState, useRef, useCallback, useEffect } from 'react'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

// Type for bounding box
export type BBox = [number, number, number, number] // [minLon, minLat, maxLon, maxLat]

export interface RegionMapProps {
  bbox?: BBox
  onBBoxChange?: (bbox: BBox) => void
  highlightRegions?: Array<{
    bbox: BBox
    color?: string
    label?: string
  }>
  className?: string
  interactive?: boolean
  showGrid?: boolean
  showLabels?: boolean
}

// Simple world map projection (Plate Carrée)
const MAP_WIDTH = 360
const MAP_HEIGHT = 180

// Convert lat/lon to SVG coordinates
function lonToX(lon: number): number {
  return ((lon + 180) / 360) * MAP_WIDTH
}

function latToY(lat: number): number {
  return ((90 - lat) / 180) * MAP_HEIGHT
}

// Convert SVG coordinates to lat/lon
function xToLon(x: number): number {
  return (x / MAP_WIDTH) * 360 - 180
}

function yToLat(y: number): number {
  return 90 - (y / MAP_HEIGHT) * 180
}

// Simplified world continent outlines (rough approximation)
const CONTINENTS = [
  // North America
  'M50,30 L70,25 L90,30 L100,40 L110,50 L100,70 L80,70 L60,60 L50,50 Z',
  // South America
  'M80,80 L90,75 L100,85 L95,110 L85,120 L75,115 L70,100 Z',
  // Europe
  'M165,30 L185,25 L200,30 L195,40 L180,45 L165,40 Z',
  // Africa
  'M165,55 L190,50 L200,60 L195,90 L175,105 L160,95 L155,70 Z',
  // Asia
  'M200,25 L260,20 L280,35 L290,50 L270,60 L240,55 L220,50 L200,40 Z',
  // Australia
  'M260,85 L280,80 L295,90 L285,105 L265,105 L255,95 Z',
  // Antarctica
  'M100,160 L260,160 L260,175 L100,175 Z',
]

// Major cities for reference
const CITIES = [
  { name: 'New York', lon: -74, lat: 40.7 },
  { name: 'London', lon: -0.1, lat: 51.5 },
  { name: 'Tokyo', lon: 139.7, lat: 35.7 },
  { name: 'Sydney', lon: 151.2, lat: -33.9 },
  { name: 'São Paulo', lon: -46.6, lat: -23.5 },
]

export function RegionMap({
  bbox,
  onBBoxChange,
  highlightRegions = [],
  className,
  interactive = true,
  showGrid = true,
  showLabels = true,
}: RegionMapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [currentBBox, setCurrentBBox] = useState<BBox | null>(bbox || null)

  // Update internal state when prop changes
  useEffect(() => {
    if (bbox) {
      setCurrentBBox(bbox)
    }
  }, [bbox])

  const getSvgCoords = useCallback((event: React.MouseEvent): { x: number; y: number } => {
    if (!svgRef.current) return { x: 0, y: 0 }
    
    const rect = svgRef.current.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * MAP_WIDTH
    const y = ((event.clientY - rect.top) / rect.height) * MAP_HEIGHT
    
    return { x, y }
  }, [])

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (!interactive || !onBBoxChange) return
    
    const coords = getSvgCoords(event)
    setIsDragging(true)
    setDragStart(coords)
    setCurrentBBox(null)
  }, [interactive, onBBoxChange, getSvgCoords])

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (!isDragging || !dragStart) return
    
    const coords = getSvgCoords(event)
    const minX = Math.min(dragStart.x, coords.x)
    const maxX = Math.max(dragStart.x, coords.x)
    const minY = Math.min(dragStart.y, coords.y)
    const maxY = Math.max(dragStart.y, coords.y)
    
    const minLon = xToLon(minX)
    const maxLon = xToLon(maxX)
    const maxLat = yToLat(minY)
    const minLat = yToLat(maxY)
    
    setCurrentBBox([minLon, minLat, maxLon, maxLat])
  }, [isDragging, dragStart, getSvgCoords])

  const handleMouseUp = useCallback(() => {
    if (isDragging && currentBBox && onBBoxChange) {
      onBBoxChange(currentBBox)
    }
    setIsDragging(false)
    setDragStart(null)
  }, [isDragging, currentBBox, onBBoxChange])

  const renderBBox = (b: BBox, color: string, label?: string, key?: string) => {
    const x = lonToX(b[0])
    const y = latToY(b[3])
    const width = lonToX(b[2]) - x
    const height = latToY(b[1]) - y

    return (
      <g key={key}>
        <rect
          x={x}
          y={y}
          width={Math.max(width, 2)}
          height={Math.max(height, 2)}
          fill={color}
          fillOpacity={0.3}
          stroke={color}
          strokeWidth={0.5}
          strokeOpacity={0.8}
        />
        {label && (
          <text
            x={x + width / 2}
            y={y + height / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            fill={color}
            fontSize={3}
            fontWeight="bold"
          >
            {label}
          </text>
        )}
      </g>
    )
  }

  return (
    <div className={cx('relative', className)}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
        className={cx(
          'w-full h-auto rounded-lg border border-base-700 bg-base-900',
          interactive && 'cursor-crosshair',
        )}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Ocean background */}
        <rect x={0} y={0} width={MAP_WIDTH} height={MAP_HEIGHT} fill="#dbeafe" />
        
        {/* Grid lines */}
        {showGrid && (
          <g stroke="#b6c6dd" strokeWidth={0.2}>
            {/* Latitude lines */}
            {[-60, -30, 0, 30, 60].map((lat) => (
              <line
                key={`lat-${lat}`}
                x1={0}
                y1={latToY(lat)}
                x2={MAP_WIDTH}
                y2={latToY(lat)}
              />
            ))}
            {/* Longitude lines */}
            {[-120, -60, 0, 60, 120].map((lon) => (
              <line
                key={`lon-${lon}`}
                x1={lonToX(lon)}
                y1={0}
                x2={lonToX(lon)}
                y2={MAP_HEIGHT}
              />
            ))}
          </g>
        )}
        
        {/* Continents */}
        <g fill="#d7e0d5" stroke="#aeb9ac" strokeWidth={0.3}>
          {CONTINENTS.map((path, i) => (
            <path key={i} d={path} />
          ))}
        </g>
        
        {/* City markers */}
        {showLabels && (
          <g>
            {CITIES.map((city) => (
              <g key={city.name}>
                <circle
                  cx={lonToX(city.lon)}
                  cy={latToY(city.lat)}
                  r={1}
                  fill="#6b7280"
                />
              </g>
            ))}
          </g>
        )}
        
        {/* Highlight regions */}
        {highlightRegions.map((region, i) =>
          renderBBox(region.bbox, region.color || '#22c55e', region.label, `region-${i}`)
        )}
        
        {/* Current/selected bbox */}
        {currentBBox && renderBBox(currentBBox, '#3b82f6', undefined, 'current')}
        
        {/* Drag selection preview */}
        {isDragging && dragStart && currentBBox && (
          <rect
            x={lonToX(currentBBox[0])}
            y={latToY(currentBBox[3])}
            width={lonToX(currentBBox[2]) - lonToX(currentBBox[0])}
            height={latToY(currentBBox[1]) - latToY(currentBBox[3])}
            fill="#3b82f6"
            fillOpacity={0.2}
            stroke="#3b82f6"
            strokeWidth={1}
            strokeDasharray="2,2"
          />
        )}
      </svg>
      
      {/* BBox display */}
      {currentBBox && (
        <div className="absolute bottom-2 left-2 px-2 py-1 bg-base-800/90 rounded text-xs text-base-200 font-mono">
          [{currentBBox.map((v) => v.toFixed(2)).join(', ')}]
        </div>
      )}
      
      {interactive && (
        <div className="absolute top-2 right-2 px-2 py-1 bg-base-800/90 rounded text-xs text-base-400">
          Drag to select region
        </div>
      )}
    </div>
  )
}

// Mini map for displaying a single region
export interface MiniMapProps {
  bbox: BBox
  className?: string
  variant?: 'forest' | 'ice' | 'flood'
}

const variantColors = {
  forest: '#22c55e',
  ice: '#3b82f6',
  flood: '#f59e0b',
}

export function MiniMap({ bbox, className, variant = 'forest' }: MiniMapProps) {
  const color = variantColors[variant]
  
  return (
    <RegionMap
      bbox={bbox}
      highlightRegions={[{ bbox, color }]}
      className={cx('max-w-[200px]', className)}
      interactive={false}
      showGrid={false}
      showLabels={false}
    />
  )
}

// Preset regions selector
export interface PresetRegion {
  name: string
  bbox: BBox
  description?: string
}

const PRESET_REGIONS: PresetRegion[] = [
  { name: 'Amazon Basin', bbox: [-73, -15, -45, 5], description: 'Primary deforestation monitoring' },
  { name: 'Congo Basin', bbox: [8, -5, 30, 10], description: 'Central African rainforest' },
  { name: 'Borneo', bbox: [108, -4, 120, 8], description: 'Southeast Asian rainforest' },
  { name: 'Arctic Ocean', bbox: [-180, 66.5, 180, 90], description: 'Sea ice monitoring' },
  { name: 'Greenland', bbox: [-73, 60, -12, 84], description: 'Ice sheet monitoring' },
  { name: 'Bangladesh', bbox: [88, 20, 93, 27], description: 'Flood-prone region' },
]

interface PresetSelectorProps {
  onSelect: (region: PresetRegion) => void
  className?: string
}

export function PresetRegionSelector({ onSelect, className }: PresetSelectorProps) {
  return (
    <div className={cx('space-y-2', className)}>
      <div className="text-sm font-medium text-base-200 mb-2">Preset Regions</div>
      <div className="grid grid-cols-2 gap-2">
        {PRESET_REGIONS.map((region) => (
          <button
            key={region.name}
            onClick={() => onSelect(region)}
            className="text-left px-3 py-2 rounded-lg border border-base-700 bg-base-800/50 hover:bg-base-800 transition"
          >
            <div className="text-sm font-medium text-base-100">{region.name}</div>
            {region.description && (
              <div className="text-xs text-base-400 mt-0.5">{region.description}</div>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
