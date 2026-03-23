import { useState, useEffect, useRef } from 'react'

const CACHE_KEY = 'cv_geocode_cache'

function loadCache(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(CACHE_KEY) ?? '{}')
  } catch {
    return {}
  }
}

function saveCache(cache: Record<string, string>) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(cache))
  } catch {}
}

export function useGeocoding(bbox: number[] | null | undefined, apiKey: string) {
  const [regionName, setRegionName] = useState<string | null>(null)
  const cacheRef = useRef<Record<string, string>>(loadCache())

  useEffect(() => {
    if (!bbox || bbox.length !== 4 || !apiKey || apiKey === 'YOUR_GOOGLE_MAPS_API_KEY_HERE') return

    const lat = (bbox[1] + bbox[3]) / 2
    const lon = (bbox[0] + bbox[2]) / 2
    const cacheKey = `${lat.toFixed(3)},${lon.toFixed(3)}`

    if (cacheRef.current[cacheKey]) {
      setRegionName(cacheRef.current[cacheKey])
      return
    }

    const url = `https://maps.googleapis.com/maps/api/geocode/json?latlng=${lat},${lon}&key=${apiKey}`
    fetch(url)
      .then((r) => r.json())
      .then((data) => {
        const result = data.results?.[0]
        if (!result) return
        // Find locality or administrative area
        const locality = result.address_components?.find((c: { types: string[] }) =>
          c.types.includes('locality'),
        )
        const admin = result.address_components?.find((c: { types: string[] }) =>
          c.types.includes('administrative_area_level_1'),
        )
        const country = result.address_components?.find((c: { types: string[] }) =>
          c.types.includes('country'),
        )
        const name = [locality?.short_name, admin?.short_name, country?.short_name]
          .filter(Boolean)
          .join(', ')
        if (name) {
          cacheRef.current[cacheKey] = name
          saveCache(cacheRef.current)
          setRegionName(name)
        }
      })
      .catch(() => {})
  }, [bbox, apiKey])

  return regionName
}
