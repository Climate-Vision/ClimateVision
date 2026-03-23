/**
 * Utility functions for ClimateVision frontend
 */

/**
 * Format a date to a human-readable string
 */
export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format a number with commas as thousands separator
 */
export function formatNumber(num: number, decimals = 0): string {
  return num.toLocaleString('en-GB', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format area in square kilometers
 */
export function formatArea(areaKm2: number): string {
  if (areaKm2 < 1) {
    return `${formatNumber(areaKm2 * 100, 2)} ha`;
  }
  return `${formatNumber(areaKm2, 2)} km²`;
}

/**
 * Calculate bounding box area in km²
 */
export function calculateBBoxArea(bbox: [number, number, number, number]): number {
  const [minLng, minLat, maxLng, maxLat] = bbox;
  const latDiff = Math.abs(maxLat - minLat);
  const lngDiff = Math.abs(maxLng - minLng);
  // Approximate conversion at equator
  const kmPerDegLat = 111;
  const kmPerDegLng = 111 * Math.cos((minLat + maxLat) / 2 * Math.PI / 180);
  return latDiff * kmPerDegLat * lngDiff * kmPerDegLng;
}

/**
 * Debounce a function
 */
export function debounce<T extends (...args: unknown[]) => void>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Clamp a number between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
