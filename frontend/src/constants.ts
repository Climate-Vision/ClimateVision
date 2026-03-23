/**
 * Application constants for ClimateVision frontend
 */

// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const API_TIMEOUT = 30000;

// Map Configuration
export const DEFAULT_MAP_CENTER: [number, number] = [9.0820, 8.6753]; // Nigeria center
export const DEFAULT_MAP_ZOOM = 6;
export const MAX_BBOX_AREA_KM2 = 10000;

// Analysis Types
export const ANALYSIS_TYPES = {
  DEFORESTATION: 'deforestation',
  LAND_COVER: 'land_cover',
  CARBON: 'carbon_estimation',
} as const;

// Polling Configuration
export const RUN_POLL_INTERVAL_MS = 5000;
export const MAX_POLL_ATTEMPTS = 120; // 10 minutes max

// UI Constants
export const TOAST_DURATION_MS = 5000;
export const DEBOUNCE_DELAY_MS = 300;

// Status Colors
export const STATUS_COLORS = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
} as const;
