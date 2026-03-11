/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        // Legacy colors kept for existing components
        base: {
          950: '#071116',
          900: '#0B1B23',
          800: '#102634',
          700: '#1a3a4a',
          400: '#5a8a9f',
          300: '#8ab4c7',
          200: '#D7E7EE',
          100: '#EEF6F9',
        },
        brand: {
          400: '#4ade80',
          500: '#22C55E',
          600: '#16A34A',
          900: '#14532d',
        },
        ocean: {
          400: '#22d3ee',
          500: '#06B6D4',
          600: '#0891b2',
        },
        amber: {
          400: '#fbbf24',
          500: '#F59E0B',
          600: '#d97706',
        },
        danger: {
          400: '#f87171',
          500: '#EF4444',
          900: '#7f1d1d',
        },
        // Design system tokens
        cv: {
          bg: '#0a0f0d',
          surface: '#111a14',
          card: '#162019',
          'card-hover': '#1c2a1f',
          border: '#1f3024',
          'border-strong': '#2d4a33',
          primary: '#22c55e',
          'primary-hover': '#16a34a',
          'primary-muted': '#14532d',
          danger: '#ef4444',
          'danger-muted': '#7f1d1d',
          warning: '#f59e0b',
          'warning-muted': '#78350f',
          info: '#3b82f6',
          'text-primary': '#f0fdf4',
          'text-secondary': '#86efac',
          'text-muted': '#4ade80',
          'text-dim': '#374151',
        },
      },
      boxShadow: {
        soft: '0 10px 30px rgba(2, 6, 23, 0.35)',
        card: '0 4px 16px rgba(0,0,0,0.4)',
        glow: '0 0 20px rgba(34,197,94,0.15)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.15s ease-out',
        'slide-in': 'slideIn 0.2s ease-out',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
        slideIn: { from: { transform: 'translateX(100%)' }, to: { transform: 'translateX(0)' } },
      },
    },
  },
  plugins: [],
}
