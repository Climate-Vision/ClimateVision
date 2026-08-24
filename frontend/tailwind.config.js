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
        // Legacy colors — remapped to a light theme.
        // Lower numbers = darker (used for text), higher numbers = lighter (used for backgrounds).
        base: {
          950: '#ffffff',
          900: '#f8faf9',
          800: '#eef2f0',
          700: '#dde5e0',
          600: '#c2cdc7',
          500: '#94a3ac',
          400: '#64748b',
          300: '#475569',
          200: '#334155',
          100: '#1e293b',
        },
        brand: {
          400: '#4ade80',
          500: '#22C55E',
          600: '#16A34A',
          700: '#15803d',
          900: '#14532d',
        },
        ocean: {
          400: '#22d3ee',
          500: '#06B6D4',
          600: '#0891b2',
          700: '#0e7490',
        },
        amber: {
          400: '#fbbf24',
          500: '#F59E0B',
          600: '#d97706',
          700: '#b45309',
        },
        danger: {
          400: '#f87171',
          500: '#EF4444',
          600: '#dc2626',
          700: '#b91c1c',
          900: '#7f1d1d',
        },
        // Design system tokens — light theme
        cv: {
          bg: '#ffffff',
          surface: '#f6f8f7',
          card: '#ffffff',
          'card-hover': '#f2f5f3',
          border: '#e4e9e6',
          'border-strong': '#cdd6d0',
          primary: '#16a34a',
          'primary-hover': '#15803d',
          'primary-muted': '#dcfce7',
          danger: '#dc2626',
          'danger-muted': '#fee2e2',
          warning: '#d97706',
          'warning-muted': '#fef3c7',
          info: '#2563eb',
          'text-primary': '#0f172a',
          'text-secondary': '#475569',
          'text-muted': '#64748b',
          'text-dim': '#94a3b8',
        },
      },
      boxShadow: {
        soft: '0 10px 30px rgba(15, 23, 42, 0.08)',
        card: '0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04)',
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
