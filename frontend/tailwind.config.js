/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: {
          950: '#071116',
          900: '#0B1B23',
          800: '#102634',
          200: '#D7E7EE',
          100: '#EEF6F9',
        },
        brand: {
          500: '#22C55E',
          600: '#16A34A',
        },
        ocean: {
          500: '#06B6D4',
        },
        amber: {
          500: '#F59E0B',
        },
        danger: {
          500: '#EF4444',
        },
      },
      boxShadow: {
        soft: '0 10px 30px rgba(2, 6, 23, 0.35)',
      },
    },
  },
  plugins: [],
}
