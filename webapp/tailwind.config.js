/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        agentProfiling: '#3b82f6',
        agentStatistical: '#8b5cf6',
        agentVisualization: '#10b981',
        agentTransformation: '#f59e0b',
        agentSupervisor: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
      },
    },
  },
  plugins: [require('@headlessui/tailwindcss')],
}
