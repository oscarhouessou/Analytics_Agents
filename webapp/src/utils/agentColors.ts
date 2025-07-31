// Palette de couleurs des agents Analytics Pro
export const agentColors = {
  profiling: '#3b82f6',      // Bleu - Data Profiling
  statistical: '#8b5cf6',    // Violet - Analyses Statistiques  
  visualization: '#10b981',  // Vert - Visualisations
  transformation: '#f59e0b', // Orange - Transformations
  supervisor: '#ef4444'      // Rouge - Agent Superviseur
} as const;

export type AgentType = keyof typeof agentColors;
