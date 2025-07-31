import React from 'react';
import { MetricCard } from './MetricCard';

interface Metric {
  title: string;
  value: string | number;
  trend?: string;
  icon?: any;
  color?: string;
}

interface MetricsGridProps {
  metrics: Metric[];
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
      {metrics.map((metric, i) => (
        <MetricCard key={i} {...metric} />
      ))}
    </div>
  );
}
