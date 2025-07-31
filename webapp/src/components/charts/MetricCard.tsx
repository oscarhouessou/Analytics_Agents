import { Card } from '../ui/Card';
import type { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  trend?: string;
  icon?: LucideIcon;
  color?: string;
}

export function MetricCard({ title, value, trend, icon: Icon, color }: MetricCardProps) {
  return (
    <Card className="p-4 flex flex-col items-start gap-2 min-w-[120px]">
      <div className="flex items-center gap-2">
        {Icon && <Icon className="w-5 h-5" style={{ color: color || '#3b82f6' }} />}
        <span className="font-semibold text-gray-700" style={{ color }}>{title}</span>
      </div>
      <div className="text-2xl font-bold" style={{ color }}>{value}</div>
      {trend && <div className="text-xs text-gray-400">{trend}</div>}
    </Card>
  );
}
