import { Card } from '../ui/Card';

interface ChartCardProps {
  title: string;
  chart: React.ReactNode;
  onExport?: () => void;
}

export function ChartCard({ title, chart, onExport }: ChartCardProps) {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-gray-700">{title}</span>
        {onExport && (
          <button onClick={onExport} className="text-gray-400 hover:text-gray-700" title="Exporter">
            <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 5v14m7-7H5"/></svg>
          </button>
        )}
      </div>
      <div>{chart}</div>
    </Card>
  );
}
