import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

interface HistogramChartProps {
  bins: number[];
  counts: number[];
  color?: string;
}

export function HistogramChart({ bins, counts, color = '#3b82f6' }: HistogramChartProps) {
  if (!bins || !counts || bins.length < 2) return <div className="text-gray-400">Aucune donnée</div>;
  const data = counts.map((count, i) => ({
    name: `${bins[i].toFixed(1)} - ${bins[i + 1].toFixed(1)}`,
    value: count,
  }));
  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={50} />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
