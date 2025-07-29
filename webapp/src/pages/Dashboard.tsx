import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { useAnalyticsStore } from '../stores/analyticsStore';

const Dashboard: React.FC = () => {
  const { currentDataset } = useAnalyticsStore();
  // Simulé : à remplacer par des vraies métriques/visualisations API
  const metrics = [
    { title: 'Lignes', value: currentDataset?.data.length ?? 0, icon: '📄', trend: '+5%' },
    { title: 'Colonnes', value: currentDataset?.columns.length ?? 0, icon: '🧩', trend: 'stable' },
    { title: 'Taille', value: '1.2MB', icon: '💾', trend: '+2%' },
  ];
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-white">
      <h2 className="text-2xl font-bold mb-4">📈 Dashboard Analytique</h2>
      <div className="w-full max-w-5xl bg-white rounded-xl shadow p-8 flex flex-col gap-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {metrics.map((m, i) => (
            <Card key={i} className="flex flex-col items-center justify-center">
              <CardHeader>
                <CardTitle>{m.icon} {m.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-blue-700">{m.value}</div>
                <div className="text-sm text-gray-400">{m.trend}</div>
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48 flex items-center justify-center text-gray-400">[HistogramChart ici]</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Corrélations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48 flex items-center justify-center text-gray-400">[HeatmapChart ici]</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
