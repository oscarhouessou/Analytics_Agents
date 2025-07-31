import { useAnalyticsStore } from '../stores/analyticsStore'
import { DataPreview } from '../components/upload/DataPreview'
import { MetricsGrid } from '../components/charts/MetricsGrid'
import { ChartCard } from '../components/charts/ChartCard'
import { HistogramChart } from '../components/charts/HistogramChart'
import { useEffect, useRef } from 'react'
import { toast } from 'sonner'

export default function ProfilingPage() {
  const { datasetSummary, error } = useAnalyticsStore();
  const mainRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (error) toast.error(error);
  }, [error]);

  useEffect(() => {
    if (datasetSummary && mainRef.current) {
      mainRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [datasetSummary]);

  const metrics = datasetSummary ? [
    { title: 'Lignes', value: datasetSummary.n_rows, trend: '', color: '#3b82f6' },
    { title: 'Colonnes', value: datasetSummary.n_cols, trend: '', color: '#8b5cf6' },
    { title: 'Colonnes manquantes', value: Object.values(datasetSummary.missing as Record<string, number>).filter((v: number) => v > 0).length, trend: '', color: '#ef4444' },
  ] : [];

  // Score santé et insights automatiques (exemple statique, à remplacer par backend)
  const healthScore = datasetSummary ? 87 : null;
  const insights = datasetSummary ? [
    { icon: '💡', text: 'Votre colonne "Revenus" a une distribution intéressante : 80% des valeurs entre 30K-70K avec quelques outliers.' },
    { icon: '🔥', text: '"Date_Achat" montre un pic en décembre : Saisonnalité forte détectée.' },
    { icon: '⚠️', text: '"Email" contient 23 doublons potentiels : Nettoyage recommandé.' },
  ] : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#e0e7ff] via-[#f0f9ff] to-[#f8fafc] flex flex-col items-center pt-10 px-2">
      <h2 className="text-3xl font-black tracking-tight mb-4 text-gray-900 drop-shadow-lg">
        <span className="bg-gradient-to-r from-blue-600 via-fuchsia-500 to-orange-400 bg-clip-text text-transparent">Profiling & Aperçu</span>
      </h2>
      <div ref={mainRef} className="rounded-3xl bg-white/80 shadow-2xl backdrop-blur-lg p-8 border border-fuchsia-100 flex flex-col gap-8 w-full max-w-3xl">
        {!datasetSummary ? (
          <div className="h-32 flex items-center justify-center text-fuchsia-400">Aucun fichier importé</div>
        ) : (
          <>
            {/* Santé des données */}
            <div className="flex flex-row items-center gap-6 mb-4">
              <div className="flex flex-col items-center bg-gradient-to-br from-blue-100 to-fuchsia-100 rounded-2xl px-6 py-4 shadow">
                <div className="text-3xl">🟢</div>
                <div className="font-bold text-blue-700 text-lg">Santé des Données</div>
                <div className="text-blue-500 text-2xl font-black">{healthScore}/100</div>
                <div className="text-blue-400 text-sm">95% complétude • 2 anomalies • 15 colonnes</div>
              </div>
              <div className="flex-1 flex flex-col gap-2">
                <MetricsGrid metrics={metrics} />
              </div>
            </div>
            {/* Aperçu & types */}
            <div className="mt-2">
              <DataPreview data={datasetSummary.sample} columns={datasetSummary.columns} />
            </div>
            {/* Visualisation par défaut */}
            {datasetSummary.defaultViz && datasetSummary.defaultViz.type === 'histogram' && (
              <div className="mt-6">
                <ChartCard title={`Histogramme: ${datasetSummary.defaultViz.column}`} chart={
                  <HistogramChart bins={datasetSummary.defaultViz.bins} counts={datasetSummary.defaultViz.counts} />
                } />
              </div>
            )}
            {/* Insights automatiques */}
            <div className="mt-8">
              <div className="font-bold text-blue-700 text-lg mb-2">🎯 Découvertes Intéressantes</div>
              <ul className="space-y-2">
                {insights.map((ins, i) => (
                  <li key={i} className="flex items-start gap-2 text-blue-600 text-base">
                    <span className="text-2xl">{ins.icon}</span>
                    <span>{ins.text}</span>
                  </li>
                ))}
              </ul>
            </div>
            {/* Call to action chat */}
            <div className="mt-8 flex flex-col items-center">
              <div className="bg-gradient-to-r from-blue-100 to-fuchsia-100 rounded-xl px-6 py-4 shadow flex flex-col items-center">
                <div className="font-bold text-blue-700 mb-2">Vous voulez creuser plus loin ?</div>
                <button className="bg-blue-600 text-white px-6 py-3 rounded-full font-bold shadow hover:bg-blue-700 transition" onClick={() => window.location.href='/home'}>
                  💬 Posez vos questions à l'IA
                </button>
                <div className="text-blue-400 text-sm mt-2">Ou explorez les visualisations : 📊 Distributions | 🔗 Corrélations</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
