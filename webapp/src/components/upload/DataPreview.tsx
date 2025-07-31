import React from 'react';

interface DataPreviewProps {
  data: Record<string, any>[];
  columns: string[];
  loading?: boolean;
  error?: string | null;
}

export function DataPreview({ data, columns, loading, error }: DataPreviewProps) {
  if (loading) return <div className="p-4 text-center text-gray-400">Chargement des données...</div>;
  if (error) return <div className="p-4 text-center text-red-500">{error}</div>;
  if (!data || data.length === 0) return <div className="p-4 text-center text-gray-400">Aperçu non disponible</div>;

  return (
    <div className="overflow-x-auto rounded-lg border mt-4">
      <table className="min-w-full text-xs">
        <thead className="bg-gray-100">
          <tr>
            {columns.map(col => (
              <th key={col} className="px-3 py-2 text-left font-semibold text-gray-700">{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row, i) => (
            <tr key={i} className="even:bg-gray-50">
              {columns.map(col => (
                <td key={col} className="px-3 py-2 whitespace-nowrap">{String(row[col] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-xs text-gray-400 p-2">Affichage des 10 premières lignes</div>
    </div>
  );
}
