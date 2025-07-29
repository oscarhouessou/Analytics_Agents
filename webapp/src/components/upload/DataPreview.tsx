import React from 'react';

interface DataPreviewProps {
  data: any[][];
  columns: string[];
  loading?: boolean;
}

export const DataPreview: React.FC<DataPreviewProps> = ({ data, columns, loading }) => {
  if (loading) return <div className="text-center text-blue-500">Chargement...</div>;
  if (!data.length) return <div className="text-center text-gray-400">Aperçu non disponible</div>;
  return (
    <div className="overflow-x-auto border rounded-lg mt-4">
      <table className="min-w-full text-sm">
        <thead className="bg-blue-50">
          <tr>
            {columns.map(col => (
              <th key={col} className="px-3 py-2 font-semibold text-left text-blue-700">{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="even:bg-blue-50">
              {row.map((cell, j) => (
                <td key={j} className="px-3 py-1 whitespace-nowrap">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
