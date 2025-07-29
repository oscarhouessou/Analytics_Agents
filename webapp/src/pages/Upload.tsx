import React from 'react';
import { DropZone } from '../components/upload/DropZone';
import { DataPreview } from '../components/upload/DataPreview';
import { useAnalyticsStore } from '../stores/analyticsStore';

const Upload: React.FC = () => {
  const { currentDataset, uploadDataset, isAnalyzing } = useAnalyticsStore();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-white">
      <h2 className="text-2xl font-bold mb-4">📁 Importer un fichier de données</h2>
      <div className="w-full max-w-xl bg-white rounded-xl shadow p-8 flex flex-col gap-6">
        <DropZone onUpload={uploadDataset} preview={true} />
        <div className="text-center text-sm text-gray-400">Taille max : 100 Mo</div>
        {currentDataset && (
          <>
            <div className="font-semibold text-blue-700 mt-4">Aperçu : {currentDataset.name}</div>
            <DataPreview data={currentDataset.data} columns={currentDataset.columns} loading={isAnalyzing} />
          </>
        )}
      </div>
    </div>
  );
};

export default Upload;
