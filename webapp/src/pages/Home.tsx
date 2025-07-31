

import { Header } from '../components/layout/Header'
import { DropZone } from '../components/upload/DropZone'
import ChatPanel from '../components/chat/ChatPanel'
import { useAnalyticsStore } from '../stores/analyticsStore'
import { useState, useRef, useEffect } from 'react'
import { toast } from 'sonner'









export default function Home() {
  const { uploadDataset, datasetSummary, error } = useAnalyticsStore();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showProfilingTransition, setShowProfilingTransition] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [demoExample, setDemoExample] = useState<string | null>(null);
  const mainRef = useRef<HTMLDivElement>(null);

  // Simule l'upload et l'analyse d'un exemple de démo
  useEffect(() => {
    if (demoExample && !file && !uploading) {
      setUploading(true);
      setUploadProgress(0);
      const fakeFile = { name: `Exemple ${demoExample}.csv` } as File;
      setFile(fakeFile);
      (async () => {
        for (let i = 1; i <= 10; i++) {
          setUploadProgress(i * 10);
          await new Promise(res => setTimeout(res, 60));
        }
        await uploadDataset(fakeFile, 'Donne-moi un résumé du dataset');
        setUploading(false);
        setShowProfilingTransition(true);
        setTimeout(() => setShowProfilingTransition(false), 1200);
      })();
    }
    // eslint-disable-next-line
  }, [demoExample]);




  // Notification d'erreur ou succès
  useEffect(() => {
    if (error) toast.error(error);
  }, [error]);

  // Scroll vers résultats après upload ou analyse
  useEffect(() => {
    if (datasetSummary && mainRef.current) {
      mainRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [datasetSummary]);

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-[#e0e7ff] via-[#f0f9ff] to-[#f8fafc] flex flex-col">
      <Header />
      <div className="flex-1 flex flex-col justify-center items-center px-2">
        {showOnboarding ? (
          <div className="flex flex-col items-center justify-center gap-8 animate-fade-in w-full max-w-xl py-16">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-400 to-fuchsia-400 flex items-center justify-center text-5xl text-white shadow-xl mb-2">✨</div>
            <h1 className="text-3xl md:text-4xl font-extrabold text-blue-700 text-center drop-shadow-lg">Bienvenue sur votre assistant d'analyse de données</h1>
            <p className="text-lg text-blue-500 text-center max-w-lg">Importez un fichier CSV, obtenez un profiling automatique, puis discutez avec l'IA pour explorer vos données et générer des insights en toute simplicité.</p>
            <div className="flex flex-col gap-4 w-full mt-4">
              <button className="px-8 py-3 rounded-full bg-gradient-to-r from-blue-500 to-fuchsia-500 text-white font-bold text-lg shadow-lg hover:scale-105 transition-transform" onClick={() => setShowOnboarding(false)}>
                Commencer avec mes données
              </button>
              <div className="text-center text-blue-400 font-semibold mt-2">Ou essayez une démo :</div>
              <div className="flex flex-row gap-3 justify-center">
                <button className="px-4 py-2 rounded-full bg-blue-100 text-blue-700 font-semibold hover:bg-blue-200 transition" onClick={() => { setShowOnboarding(false); setDemoExample('RH'); }}>Exemple RH</button>
                <button className="px-4 py-2 rounded-full bg-fuchsia-100 text-fuchsia-700 font-semibold hover:bg-fuchsia-200 transition" onClick={() => { setShowOnboarding(false); setDemoExample('Ventes'); }}>Exemple Ventes</button>
                <button className="px-4 py-2 rounded-full bg-emerald-100 text-emerald-700 font-semibold hover:bg-emerald-200 transition" onClick={() => { setShowOnboarding(false); setDemoExample('Finance'); }}>Exemple Finance</button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-6 animate-fade-in w-full">
            {!file && (
              <>
                <h1 className="text-2xl md:text-3xl font-extrabold text-blue-700 text-center drop-shadow-lg">Importez votre fichier CSV</h1>
                <DropZone onUpload={async f => {
                  setUploading(true);
                  setUploadProgress(0);
                  setFile(f);
                  for (let i = 1; i <= 10; i++) {
                    setUploadProgress(i * 10);
                    await new Promise(res => setTimeout(res, 60));
                  }
                  await uploadDataset(f, 'Donne-moi un résumé du dataset');
                  setUploading(false);
                  setShowProfilingTransition(true);
                  setTimeout(() => setShowProfilingTransition(false), 1200);
                }} preview={false} />
                {uploading && (
                  <div className="w-64 mt-4">
                    <div className="h-2 bg-blue-100 rounded-full overflow-hidden">
                      <div className="h-2 bg-gradient-to-r from-blue-400 to-fuchsia-400 rounded-full transition-all duration-200" style={{ width: `${uploadProgress}%` }} />
                    </div>
                    <div className="text-xs text-blue-400 mt-1 text-center">Upload en cours... {uploadProgress}%</div>
                  </div>
                )}
                {showProfilingTransition && (
                  <div className="mt-6 animate-fade-in flex flex-col items-center">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-400 to-fuchsia-400 flex items-center justify-center text-4xl text-white shadow-xl mb-2">📊</div>
                    <div className="font-bold text-blue-700 text-lg">Profiling automatique...</div>
                    <div className="text-blue-400 text-sm">Lecture des données, détection des types, calcul des métriques...</div>
                  </div>
                )}
              </>
            )}
            {/* Profiling automatique après upload */}
            {file && datasetSummary && !showProfilingTransition && (
              <div className="w-full max-w-3xl bg-white rounded-xl shadow-lg p-6 mt-4 animate-fade-in">
                <h2 className="text-xl font-bold text-blue-700 mb-2">Profiling du fichier : <span className="font-mono text-blue-500">{file.name}</span></h2>
                <div className="text-sm text-gray-700 whitespace-pre-line mb-4">{datasetSummary}</div>
                <button className="ml-auto block text-xs text-red-400 hover:underline mb-2" onClick={() => setFile(null)} aria-label="Retirer le fichier importé">Retirer le fichier</button>
              </div>
            )}
            {/* Chat IA avec suggestions intelligentes */}
            {file && datasetSummary && !showProfilingTransition && (
              <div className="w-full max-w-3xl mt-6 animate-fade-in">
                <ChatPanel />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}


