import { useNavigate } from 'react-router-dom'

export default function DemoModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 animate-fade-in">
      <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-lg w-full flex flex-col items-center relative">
        <button className="absolute top-4 right-4 text-blue-400 hover:text-blue-700 text-2xl" onClick={onClose}>&times;</button>
        <h2 className="text-2xl font-bold text-blue-700 mb-2 text-center">Démo Interactive</h2>
        <div className="w-full h-48 bg-gradient-to-r from-blue-100 to-fuchsia-100 rounded-xl flex items-center justify-center text-blue-300 text-lg mb-4">
          [Vidéo ou animation à intégrer ici]
        </div>
        <ul className="text-blue-600 text-base mb-4 list-disc list-inside">
          <li>Upload d'un fichier CSV</li>
          <li>Profiling automatique et visualisation</li>
          <li>Chat IA avec suggestions intelligentes</li>
        </ul>
        <button className="mt-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-fuchsia-500 text-white rounded-full font-bold text-lg shadow-xl hover:scale-105 transition-transform" onClick={onClose}>
          Fermer la démo
        </button>
      </div>
    </div>
  )
}
