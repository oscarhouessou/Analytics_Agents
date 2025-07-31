import { useNavigate } from 'react-router-dom'
import { useState } from 'react'
import DemoModal from '../components/landing/DemoModal'

export default function Landing() {
  const navigate = useNavigate();
  const [showDemo, setShowDemo] = useState(false);
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 via-fuchsia-50 to-orange-50">
      {/* Hero Section */}
      <section className="flex flex-col items-center justify-center flex-1 px-4 py-12">
        <h1 className="text-4xl md:text-5xl font-black text-blue-900 text-center mb-4 drop-shadow-xl">
          Transformez vos données en <span className="bg-gradient-to-r from-fuchsia-500 to-orange-400 bg-clip-text text-transparent">insights</span> en 3 minutes
        </h1>
        <p className="text-lg md:text-xl text-blue-500 text-center mb-8 max-w-2xl">
          La plateforme d'analyse qui rend la data conversationnelle, visuelle et accessible à tous.
        </p>
        {/* Demo Interactive Placeholder */}
        <div className="w-full max-w-xl mb-8 flex flex-col items-center">
          <div className="rounded-2xl shadow-lg bg-white/80 p-6 flex flex-col items-center animate-fade-in">
            <span className="text-2xl mb-2">🚀</span>
            <span className="font-semibold text-blue-700">Démo interactive (15s)</span>
            <div className="w-full h-32 bg-gradient-to-r from-blue-100 to-fuchsia-100 rounded-xl mt-2 flex items-center justify-center text-blue-300 text-lg">
              [Vidéo ou animation à intégrer ici]
            </div>
            <button className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-full font-semibold shadow hover:bg-blue-700" onClick={() => setShowDemo(true)}>
              Voir la démo
            </button>
          </div>
        </div>
        <button
          className="mt-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-fuchsia-500 text-white rounded-full font-bold text-lg shadow-xl hover:scale-105 transition-transform"
          onClick={() => navigate('/home')}
        >Commencer gratuitement</button>
        {showDemo && <DemoModal onClose={() => setShowDemo(false)} />}
      </section>
      {/* Micro-Onboarding */}
      <section className="w-full flex flex-col md:flex-row items-center justify-center gap-8 py-12 bg-white/70 border-t border-blue-100">
        <div className="flex flex-col items-center gap-2 max-w-xs">
          <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center text-3xl">📁</div>
          <div className="font-bold text-blue-700 text-lg">Uploadez vos données</div>
          <div className="text-blue-400 text-sm text-center">Glissez-déposez ou sélectionnez un fichier pour démarrer.</div>
        </div>
        <div className="flex flex-col items-center gap-2 max-w-xs">
          <div className="w-16 h-16 rounded-full bg-fuchsia-100 flex items-center justify-center text-3xl">📊</div>
          <div className="font-bold text-fuchsia-700 text-lg">Découvrez votre profil data</div>
          <div className="text-fuchsia-400 text-sm text-center">Visualisez la santé, les types et les insights clés de vos données.</div>
        </div>
        <div className="flex flex-col items-center gap-2 max-w-xs">
          <div className="w-16 h-16 rounded-full bg-orange-100 flex items-center justify-center text-3xl">💬</div>
          <div className="font-bold text-orange-700 text-lg">Dialoguez avec votre IA</div>
          <div className="text-orange-400 text-sm text-center">Posez vos questions, obtenez des analyses et des visualisations instantanées.</div>
        </div>
      </section>
      {/* Exemples de datasets */}
      <section className="w-full flex flex-col items-center py-8 bg-gradient-to-r from-blue-50 to-fuchsia-50">
        <div className="font-bold text-blue-700 mb-4 text-lg">Essayez avec un exemple :</div>
        <div className="flex flex-row gap-4 flex-wrap justify-center">
          <button
            className="bg-white border border-blue-100 rounded-xl px-6 py-3 shadow hover:bg-blue-50 transition"
            onClick={() => navigate('/home?example=ecommerce')}
          >🛍️ E-commerce</button>
          <button
            className="bg-white border border-fuchsia-100 rounded-xl px-6 py-3 shadow hover:bg-fuchsia-50 transition"
            onClick={() => navigate('/home?example=rh')}
          >👥 RH</button>
          <button
            className="bg-white border border-orange-100 rounded-xl px-6 py-3 shadow hover:bg-orange-50 transition"
            onClick={() => navigate('/home?example=finance')}
          >💰 Finance</button>
        </div>
      </section>
    </div>
  )
}
