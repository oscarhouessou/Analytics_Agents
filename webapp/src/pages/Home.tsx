import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => (
  <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-white">
    <h1 className="text-4xl font-bold mb-4">📊 AnalyticsPro WebApp</h1>
    <p className="text-lg text-gray-600 mb-8">Plateforme d'analyse de données multi-agents IA (React + FastAPI)</p>
    <div className="flex flex-col gap-4">
      <Link to="/upload" className="px-6 py-3 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition">Commencer l'analyse</Link>
      <Link to="/about" className="text-blue-500 underline">À propos</Link>
    </div>
  </div>
);

export default Home;
