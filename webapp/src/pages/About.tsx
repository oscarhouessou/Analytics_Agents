import React from 'react';

const About: React.FC = () => (
  <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-white">
    <h2 className="text-2xl font-bold mb-4">À propos d'AnalyticsPro</h2>
    <div className="w-full max-w-2xl bg-white rounded-xl shadow p-8 flex flex-col gap-6">
      <p className="text-gray-700">AnalyticsPro est une plateforme d'analyse de données assistée par IA multi-agents, combinant React, FastAPI, et des technologies modernes pour offrir une expérience d'analyse intuitive et puissante.</p>
      <ul className="list-disc list-inside text-gray-600">
        <li>Frontend : React 18, TypeScript, Vite, Tailwind CSS</li>
        <li>Backend : FastAPI, LangChain, OpenAI</li>
        <li>Visualisation : Recharts, D3.js, React Flow</li>
      </ul>
      <p className="text-gray-500 text-sm">© 2024 AnalyticsPro. Tous droits réservés.</p>
    </div>
  </div>
);

export default About;
