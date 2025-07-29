import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Upload from './pages/Upload';
import Chat from './pages/Chat';
import Dashboard from './pages/Dashboard';
import About from './pages/About';

const nav = [
  { to: '/', label: 'Accueil' },
  { to: '/upload', label: 'Upload' },
  { to: '/chat', label: 'Chat' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/about', label: 'À propos' },
];

const AppRouter: React.FC = () => (
  <BrowserRouter>
    <nav className="w-full flex gap-4 px-6 py-3 bg-white shadow items-center sticky top-0 z-10">
      <span className="font-bold text-blue-600">AnalyticsPro</span>
      {nav.map(n => (
        <Link key={n.to} to={n.to} className="text-blue-700 hover:underline text-sm">
          {n.label}
        </Link>
      ))}
    </nav>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/chat" element={<Chat />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/about" element={<About />} />
    </Routes>
  </BrowserRouter>
);

export default AppRouter;
