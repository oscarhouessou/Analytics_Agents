import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import ProfilingPage from './pages/profiling'
import Landing from './pages/Landing'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/home" element={<Home />} />
        <Route path="/profiling" element={<ProfilingPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
