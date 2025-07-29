import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './AppRouter';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Impossible de trouver l'élément #root dans le HTML.");
}
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
