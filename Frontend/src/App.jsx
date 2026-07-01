import React from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import Upload from './components/Upload';
import Analysis from './components/Analysis';
import Report from './components/Report';
import Chatbot from './components/Chatbot';
import { Layers, Sparkles, FileText, UploadCloud } from 'lucide-react';
import './index.css';

function App() {
  const location = useLocation();

  const isSelected = (path) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-amber-700 selection:text-white">
      {/* Dynamic Background Glows */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute bottom-10 right-1/4 w-[400px] h-[400px] bg-amber-800/10 rounded-full blur-3xl pointer-events-none"></div>

      {/* Header / Nav */}
      <header className="sticky top-0 z-40 bg-slate-900/80 backdrop-blur-md border-b border-slate-800/60 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="w-10 h-10 bg-gradient-to-tr from-amber-600 to-amber-400 rounded-xl flex items-center justify-center shadow-lg shadow-amber-500/10 transition-transform group-hover:scale-105">
                <Layers className="w-5.5 h-5.5 text-slate-950 font-bold" />
              </div>
              <div>
                <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-white via-slate-200 to-amber-400 bg-clip-text text-transparent">
                  FloorPlan.AI
                </span>
                <span className="block text-[10px] text-amber-500 uppercase tracking-widest font-semibold">
                  Architectural Evaluation
                </span>
              </div>
            </Link>

            {/* Navigation links */}
            <nav className="flex space-x-1.5 bg-slate-950/80 p-1 border border-slate-800/60 rounded-xl">
              <Link
                to="/"
                className={`flex items-center space-x-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-300 ${
                  isSelected('/') 
                    ? 'bg-amber-600 text-slate-950 shadow-md shadow-amber-600/20' 
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
                }`}
              >
                <UploadCloud className="w-4 h-4" />
                <span>Upload</span>
              </Link>
              <Link
                to="/report"
                className={`flex items-center space-x-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-300 ${
                  isSelected('/report') 
                    ? 'bg-amber-600 text-slate-950 shadow-md shadow-amber-600/20' 
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
                }`}
              >
                <FileText className="w-4 h-4" />
                <span>Report</span>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto py-10 px-4 sm:px-6 lg:px-8 relative z-10">
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/analysis/:jobId" element={<Analysis />} />
          <Route path="/report" element={<Report />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/40 py-6 text-center text-xs text-slate-500 relative z-10">
        <div className="max-w-7xl mx-auto px-4">
          <p>© 2026 FloorPlan.AI. All rights reserved. Powered by FloorGenie Agent.</p>
        </div>
      </footer>

      {/* AI Chatbot Agent */}
      <Chatbot />
    </div>
  );
}

export default App;