import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useResult } from '../context/ResultContext';
import { uploadFloorPlan } from '../services/api';
import { UploadCloud, FileImage, Sparkles, CheckCircle2, AlertCircle, HelpCircle } from 'lucide-react';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { setResult, setImageSrc } = useResult();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setError('');
    } else {
      setError('Please select a valid image file (PNG or JPG)');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Create Object URL for preview
      const objectUrl = URL.createObjectURL(file);
      const data = await uploadFloorPlan(file);
      setImageSrc(objectUrl);
      setResult(data);
      navigate(`/analysis/${data.job_id}`);
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.error || 
        err.message || 
        'Could not communicate with the evaluation server. Please ensure the backend is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-500">
      {/* Intro Header */}
      <div className="text-center space-y-4">
        <div className="inline-flex items-center space-x-2 px-3 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-500 rounded-full text-xs font-semibold uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Instant Architectural Intelligence</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight">
          Evaluate Your Floor Plans In <span className="bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">Seconds</span>
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto text-base sm:text-lg">
          Upload a 2D floor plan to analyze room layout efficiency, assess movement accessibility, and receive styling ideas from our AI.
        </p>
      </div>

      {/* Main Upload Box */}
      <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 sm:p-8 backdrop-blur-xl shadow-2xl relative overflow-hidden group">
        {/* Glow behind container */}
        <div className="absolute top-0 right-0 w-32 h-32 bg-amber-600/5 rounded-full blur-2xl group-hover:bg-amber-600/10 transition-all duration-500 pointer-events-none"></div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="relative">
            <label className="block text-sm font-semibold text-slate-300 mb-2">
              Floor Plan Image File
            </label>
            <div className="mt-1 flex flex-col items-center justify-center px-6 py-12 border-2 border-dashed border-slate-800 hover:border-amber-500/60 bg-slate-950/40 rounded-xl cursor-pointer transition-all duration-300 group">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="sr-only"
                id="file-upload"
                disabled={loading}
              />
              <label htmlFor="file-upload" className="cursor-pointer w-full text-center">
                <div className="space-y-4">
                  <div className="w-14 h-14 mx-auto rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 group-hover:text-amber-400 group-hover:border-amber-500/30 group-hover:shadow-lg group-hover:shadow-amber-500/5 transition-all duration-300">
                    <UploadCloud className="w-7 h-7" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-200">
                      {file ? (
                        <span className="text-amber-400 font-semibold flex items-center justify-center gap-1.5">
                          <FileImage className="w-4 h-4" /> {file.name}
                        </span>
                      ) : (
                        'Click to upload or drag & drop'
                      )}
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      PNG, JPG, or JPEG up to 50MB
                    </p>
                  </div>
                </div>
              </label>
            </div>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl flex items-start space-x-2 text-sm">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !file}
            className={`w-full py-4 px-6 rounded-xl font-bold text-sm tracking-wide transition-all duration-300 flex items-center justify-center space-x-2 ${
              loading 
                ? 'bg-amber-600/40 text-slate-900 cursor-not-allowed shadow-none' 
                : file
                  ? 'bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-slate-950 shadow-lg shadow-amber-500/10 hover:shadow-xl hover:shadow-amber-500/20 transform hover:-translate-y-0.5'
                  : 'bg-slate-800 text-slate-500 cursor-not-allowed'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-slate-950" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Analyzing Layout & Space...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Evaluate Plan Architecture</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Info Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
        <div className="bg-slate-900/30 border border-slate-900 rounded-xl p-5 flex items-start space-x-3.5">
          <div className="p-2.5 bg-slate-900 border border-slate-800 rounded-lg text-amber-500">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-200">Space Utilization</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Detects walls, partitions, and calculates usable vs. wasted square footage.
            </p>
          </div>
        </div>

        <div className="bg-slate-900/30 border border-slate-900 rounded-xl p-5 flex items-start space-x-3.5">
          <div className="p-2.5 bg-slate-900 border border-slate-800 rounded-lg text-amber-500">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-200">Movement Flow</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Employs graph-based A* routing to map accessibility between all spaces.
            </p>
          </div>
        </div>

        <div className="bg-slate-900/30 border border-slate-900 rounded-xl p-5 flex items-start space-x-3.5">
          <div className="p-2.5 bg-slate-900 border border-slate-800 rounded-lg text-amber-500">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-200">Adjacency Scoring</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Checks room placement compatibility using heuristic architectural rules.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;