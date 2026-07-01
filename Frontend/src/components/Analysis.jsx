import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useResult } from '../context/ResultContext';
import { ChevronRight, ArrowLeft, Layers, ShieldAlert, Navigation, Columns, LayoutGrid, CheckCircle } from 'lucide-react';

const Analysis = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { result, imageSrc } = useResult();

  useEffect(() => {
    if (!result) {
      navigate('/');
    }
  }, [result, navigate]);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 font-semibold animate-pulse">Loading analysis results...</p>
      </div>
    );
  }

  const { rooms = [], metrics = {}, suggestions = [] } = result;
  const overallScore = metrics.overall_quality || 0;

  // Render SVG overlays on top of the actual uploaded image
  const renderVisualization = () => {
    const width = result?.visualization?.image_shape?.width || 600;
    const height = result?.visualization?.image_shape?.height || 400;

    return (
      <div className="relative bg-slate-950/80 rounded-2xl border border-slate-800/80 overflow-hidden shadow-inner group flex items-center justify-center min-h-[350px]">
        {/* Blueprint Grid Pattern behind image */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-30"></div>

        {imageSrc ? (
          <div className="relative max-w-full max-h-[450px] overflow-hidden flex items-center justify-center">
            {/* The actual uploaded floor plan */}
            <img 
              src={imageSrc} 
              alt="Floor Plan" 
              className="max-w-full max-h-[450px] object-contain rounded-lg opacity-85 select-none"
            />
            {/* SVG overlays mapped on top */}
            <svg 
              className="absolute inset-0 w-full h-full pointer-events-none" 
              viewBox={`0 0 ${width} ${height}`} 
              preserveAspectRatio="xMidYMid meet"
            >
              {rooms.map((room, index) => {
                // Determine dimensions or fallback
                const bbox = room.bounding_box || { x: 80 + index * 100, y: 80 + index * 70, width: 120, height: 90 };
                
                // Design soft translucent colors matching specific room types
                let fillColor = 'rgba(59, 130, 246, 0.15)'; // Default blue
                let strokeColor = '#3b82f6';
                
                if (room.type === 'Bathroom') {
                  fillColor = 'rgba(239, 68, 68, 0.15)'; // Red/Pink
                  strokeColor = '#ef4444';
                } else if (room.type === 'Kitchen') {
                  fillColor = 'rgba(245, 158, 11, 0.15)'; // Amber
                  strokeColor = '#f59e0b';
                } else if (room.type === 'Bedroom') {
                  fillColor = 'rgba(168, 85, 247, 0.15)'; // Purple
                  strokeColor = '#a855f7';
                } else if (room.type === 'Living room') {
                  fillColor = 'rgba(34, 197, 94, 0.15)'; // Green
                  strokeColor = '#22c55e';
                }

                return (
                  <g key={room.id} className="transition-all duration-300">
                    <rect
                      x={bbox.x}
                      y={bbox.y}
                      width={bbox.width}
                      height={bbox.height}
                      fill={fillColor}
                      stroke={strokeColor}
                      strokeWidth={2}
                      rx={4}
                      className="hover:fill-opacity-35 cursor-crosshair transition-all duration-200"
                    />
                    {/* Centered Room Label */}
                    {bbox.width > 30 && bbox.height > 20 && (
                      <>
                        <rect
                          x={bbox.x + bbox.width / 2 - Math.min(75, bbox.width * 0.8) / 2}
                          y={bbox.y + bbox.height / 2 - 9}
                          width={Math.min(75, bbox.width * 0.8)}
                          height={18}
                          fill="rgba(15, 23, 42, 0.92)"
                          rx={4}
                          stroke={strokeColor}
                          strokeWidth={1}
                        />
                        <text
                          x={bbox.x + bbox.width / 2}
                          y={bbox.y + bbox.height / 2 + 3}
                          fill="#f8fafc"
                          fontSize={Math.max(7, Math.min(10, bbox.width / 8))}
                          fontWeight="extrabold"
                          textAnchor="middle"
                        >
                          {room.type}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        ) : (
          <div className="text-center py-20 relative z-10 space-y-3 text-slate-500">
            <LayoutGrid className="w-12 h-12 mx-auto stroke-1" />
            <p className="text-sm font-semibold">Visualizing Generated Room Boundaries</p>
          </div>
        )}
      </div>
    );
  };

  const getScoreColorClass = (score) => {
    if (score >= 80) return 'text-emerald-400';
    if (score >= 60) return 'text-amber-400';
    return 'text-rose-400';
  };

  const getScoreBgClass = (score) => {
    if (score >= 80) return 'bg-emerald-500';
    if (score >= 60) return 'bg-amber-500';
    return 'bg-rose-500';
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Title Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-white flex items-center gap-2">
            <Layers className="w-7 h-7 text-amber-500" />
            <span>Floor Plan Evaluation</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Job ID: <span className="text-slate-300 font-mono">{jobId}</span>
          </p>
        </div>
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 rounded-xl text-xs font-bold transition-all duration-200"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Upload Another</span>
        </button>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Side: Visualization (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900/40 border border-slate-900 rounded-2xl p-5 backdrop-blur-xl shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-900 pb-3.5">
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <Columns className="w-5 h-5 text-amber-500" />
                <span>Detection & Partition Overlay</span>
              </h2>
              <span className="text-xs px-2.5 py-1 bg-amber-500/10 text-amber-500 border border-amber-500/20 rounded-full font-semibold">
                {rooms.length} Spaces Identified
              </span>
            </div>
            {renderVisualization()}
          </div>
        </div>

        {/* Right Side: Score Summary (1 col) */}
        <div className="space-y-6">
          
          {/* Overall Score */}
          <div className="bg-slate-900/40 border border-slate-900 rounded-2xl p-6 backdrop-blur-xl shadow-xl text-center space-y-4">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Overall Design Score</h2>
            <div className="relative inline-flex items-center justify-center">
              {/* Radial Score Gauge */}
              <div className="w-36 h-36 rounded-full border-4 border-slate-800 flex flex-col items-center justify-center">
                <span className={`text-4xl font-extrabold ${getScoreColorClass(overallScore)}`}>
                  {overallScore}
                </span>
                <span className="text-[10px] text-slate-500 font-semibold uppercase mt-0.5">out of 100</span>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed max-w-xs mx-auto">
              Based on composite calculations for floor space allocation, movement flow convenience, and layout adjacencies.
            </p>
          </div>

          {/* Detailed Score Breakdown */}
          <div className="bg-slate-900/40 border border-slate-900 rounded-2xl p-6 backdrop-blur-xl shadow-xl space-y-4">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider pb-1">Efficiency Breakdown</h3>
            <div className="space-y-4.5">
              {/* Space Utilization */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold">
                  <span className="text-slate-300">Space Utilization</span>
                  <span className={getScoreColorClass(metrics.space_utilization?.space_score || 0)}>
                    {metrics.space_utilization?.space_score || 0}%
                  </span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-1.5 border border-slate-900 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${getScoreBgClass(metrics.space_utilization?.space_score || 0)}`}
                    style={{ width: `${metrics.space_utilization?.space_score || 0}%` }}
                  ></div>
                </div>
              </div>

              {/* Adjacency */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold">
                  <span className="text-slate-300">Adjacency Scoring</span>
                  <span className={getScoreColorClass(metrics.adjacency?.adjacency_score || 0)}>
                    {metrics.adjacency?.adjacency_score || 0}%
                  </span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-1.5 border border-slate-900 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${getScoreBgClass(metrics.adjacency?.adjacency_score || 0)}`}
                    style={{ width: `${metrics.adjacency?.adjacency_score || 0}%` }}
                  ></div>
                </div>
              </div>

              {/* Accessibility */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold">
                  <span className="text-slate-300">Movement Flow (Access)</span>
                  <span className={getScoreColorClass(metrics.accessibility?.accessibility_score || 0)}>
                    {metrics.accessibility?.accessibility_score || 0}%
                  </span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-1.5 border border-slate-900 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${getScoreBgClass(metrics.accessibility?.accessibility_score || 0)}`}
                    style={{ width: `${metrics.accessibility?.accessibility_score || 0}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Action Trigger Card */}
          <div className="bg-slate-900/40 border border-slate-900 rounded-2xl p-6 backdrop-blur-xl shadow-xl flex flex-col justify-between">
            <div className="mb-4">
              <h3 className="text-sm font-bold text-slate-200">View Comprehensive Report</h3>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                Check individual room sizes, layout recommendations, and print or export a full PDF of your design score.
              </p>
            </div>
            <button
              onClick={() => navigate('/report')}
              className="w-full py-3.5 px-5 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-slate-950 rounded-xl font-bold text-xs tracking-wide transition-all duration-300 flex items-center justify-center space-x-1.5 shadow-md shadow-amber-500/5 hover:shadow-lg"
            >
              <span>View Detailed Report</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Analysis;