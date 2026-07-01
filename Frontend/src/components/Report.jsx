import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useResult } from '../context/ResultContext';
import { generateReport } from '../services/api';
import {
  FileText, ArrowLeft, Printer, AlertTriangle, Sparkles,
  LayoutGrid, Loader2, BookOpen, TrendingDown, CheckCircle2, XCircle
} from 'lucide-react';

// Render **bold** and newlines from AI report text
const RenderReport = ({ text }) => {
  if (!text) return null;
  return (
    <div className="space-y-1.5 text-sm leading-relaxed text-slate-300 print:text-slate-700">
      {text.split('\n').map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-2" />;

        const isBullet = line.trim().startsWith('- ') || line.trim().startsWith('* ');
        const content = isBullet ? line.trim().substring(2) : line;

        // Parse **bold**
        const boldRegex = /\*\*(.*?)\*\*/g;
        const parts = [];
        let last = 0, match;
        while ((match = boldRegex.exec(content)) !== null) {
          if (match.index > last) parts.push(content.substring(last, match.index));
          parts.push(<strong key={match.index} className="text-white font-bold print:text-black">{match[1]}</strong>);
          last = boldRegex.lastIndex;
        }
        if (last < content.length) parts.push(content.substring(last));
        const rendered = parts.length > 0 ? parts : content;

        if (isBullet) {
          return (
            <div key={i} className="flex items-start gap-2 ml-2">
              <span className="text-amber-500 mt-0.5 flex-shrink-0">•</span>
              <span>{rendered}</span>
            </div>
          );
        }
        return <p key={i}>{rendered}</p>;
      })}
    </div>
  );
};

const Report = () => {
  const { result } = useResult();
  const navigate = useNavigate();
  const [aiReport, setAiReport] = useState('');
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState('');

  const { rooms = [], metrics = {}, suggestions = [], job_id } = result || {};
  const overallScore = metrics.overall_quality || 0;
  const spaceScore = metrics.space_utilization?.space_score || 0;
  const adjScore = metrics.adjacency?.adjacency_score || 0;
  const accScore = metrics.accessibility?.accessibility_score || 0;

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 font-semibold animate-pulse">No analysis results available. Upload a floor plan first.</p>
      </div>
    );
  }

  const fetchAiReport = async () => {
    setReportLoading(true);
    setReportError('');
    try {
      const data = await generateReport(result);
      setAiReport(data.report);
    } catch (err) {
      setReportError('Failed to generate report. Please ensure the backend is running.');
    } finally {
      setReportLoading(false);
    }
  };

  useEffect(() => {
    if (result && !aiReport) {
      fetchAiReport();
    }
  }, [result]);

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-emerald-400';
    if (score >= 60) return 'text-amber-400';
    return 'text-rose-400';
  };

  const getBarColor = (score) => {
    if (score >= 80) return 'bg-emerald-500';
    if (score >= 60) return 'bg-amber-500';
    return 'bg-rose-500';
  };

  const getVerdict = (score) => {
    if (score >= 80) return { label: 'Good', icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' };
    if (score >= 60) return { label: 'Average', icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20' };
    return { label: 'Poor', icon: XCircle, color: 'text-rose-400', bg: 'bg-rose-500/10 border-rose-500/20' };
  };

  const verdict = getVerdict(overallScore);
  const VerdictIcon = verdict.icon;

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500 print:bg-white print:text-black">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5 print:border-slate-300">
        <div>
          <h1 className="text-3xl font-black text-white flex items-center gap-2 print:text-black">
            <FileText className="w-7 h-7 text-amber-500" />
            <span>Comprehensive Evaluation Report</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1 print:text-slate-600">
            Automated architectural diagnostics · Job: <span className="font-mono text-slate-300 print:text-black">{job_id}</span>
          </p>
        </div>
        <div className="flex items-center gap-3 print:hidden">
          <button
            onClick={() => navigate(`/analysis/${job_id}`)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 rounded-xl text-xs font-bold transition-all"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Analysis
          </button>
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-700 text-slate-950 rounded-xl text-xs font-extrabold transition-all shadow-lg shadow-amber-500/10"
          >
            <Printer className="w-4 h-4" />
            Print Report
          </button>
        </div>
      </div>

      {/* Verdict Banner */}
      <div className={`flex items-center gap-4 p-5 rounded-2xl border ${verdict.bg} print:border-slate-300`}>
        <VerdictIcon className={`w-10 h-10 ${verdict.color} flex-shrink-0`} />
        <div>
          <p className={`text-xl font-black ${verdict.color}`}>
            Overall Rating: {verdict.label} ({overallScore}/100)
          </p>
          <p className="text-slate-400 text-sm mt-0.5 print:text-slate-600">
            {overallScore >= 80
              ? 'This floor plan meets most architectural quality standards.'
              : overallScore >= 60
              ? 'This floor plan has notable issues that should be addressed before construction.'
              : 'This floor plan has significant design problems that would affect livability and functionality.'}
          </p>
        </div>
      </div>

      {/* Score Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Overall Quality', score: overallScore, sub: 'Composite' },
          { label: 'Space Utilization', score: spaceScore, sub: 'Efficiency' },
          { label: 'Movement Flow', score: accScore, sub: 'Accessibility' },
          { label: 'Room Adjacency', score: adjScore, sub: 'Compatibility' },
        ].map(({ label, score, sub }) => (
          <div key={label} className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between print:border-slate-200 print:bg-slate-50">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</span>
            <div className="my-3">
              <span className={`text-4xl font-extrabold ${getScoreColor(score)}`}>{score}</span>
              <span className="text-[10px] text-slate-500 font-semibold uppercase block mt-1">{sub}</span>
            </div>
            <div className="w-full bg-slate-950 rounded-full h-1.5 print:bg-slate-200">
              <div className={`h-1.5 rounded-full transition-all duration-700 ${getBarColor(score)}`} style={{ width: `${score}%` }} />
            </div>
          </div>
        ))}
      </div>

      {/* AI Diagnostic Report */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5 print:border-slate-200 print:bg-transparent">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2 print:text-black">
            <BookOpen className="w-5 h-5 text-amber-500" />
            <span>AI Diagnostic Report</span>
          </h2>
          {!reportLoading && (
            <button
              onClick={fetchAiReport}
              className="text-xs text-amber-500 hover:text-amber-400 border border-amber-500/30 hover:border-amber-400/50 px-3 py-1 rounded-lg transition-all print:hidden"
            >
              Regenerate
            </button>
          )}
        </div>

        {reportLoading && (
          <div className="flex items-center gap-3 py-8 justify-center">
            <Loader2 className="w-5 h-5 text-amber-500 animate-spin" />
            <span className="text-slate-400 text-sm animate-pulse">Generating diagnostic report with AI...</span>
          </div>
        )}

        {reportError && !reportLoading && (
          <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl">
            <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
            <p className="text-sm text-rose-300">{reportError}</p>
          </div>
        )}

        {aiReport && !reportLoading && (
          <div className="border-l-2 border-amber-500/30 pl-5">
            <RenderReport text={aiReport} />
          </div>
        )}
      </div>

      {/* What Needs Improvement */}
      {overallScore < 80 && (
        <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 print:border-slate-200 print:bg-transparent">
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2 print:text-black">
            <TrendingDown className="w-5 h-5 text-rose-400" />
            <span>Issues Detected</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {spaceScore < 80 && (
              <div className="p-4 bg-rose-500/5 border border-rose-500/20 rounded-xl space-y-2">
                <p className="text-xs font-bold text-rose-400 uppercase tracking-wider">Space Utilization · {spaceScore}/100</p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {spaceScore < 50
                    ? 'More than half the floor area is wasted — walls, corridors, and dead zones are consuming usable space.'
                    : 'A significant portion of floor area is not being used efficiently by functional rooms.'}
                </p>
              </div>
            )}
            {adjScore < 80 && (
              <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-xl space-y-2">
                <p className="text-xs font-bold text-amber-400 uppercase tracking-wider">Room Adjacency · {adjScore}/100</p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {adjScore < 50
                    ? 'Rooms that should be near each other (kitchen-dining, bedroom-bathroom) are poorly positioned in this layout.'
                    : 'Some important room relationships are missing — certain rooms should share walls or be closer together.'}
                </p>
              </div>
            )}
            {accScore < 80 && (
              <div className="p-4 bg-purple-500/5 border border-purple-500/20 rounded-xl space-y-2">
                <p className="text-xs font-bold text-purple-400 uppercase tracking-wider">Movement Flow · {accScore}/100</p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {accScore < 50
                    ? 'Movement between rooms is severely restricted — occupants cannot easily walk from one room to another.'
                    : 'Some rooms have poor connectivity. Missing doors or corridors are reducing movement efficiency.'}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 print:border-slate-200 print:bg-transparent">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2 print:text-black">
          <Sparkles className="w-5 h-5 text-amber-500" />
          <span>Architectural Recommendations</span>
        </h2>
        {suggestions.length > 0 ? (
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {suggestions.map((s, i) => (
              <li key={i} className="flex items-start gap-3.5 p-4 bg-slate-950/40 border border-slate-800 rounded-xl print:bg-slate-50 print:border-slate-200">
                <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded-lg text-amber-500 flex-shrink-0">
                  <AlertTriangle className="w-4 h-4" />
                </div>
                <span className="text-xs text-slate-300 leading-relaxed print:text-slate-700">{s}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-slate-500">No layout recommendations triggered.</p>
        )}
      </div>

      {/* Room Breakdown Table */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 print:border-slate-200 print:bg-transparent">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2 print:text-black">
          <LayoutGrid className="w-5 h-5 text-amber-500" />
          <span>Room Breakdown</span>
        </h2>
        <div className="border border-slate-800 rounded-xl overflow-hidden print:border-slate-200">
          <table className="min-w-full divide-y divide-slate-800 print:divide-slate-200">
            <thead className="bg-slate-950/60 print:bg-slate-100">
              <tr>
                {['Room', 'Type', 'Area (px²)', 'Dimensions'].map(h => (
                  <th key={h} className="px-5 py-3.5 text-left text-[10px] font-bold text-slate-500 uppercase tracking-wider print:text-slate-600">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-900/60 bg-slate-950/20 print:bg-white print:divide-slate-200">
              {rooms.map((room) => {
                const colors = {
                  Bathroom: 'bg-red-500/10 text-red-400 border-red-500/20',
                  Kitchen: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
                  Bedroom: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
                  'Living room': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
                  'Dining room': 'bg-blue-500/10 text-blue-400 border-blue-500/20',
                };
                const cls = colors[room.type] || 'bg-slate-500/10 text-slate-400 border-slate-500/20';
                return (
                  <tr key={room.id} className="hover:bg-slate-900/20 transition-colors print:hover:bg-transparent">
                    <td className="px-5 py-3 text-xs font-mono font-bold text-slate-300 print:text-slate-800">#{room.id + 1}</td>
                    <td className="px-5 py-3">
                      <span className={`inline-block px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${cls}`}>{room.type}</span>
                    </td>
                    <td className="px-5 py-3 text-xs text-slate-400 print:text-slate-700">{room.area?.toLocaleString() ?? 'N/A'}</td>
                    <td className="px-5 py-3 text-xs text-slate-400 print:text-slate-700">
                      {room.bounding_box ? `${room.bounding_box.width} × ${room.bounding_box.height}` : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Report;