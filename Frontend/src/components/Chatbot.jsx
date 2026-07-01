import React, { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../services/api';
import { useResult } from '../context/ResultContext';
import { MessageSquare, X, Send, Sparkles, User, Bot, HelpCircle, LayoutGrid } from 'lucide-react';

const Chatbot = () => {
  const { result } = useResult();
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [history, setHistory] = useState([
    {
      role: 'model',
      content: result
        ? `Hello! I'm FloorGenie, your floor plan advisor. I can see you've uploaded a floor plan with an overall score of **${result.metrics?.overall_quality ?? 'N/A'}/100**. Ask me anything about it — why scores are low, how to improve specific rooms, or general design advice!`
        : "Hello! I'm FloorGenie, your floor plan advisor. Upload a floor plan first and I'll give you specific advice about your layout. Or ask me anything about flooring, room design, or space planning!"
    }
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Dynamic suggested questions based on whether analysis exists
  const suggestedQuestions = result
    ? [
        "Why is my floor plan score low?",
        "How can I improve room adjacency?",
        "What's wrong with my layout?",
        "How do I fix the movement flow?",
        "Which rooms should be next to each other?",
      ]
    : [
        "Hardwood vs. Laminate flooring?",
        "Best flooring for bathrooms?",
        "How to improve floor plan flow?",
        "Ideal room sizes for a 3-bedroom home?",
      ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, loading, isOpen]);

  // Reset greeting when analysis result changes
  useEffect(() => {
    if (result) {
      setHistory([{
        role: 'model',
        content: `Hello! I'm FloorGenie, your floor plan advisor. I can see you've uploaded a floor plan with an overall score of **${result.metrics?.overall_quality ?? 'N/A'}/100**.\n\nI have access to your full analysis — ${result.rooms?.length ?? 0} rooms detected, space score ${result.metrics?.space_utilization?.space_score ?? 'N/A'}, adjacency ${result.metrics?.adjacency?.adjacency_score ?? 'N/A'}, movement flow ${result.metrics?.accessibility?.accessibility_score ?? 'N/A'}.\n\nAsk me anything — why scores are low, how to improve the layout, or what to change first!`
      }]);
    }
  }, [result]);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!message.trim() || loading) return;

    const userMessage = message;
    setMessage('');
    setHistory(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await sendChatMessage(userMessage, history, result || null);
      setHistory(prev => [...prev, { role: 'model', content: response.reply }]);
    } catch (err) {
      console.error(err);
      setHistory(prev => [
        ...prev,
        { role: 'model', content: "Sorry, I encountered a connection issue. Please make sure the backend server is running and try again." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (question) => {
    setMessage(question);
    setTimeout(() => {
      document.getElementById('chatbot-input')?.focus();
    }, 50);
  };

  const formatText = (text) => {
    if (!text) return '';
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      let formattedLine = line;
      const isBullet = line.trim().startsWith('* ') || line.trim().startsWith('- ');
      if (isBullet) formattedLine = line.trim().substring(2);

      const boldRegex = /\*\*(.*?)\*\*/g;
      const parts = [];
      let lastIndex = 0;
      let match;
      while ((match = boldRegex.exec(formattedLine)) !== null) {
        if (match.index > lastIndex) parts.push(formattedLine.substring(lastIndex, match.index));
        parts.push(<strong key={match.index} className="font-bold text-amber-800 dark:text-amber-300">{match[1]}</strong>);
        lastIndex = boldRegex.lastIndex;
      }
      if (lastIndex < formattedLine.length) parts.push(formattedLine.substring(lastIndex));
      const content = parts.length > 0 ? parts : formattedLine;

      if (isBullet) return <li key={idx} className="ml-4 list-disc my-0.5 text-sm text-slate-700">{content}</li>;
      if (!line.trim()) return <div key={idx} className="h-1" />;
      return <p key={idx} className="my-0.5 text-sm leading-relaxed text-slate-700">{content}</p>;
    });
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 font-sans">
      {/* Floating Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="flex items-center justify-center w-14 h-14 bg-gradient-to-tr from-amber-700 to-amber-500 hover:from-amber-800 hover:to-amber-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 group relative"
        >
          <MessageSquare className="w-6 h-6 transition-transform duration-300 group-hover:rotate-6" />
          {result && (
            <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-emerald-500"></span>
            </span>
          )}
        </button>
      )}

      {/* Expanded Chat Dialog */}
      {isOpen && (
        <div className="w-[390px] h-[580px] bg-white border border-slate-200 rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-6 duration-300">
          {/* Header */}
          <div className="bg-gradient-to-r from-amber-800 to-amber-600 p-4 text-white flex justify-between items-center shadow-md flex-shrink-0">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 bg-amber-100 rounded-full flex items-center justify-center text-amber-800 shadow">
                <Sparkles className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <h3 className="font-semibold text-sm leading-tight">FloorGenie Advisor</h3>
                <div className="flex items-center space-x-1.5 mt-0.5">
                  <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
                  <span className="text-[10px] text-amber-100 font-medium uppercase tracking-wider">
                    {result ? 'Analyzing your floor plan' : 'Floor Plan & Design Expert'}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-amber-100 hover:text-white p-1 hover:bg-amber-900/30 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Floor plan context badge */}
          {result && (
            <div className="flex items-center gap-2 px-4 py-2 bg-emerald-50 border-b border-emerald-100 flex-shrink-0">
              <LayoutGrid className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
              <span className="text-[11px] text-emerald-700 font-semibold">
                Floor plan loaded · {result.rooms?.length ?? 0} rooms · Score: {result.metrics?.overall_quality ?? 'N/A'}/100
              </span>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 bg-slate-50 space-y-3">
            {history.map((msg, index) => {
              const isUser = msg.role === 'user';
              return (
                <div key={index} className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start gap-2`}>
                  {!isUser && (
                    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-amber-700 to-amber-500 text-white flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5">
                      <Bot className="w-4 h-4" />
                    </div>
                  )}
                  <div className={`max-w-[82%] rounded-2xl px-3.5 py-2.5 shadow-sm text-sm ${
                    isUser
                      ? 'bg-gradient-to-br from-amber-700 to-amber-600 text-white rounded-tr-none'
                      : 'bg-white text-slate-800 border border-slate-100 rounded-tl-none'
                  }`}>
                    {isUser
                      ? <p className="leading-relaxed text-sm">{msg.content}</p>
                      : <div className="space-y-0.5">{formatText(msg.content)}</div>
                    }
                  </div>
                  {isUser && (
                    <div className="w-7 h-7 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5">
                      <User className="w-4 h-4" />
                    </div>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className="flex justify-start items-start gap-2">
                <div className="w-7 h-7 rounded-full bg-gradient-to-br from-amber-700 to-amber-500 text-white flex items-center justify-center flex-shrink-0 shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="bg-white border border-slate-100 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
                  <div className="flex space-x-1.5 items-center">
                    <span className="w-2 h-2 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-2 h-2 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-2 h-2 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Suggestions */}
          {!loading && history.length <= 2 && (
            <div className="px-3 py-2 bg-white border-t border-slate-100 flex-shrink-0">
              <p className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1.5 flex items-center gap-1">
                <HelpCircle className="w-3 h-3 text-amber-500" /> Quick questions:
              </p>
              <div className="flex flex-wrap gap-1.5">
                {suggestedQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSuggestionClick(q)}
                    className="text-[11px] bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-200 rounded-full px-2.5 py-1 transition-all duration-200 font-medium"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="p-3 border-t border-slate-100 bg-white flex items-center gap-2 flex-shrink-0">
            <input
              id="chatbot-input"
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder={result ? "Ask about your floor plan..." : "Ask about flooring, layouts..."}
              className="flex-1 bg-slate-50 hover:bg-slate-100 focus:bg-white border border-slate-200 focus:border-amber-500 rounded-xl px-4 py-2 text-sm focus:outline-none transition-all duration-200"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !message.trim()}
              className="p-2.5 bg-gradient-to-br from-amber-700 to-amber-500 hover:from-amber-800 hover:to-amber-600 text-white rounded-xl disabled:opacity-40 transition-all duration-200 flex items-center justify-center shadow-md disabled:shadow-none"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
