import React, { useState, useEffect } from 'react';
import {
  Presentation,
  ChevronLeft,
  ChevronRight,
  FileText,
  Maximize2,
  Minimize2,
  CheckCircle2,
  Activity,
  Layers,
  Sparkles,
  Award,
  AlertTriangle
} from 'lucide-react';
import {
  DatasetMeta,
  PreprocessingConfig,
  PreprocessingSummary,
  LogisticModelArtifact,
  DecisionTreeModelArtifact,
  ModelMetrics,
  CrossValidationSummary
} from '../types';
import { generatePresentationSlides } from '../report/presentationDeck';

interface PresentationTabProps {
  meta: DatasetMeta;
  config: PreprocessingConfig;
  summary: PreprocessingSummary | null;
  logisticModel: LogisticModelArtifact | null;
  treeModel: DecisionTreeModelArtifact | null;
  logisticMetrics: ModelMetrics | null;
  treeMetrics: ModelMetrics | null;
  logisticCV: CrossValidationSummary | null;
  treeCV: CrossValidationSummary | null;
  threshold: number;
}

export const PresentationTab: React.FC<PresentationTabProps> = ({
  meta,
  config,
  summary,
  logisticModel,
  treeModel,
  logisticMetrics,
  treeMetrics,
  logisticCV,
  treeCV,
  threshold
}) => {
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [showNotes, setShowNotes] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const slides = generatePresentationSlides(
    meta,
    config,
    summary,
    logisticModel,
    treeModel,
    logisticMetrics,
    treeMetrics,
    logisticCV,
    treeCV,
    threshold
  );

  const activeSlide = slides[currentSlideIndex] || slides[0];

  // Keyboard navigation for presentation mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'Space') {
        e.preventDefault();
        setCurrentSlideIndex(prev => Math.min(prev + 1, slides.length - 1));
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setCurrentSlideIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'p' || e.key === 'P') {
        setShowNotes(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [slides.length]);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Top Deck Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
        <div className="flex items-center gap-2">
          <Presentation className="w-5 h-5 text-rose-500" />
          <span className="font-bold text-sm text-slate-900 dark:text-white">
            Capstone Slide Deck ({currentSlideIndex + 1} of {slides.length})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Presenter Notes Toggle */}
          <button
            onClick={() => setShowNotes(!showNotes)}
            className={`px-3 py-1.5 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer ${
              showNotes
                ? 'bg-rose-50 dark:bg-rose-950/50 border-rose-300 dark:border-rose-800 text-rose-600 dark:text-rose-400'
                : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Speaker Notes (P)</span>
          </button>

          {/* Previous Slide */}
          <button
            onClick={() => setCurrentSlideIndex(prev => Math.max(prev - 1, 0))}
            disabled={currentSlideIndex === 0}
            className="p-2 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 transition cursor-pointer"
            title="Previous Slide (Left Arrow)"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          {/* Next Slide */}
          <button
            onClick={() => setCurrentSlideIndex(prev => Math.min(prev + 1, slides.length - 1))}
            disabled={currentSlideIndex === slides.length - 1}
            className="p-2 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 transition cursor-pointer"
            title="Next Slide (Right Arrow or Space)"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Slide Stage */}
      <div className="relative aspect-[16/9] w-full max-w-5xl mx-auto bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 text-white rounded-2xl sm:rounded-3xl border border-slate-800 shadow-xl overflow-hidden flex flex-col justify-between p-8 sm:p-14">
        {/* Slide Header */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs uppercase tracking-widest text-rose-400 font-bold">
            <span>Machine Learning Capstone &bull; Slide {currentSlideIndex + 1}</span>
            <span className="text-slate-400 font-mono">Week 4 Deliverable</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight">
            {activeSlide.title}
          </h2>
          <p className="text-sm sm:text-lg text-slate-300 font-light">
            {activeSlide.subtitle}
          </p>
        </div>

        {/* Slide Body Bullets */}
        <div className="my-auto py-6 space-y-4 max-w-3xl">
          {activeSlide.bullets.map((bullet: string, idx: number) => (
            <div key={idx} className="flex items-start gap-3.5 text-sm sm:text-base text-slate-200 leading-relaxed">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shrink-0 mt-2" />
              <span>{bullet}</span>
            </div>
          ))}
        </div>

        {/* Slide Footer */}
        <div className="pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400 font-mono">
          <span>Heart Disease Pipeline: End-to-End ML</span>
          <span>{currentSlideIndex + 1} / {slides.length}</span>
        </div>
      </div>

      {/* Slide Thumbnails Drawer */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 max-w-5xl mx-auto">
        {slides.map((s: any, idx: number) => (
          <button
            key={s.id}
            onClick={() => setCurrentSlideIndex(idx)}
            className={`p-2.5 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between h-20 ${
              idx === currentSlideIndex
                ? 'bg-rose-50 dark:bg-rose-950/40 border-rose-500 ring-2 ring-rose-500/30'
                : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 hover:border-slate-300'
            }`}
          >
            <span className="text-[10px] font-bold text-slate-400">Slide {idx + 1}</span>
            <span className="text-xs font-semibold text-slate-900 dark:text-white line-clamp-2 leading-tight">
              {s.title}
            </span>
          </button>
        ))}
      </div>

      {/* Presenter Notes Drawer (Collapsible) */}
      {showNotes && (
        <div className="max-w-5xl mx-auto p-5 bg-amber-50 dark:bg-amber-950/30 border border-amber-500/40 rounded-2xl text-amber-900 dark:text-amber-200 text-xs sm:text-sm space-y-2 animate-fadeIn">
          <div className="font-bold flex items-center gap-1.5 uppercase text-xs tracking-wider text-amber-700 dark:text-amber-400">
            <FileText className="w-4 h-4" />
            <span>Speaker Presentation Notes</span>
          </div>
          <p className="leading-relaxed font-sans">
            {activeSlide.speakerNotes || activeSlide.notes || ''}
          </p>
        </div>
      )}
    </div>
  );
};
