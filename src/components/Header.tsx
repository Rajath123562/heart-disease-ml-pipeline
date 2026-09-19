import React, { useRef } from 'react';
import {
  Activity,
  Upload,
  Database,
  Sun,
  Moon,
  Tv,
  HelpCircle,
  AlertTriangle,
  Sparkles
} from 'lucide-react';
import { DatasetMeta } from '../types';

interface HeaderProps {
  meta?: DatasetMeta;
  isDarkMode?: boolean;
  theme?: string;
  activeTab?: string;
  setActiveTab?: (tab: any) => void;
  onToggleTheme?: () => void;
  onUploadCSV?: (csvText: string, filename: string) => void;
  onSelectDatasetType?: (type: 'uci_authentic' | 'synthetic') => void;
  onOpenAbout?: () => void;
  onOpenPresentation?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  meta,
  isDarkMode,
  theme,
  activeTab: _activeTab,
  setActiveTab: _setActiveTab,
  onToggleTheme,
  onUploadCSV,
  onSelectDatasetType,
  onOpenAbout,
  onOpenPresentation
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const darkMode = isDarkMode ?? theme === 'dark';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      const content = event.target?.result as string;
      if (content && onUploadCSV) {
        onUploadCSV(content, file.name);
      }
    };
    reader.readAsText(file);
    // Reset so same file can be re-uploaded if desired
    e.target.value = '';
  };

  return (
    <header className="sticky top-0 z-30 border-b backdrop-blur-md bg-white/90 dark:bg-slate-900/90 border-slate-200 dark:border-slate-800 transition-colors">
      {/* Synthetic Warning Banner */}
      {meta?.isSynthetic && (
        <div className="bg-amber-500/15 border-b border-amber-500/30 px-4 py-2 text-xs text-amber-800 dark:text-amber-300 flex items-center justify-between">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
            <span>
              <strong>SYNTHETIC DEMO DATASET ACTIVE:</strong> This dataset (~300 records) was generated algorithmically for pipeline verification and does not represent real patient medical records.
            </span>
          </div>
          <button
            onClick={() => onSelectDatasetType?.('uci_authentic')}
            className="underline font-semibold hover:text-amber-900 dark:hover:text-amber-100 ml-4 shrink-0"
          >
            Switch to Authentic UCI Cleveland Dataset
          </button>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Left: Branding & Title */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-rose-600 to-red-500 flex items-center justify-center text-white shadow-md shadow-rose-500/20 shrink-0">
            <Activity className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-slate-900 dark:text-white text-base sm:text-lg leading-tight">
                Heart Disease Prediction
              </h1>
              <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded-full bg-rose-100 text-rose-700 dark:bg-rose-950/70 dark:text-rose-400 border border-rose-300 dark:border-rose-800">
                ML Pipeline
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 hidden sm:block">
              Machine Learning Capstone • Zero-Dependency TypeScript Core
            </p>
          </div>
        </div>

        {/* Right: Actions & Tools */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Dataset Selector / Upload */}
          <div className="flex items-center rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-100/70 dark:bg-slate-800/70 p-1 text-xs">
            <button
              onClick={() => onSelectDatasetType?.('uci_authentic')}
              className={`px-2.5 py-1 rounded font-medium transition-all ${
                meta?.source === 'uci_authentic'
                  ? 'bg-white dark:bg-slate-700 text-rose-600 dark:text-rose-400 shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
              title="Standard 303-row UCI Cleveland Clinic Dataset"
            >
              <span className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5" />
                <span className="hidden md:inline">UCI Cleveland</span>
              </span>
            </button>

            <button
              onClick={() => onSelectDatasetType?.('synthetic')}
              className={`px-2.5 py-1 rounded font-medium transition-all ${
                meta?.source === 'synthetic'
                  ? 'bg-white dark:bg-slate-700 text-amber-600 dark:text-amber-400 shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
              title="Algorithmically generated 300-row demo data"
            >
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Synthetic Demo</span>
              </span>
            </button>
          </div>

          {/* Upload CSV hidden input & button */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-slate-700 transition"
            title="Upload Kaggle heart.csv or heart_disease_uci.csv"
          >
            <Upload className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Upload CSV</span>
          </button>

          {/* Presentation Mode Quick Button */}
          {onOpenPresentation && (
            <button
              onClick={onOpenPresentation}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-rose-600 hover:bg-rose-700 text-white shadow-sm transition"
              title="Open Slide Deck Presentation Mode"
            >
              <Tv className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Slides</span>
            </button>
          )}

          {/* About Modal */}
          {onOpenAbout && (
            <button
              onClick={onOpenAbout}
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              title="About Capstone & Rubric Alignment"
            >
              <HelpCircle className="w-5 h-5" />
            </button>
          )}

          {/* Theme Toggle */}
          {onToggleTheme && (
            <button
              onClick={onToggleTheme}
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
