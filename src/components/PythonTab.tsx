import React, { useState } from 'react';
import {
  Code,
  Copy,
  Download,
  BookOpen,
  Terminal,
  FileCode,
  CheckCircle2
} from 'lucide-react';
import {
  PreprocessingConfig,
  LogisticHyperparams,
  DecisionTreeHyperparams
} from '../types';
import {
  generatePythonScript,
  generateJupyterNotebook,
  downloadFile
} from '../ml/export';

interface PythonTabProps {
  config: PreprocessingConfig;
  logisticHyperparams: LogisticHyperparams;
  treeHyperparams: DecisionTreeHyperparams;
}

export const PythonTab: React.FC<PythonTabProps> = ({
  config,
  logisticHyperparams,
  treeHyperparams
}) => {
  const [copied, setCopied] = useState(false);

  const pythonCode = generatePythonScript(
    config,
    {
      lr: logisticHyperparams.learningRate,
      epochs: logisticHyperparams.epochs,
      reg: logisticHyperparams.regularization
    },
    {
      maxDepth: treeHyperparams.maxDepth,
      criterion: treeHyperparams.criterion
    }
  );

  const handleCopy = () => {
    navigator.clipboard.writeText(pythonCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadPy = () => {
    downloadFile('heart_disease_pipeline.py', pythonCode, 'text/x-python');
  };

  const handleDownloadIpynb = () => {
    const notebookJson = generateJupyterNotebook(
      config,
      {
        lr: logisticHyperparams.learningRate,
        epochs: logisticHyperparams.epochs,
        reg: logisticHyperparams.regularization
      },
      {
        maxDepth: treeHyperparams.maxDepth,
        criterion: treeHyperparams.criterion
      }
    );
    downloadFile('Heart_Disease_Pipeline.ipynb', notebookJson, 'application/json');
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Code className="w-5 h-5 text-rose-500" />
            <span>Equivalent Scikit-Learn Python Pipeline</span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 max-w-2xl">
            Production-grade Python code utilizing Pandas, Scikit-Learn, and Joblib that matches our TypeScript logic step-for-step.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 font-semibold text-xs rounded-xl border border-slate-300 dark:border-slate-700 transition cursor-pointer"
          >
            {copied ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy Code</span>
              </>
            )}
          </button>

          <button
            onClick={handleDownloadPy}
            className="flex items-center gap-1.5 px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs rounded-xl shadow-xs transition cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download .py</span>
          </button>

          <button
            onClick={handleDownloadIpynb}
            className="flex items-center gap-1.5 px-3.5 py-2 bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs rounded-xl shadow-md shadow-rose-600/20 transition cursor-pointer"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Download .ipynb (Notebook)</span>
          </button>
        </div>
      </div>

      {/* Code Container */}
      <div className="bg-slate-950 rounded-2xl border border-slate-800 shadow-sm overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-slate-900/80">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
            <FileCode className="w-4 h-4 text-rose-400" />
            <span>heart_disease_pipeline.py</span>
          </div>
          <span className="text-[11px] font-mono text-slate-500">Python 3.9+ / Scikit-Learn 1.3+</span>
        </div>

        <pre className="p-5 font-mono text-xs text-slate-300 overflow-x-auto leading-relaxed max-h-[600px] overflow-y-auto select-all">
          {pythonCode}
        </pre>
      </div>
    </div>
  );
};
