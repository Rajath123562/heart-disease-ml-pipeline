import React from 'react';
import {
  Play,
  RotateCcw,
  CheckCircle2,
  Database,
  Layers,
  Sparkles,
  Award
} from 'lucide-react';
import {
  DatasetMeta,
  PreprocessingConfig,
  PreprocessingSummary,
  ModelMetrics
} from '../types';

interface SidebarProps {
  meta?: DatasetMeta;
  config?: PreprocessingConfig;
  summary: PreprocessingSummary | null;
  logisticMetrics: ModelMetrics | null;
  treeMetrics: ModelMetrics | null;
  threshold?: number;
  isTraining: boolean;
  onRunFullPipeline?: () => void;
  onTrainModels?: () => void;
  onResetDefaults?: () => void;
  onResetDataset?: () => void;
  onSelectTab?: (tabId: string) => void;
  setActiveTab?: (tabId: string) => void;
  onUploadCsv?: (csvText: string, filename: string) => void;
  activeTab: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  meta,
  config,
  summary,
  logisticMetrics,
  treeMetrics,
  threshold = 0.5,
  isTraining,
  onRunFullPipeline,
  onTrainModels,
  onResetDefaults,
  onResetDataset,
  onSelectTab,
  setActiveTab,
  onUploadCsv,
  activeTab
}) => {
  const handleTab = setActiveTab || onSelectTab || (() => {});
  const handleTrain = onTrainModels || onRunFullPipeline || (() => {});
  const handleReset = onResetDataset || onResetDefaults || (() => {});

  // Steps completion status
  const isLoaded = meta ? meta.rowCount > 0 : true;
  const isPreprocessed = !!summary;
  const isLogisticTrained = !!logisticMetrics;
  const isTreeTrained = !!treeMetrics;
  const isEvaluated = isLogisticTrained && isTreeTrained;

  const pipelineSteps = [
    { id: 'overview', label: '1. Dataset Ingestion', done: isLoaded },
    { id: 'eda', label: '2. Exploratory Analysis', done: isLoaded },
    { id: 'cleaning', label: '3. Clean & Preprocess', done: isPreprocessed },
    { id: 'cleaning', label: '4. Stratified Split (80/20)', done: isPreprocessed },
    { id: 'cleaning', label: '5. Train-Only Scaling', done: isPreprocessed },
    { id: 'training', label: '6. Train Logistic & Tree', done: isEvaluated },
    { id: 'evaluation', label: '7. Confusion & ROC', done: isEvaluated },
    { id: 'predict', label: '8. Patient Risk Inference', done: isEvaluated }
  ];

  return (
    <aside className="w-full lg:w-72 shrink-0 space-y-4">
      {/* Primary Action Panel */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm space-y-3">
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Pipeline Orchestration
        </h2>

        <div className="space-y-2">
          <button
            onClick={handleTrain}
            disabled={isTraining}
            className="w-full py-2.5 px-4 bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-semibold text-sm rounded-xl shadow-md shadow-rose-600/20 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer"
          >
            {isTraining ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Training Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Full Pipeline</span>
              </>
            )}
          </button>

          <button
            onClick={handleReset}
            className="w-full py-2 px-3 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-medium text-xs rounded-xl transition flex items-center justify-center gap-2 border border-slate-200 dark:border-slate-700/60"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset to Defaults</span>
          </button>
        </div>

        {/* Dataset Status Summary */}
        <div className="pt-3 border-t border-slate-100 dark:border-slate-800/80 space-y-2 text-xs">
          <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-rose-500" />
              Dataset Source:
            </span>
            <span className="font-semibold text-slate-800 dark:text-slate-200 truncate max-w-[120px]">
              {meta ? (meta.source === 'uci_authentic' ? 'UCI Cleveland' : (meta.source === 'synthetic' ? 'Synthetic Demo' : 'Custom CSV')) : 'UCI Cleveland'}
            </span>
          </div>

          <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
            <span>Instances / Features:</span>
            <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
              {meta ? `${meta.rowCount} / ${meta.columnCount}` : '303 / 14'}
            </span>
          </div>

          <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
            <span>Target Column:</span>
            <span className="font-mono font-semibold text-rose-600 dark:text-rose-400">
              {meta ? meta.targetColumn : 'target'}
            </span>
          </div>

          <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
            <span>Decision Threshold:</span>
            <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
              {threshold.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Pipeline Settings Snapshot */}
      {config && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center justify-between">
            <span>Config Snapshot</span>
            <Layers className="w-3.5 h-3.5 text-slate-400" />
          </h3>

          <div className="space-y-1.5 text-xs text-slate-600 dark:text-slate-400">
            <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/60">
              <span>Split Ratio:</span>
              <span className="font-medium text-slate-800 dark:text-slate-200">
                {((1 - config.testSize) * 100).toFixed(0)}% Train / {(config.testSize * 100).toFixed(0)}% Test
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/60">
              <span>Feature Scaler:</span>
              <span className="font-medium text-slate-800 dark:text-slate-200">
                {config.scalingMethod === 'standard' ? 'StandardScaler (z)' : (config.scalingMethod === 'minmax' ? 'MinMaxScaler' : 'None')}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-100 dark:border-slate-800/60">
              <span>Leakage Prevention:</span>
              <span className="font-medium text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Fit on Train
              </span>
            </div>

            <div className="flex justify-between py-1">
              <span>Missing Strategy:</span>
              <span className="font-medium text-slate-800 dark:text-slate-200">
                {config.missingStrategy === 'median_mode' ? 'Median / Mode' : config.missingStrategy}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Model Performance Mini-Card (if trained) */}
      {logisticMetrics && (
        <div className="bg-gradient-to-br from-rose-500/10 via-amber-500/5 to-transparent border border-rose-500/20 rounded-2xl p-4 shadow-sm space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-rose-600 dark:text-rose-400 flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5" /> Test Set Scores
            </span>
            <span className="text-[10px] text-slate-500">Holdout N={summary?.testCount}</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center pt-1">
            <div className="bg-white/80 dark:bg-slate-800/80 p-2 rounded-xl border border-rose-100 dark:border-slate-700/60">
              <div className="text-[10px] text-slate-500 uppercase font-semibold">Logistic AUC</div>
              <div className="text-base font-extrabold text-rose-600 dark:text-rose-400">
                {logisticMetrics.rocAuc.toFixed(3)}
              </div>
              <div className="text-[10px] text-slate-400">Recall: {(logisticMetrics.recall * 100).toFixed(0)}%</div>
            </div>

            <div className="bg-white/80 dark:bg-slate-800/80 p-2 rounded-xl border border-slate-200 dark:border-slate-700/60">
              <div className="text-[10px] text-slate-500 uppercase font-semibold">Tree AUC</div>
              <div className="text-base font-extrabold text-slate-700 dark:text-slate-300">
                {treeMetrics?.rocAuc ? treeMetrics.rocAuc.toFixed(3) : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-400">Recall: {treeMetrics ? (treeMetrics.recall * 100).toFixed(0) : 0}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline Progress Checklist */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center justify-between">
          <span>Pipeline Checklist</span>
          <Sparkles className="w-3.5 h-3.5 text-amber-500" />
        </h3>

        <div className="space-y-1.5 text-xs">
          {pipelineSteps.map(step => (
            <button
              key={step.label}
              onClick={() => handleTab(step.id)}
              className={`w-full flex items-center justify-between py-1.5 px-2 rounded-lg text-left transition ${
                activeTab === step.id
                  ? 'bg-slate-100 dark:bg-slate-800 font-semibold text-rose-600 dark:text-rose-400'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              <span className="truncate">{step.label}</span>
              {step.done ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 ml-1.5" />
              ) : (
                <div className="w-3 h-3 rounded-full border border-slate-300 dark:border-slate-600 shrink-0 ml-1.5" />
              )}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
};
