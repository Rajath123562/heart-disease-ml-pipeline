import React, { useRef, useState } from 'react';
import {
  Download,
  Upload,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Package,
  Layers,
  Sparkles,
  Copy
} from 'lucide-react';
import {
  DatasetMeta,
  PreprocessingConfig,
  PreprocessingSummary,
  LogisticModelArtifact,
  DecisionTreeModelArtifact,
  ModelMetrics,
  ExportedModelBundle
} from '../types';
import {
  createModelBundle,
  downloadFile,
  parseModelBundle
} from '../ml/export';

interface SaveLoadTabProps {
  meta: DatasetMeta;
  config: PreprocessingConfig;
  summary: PreprocessingSummary | null;
  logisticModel: LogisticModelArtifact | null;
  treeModel: DecisionTreeModelArtifact | null;
  logisticMetrics: ModelMetrics | null;
  treeMetrics: ModelMetrics | null;
  threshold: number;
  onImportModel: (bundle: ExportedModelBundle) => void;
}

export const SaveLoadTab: React.FC<SaveLoadTabProps> = ({
  meta,
  config,
  summary,
  logisticModel,
  treeModel,
  logisticMetrics,
  treeMetrics,
  threshold,
  onImportModel
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importStatus, setImportStatus] = useState<{
    success?: boolean;
    message?: string;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  // Generate current bundle preview
  const currentBundle: ExportedModelBundle | null = summary
    ? createModelBundle(
        meta,
        config,
        summary,
        logisticModel,
        treeModel,
        logisticMetrics,
        treeMetrics,
        threshold
      )
    : null;

  const bundleJsonString = currentBundle
    ? JSON.stringify(currentBundle, null, 2)
    : '{\n  "status": "Pipeline not yet trained. Run training to generate model bundle."\n}';

  const handleExport = () => {
    if (!currentBundle) return;
    const filename = `heart_disease_model_${new Date().toISOString().slice(0, 10)}.json`;
    downloadFile(filename, bundleJsonString);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      try {
        const text = event.target?.result as string;
        const bundle = parseModelBundle(text);
        onImportModel(bundle);
        setImportStatus({
          success: true,
          message: `Successfully restored model package (Version ${bundle.version}, trained on ${bundle.metadata?.name || 'Dataset'})!`
        });
      } catch (err: any) {
        setImportStatus({
          success: false,
          message: err.message || 'Failed to parse model JSON bundle.'
        });
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(bundleJsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm space-y-2">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Package className="w-5 h-5 text-rose-500" />
          <span>Model Serialization & Portable JSON Artifacts</span>
        </h2>
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed max-w-4xl">
          Export your fully trained pipeline as a self-contained, standalone <code>model.json</code> package.
          This bundle contains all preprocessing settings, frozen scaler parameters ($\mu$ & $\sigma$), categorical feature orders, logistic regression analytical weights ($w, b$), tree node structures, and clinical evaluation metrics.
          You can re-import this file anytime to make predictions without retraining.
        </p>
      </div>

      {/* Action Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Export Card */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-rose-600 dark:text-rose-400">
                1. Save & Download Artifact
              </span>
              <Download className="w-4 h-4 text-slate-400" />
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Export Model (`model.json`)
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Downloads the complete pipeline bundle. Suitable for client-side restoration, mobile web deployment, or archiving your Week 4 Capstone submission.
            </p>
          </div>

          <div className="pt-2">
            <button
              onClick={handleExport}
              disabled={!logisticModel && !treeModel}
              className="w-full py-2.5 px-4 bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs rounded-xl shadow-md shadow-rose-600/20 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download model.json</span>
            </button>
          </div>
        </div>

        {/* Import Card */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                2. Restore Saved Artifact
              </span>
              <Upload className="w-4 h-4 text-slate-400" />
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Import Model (`model.json`)
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Upload a previously exported JSON bundle. Restores all preprocessing scalers and trained weights, allowing immediate patient inference without retraining.
            </p>
          </div>

          <div className="pt-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={handleFileChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 font-semibold text-xs rounded-xl border border-slate-300 dark:border-slate-700 transition flex items-center justify-center gap-2 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Select model.json File</span>
            </button>
          </div>
        </div>
      </div>

      {/* Import Feedback Toast */}
      {importStatus && (
        <div
          className={`p-4 rounded-xl border text-xs flex items-center gap-2.5 ${
            importStatus.success
              ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300'
              : 'bg-rose-50 dark:bg-rose-950/40 border-rose-300 dark:border-rose-800 text-rose-800 dark:text-rose-300'
          }`}
        >
          {importStatus.success ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
          )}
          <span>{importStatus.message}</span>
        </div>
      )}

      {/* JSON Inspection Terminal Card */}
      <div className="bg-slate-950 rounded-2xl border border-slate-800 p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileCode className="w-4 h-4 text-rose-400" />
            <span className="text-xs font-mono font-bold text-slate-300">
              Payload Inspector: model.json
            </span>
          </div>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 transition"
          >
            <Copy className="w-3 h-3" />
            <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
          </button>
        </div>

        <pre className="h-64 overflow-y-auto font-mono text-[11px] text-slate-400 bg-slate-900/90 p-4 rounded-xl border border-slate-800 leading-relaxed select-all">
          {bundleJsonString}
        </pre>
      </div>
    </div>
  );
};
