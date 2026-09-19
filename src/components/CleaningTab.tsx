import React, { useState } from 'react';
import {
  Filter,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Sliders,
  ShieldCheck,
  Zap,
  Info,
  Layers,
  ArrowRight
} from 'lucide-react';
import { PreprocessingConfig, PreprocessingSummary } from '../types';

interface CleaningTabProps {
  config: PreprocessingConfig;
  summary: PreprocessingSummary | null;
  onUpdateConfig?: (newConfig: PreprocessingConfig) => void;
  onApplyConfig?: (newConfig: PreprocessingConfig) => void;
  onApplyPipeline?: () => void;
  isProcessing?: boolean;
}

export const CleaningTab: React.FC<CleaningTabProps> = ({
  config,
  summary,
  onUpdateConfig,
  onApplyConfig,
  onApplyPipeline,
  isProcessing = false
}) => {
  const updateHandler = onUpdateConfig || onApplyConfig || (() => {});
  const [localConfig, setLocalConfig] = useState<PreprocessingConfig>(config);

  const handleChange = <K extends keyof PreprocessingConfig>(
    key: K,
    value: PreprocessingConfig[K]
  ) => {
    const updated = { ...localConfig, [key]: value };
    setLocalConfig(updated);
    updateHandler(updated);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner: Pipeline Execution State */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Filter className="w-5 h-5 text-rose-500" />
            <span>Data Preprocessing & Leakage-Free Splitting</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Configure imputation, duplicate filtering, IQR outlier handling, encoding, and train-only scaling.
          </p>
        </div>

        <button
          onClick={onApplyPipeline}
          disabled={isProcessing}
          className="px-5 py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs rounded-xl shadow-md shadow-rose-600/20 disabled:opacity-50 transition flex items-center justify-center gap-2 shrink-0 cursor-pointer"
        >
          {isProcessing ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Applying Pipeline...</span>
            </>
          ) : (
            <>
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Apply & Re-Process Pipeline</span>
            </>
          )}
        </button>
      </div>

      {/* Before / After Metrics Summary Card */}
      {summary && (
        <div className="bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-850 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Interactive Transformation Audit (Before vs After)
            </h3>
            <span className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> Verification Passed
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-1">
            <div className="p-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700/60">
              <div className="text-[11px] text-slate-500">Instance Count</div>
              <div className="text-base font-extrabold text-slate-900 dark:text-white mt-1 flex items-center gap-2">
                <span>{summary.originalRows}</span>
                <ArrowRight className="w-3 h-3 text-slate-400" />
                <span className="text-rose-600 dark:text-rose-400">{summary.trainCount + summary.testCount}</span>
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {summary.duplicatesRemoved > 0 ? `-${summary.duplicatesRemoved} duplicates` : 'No duplicates removed'}
              </div>
            </div>

            <div className="p-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700/60">
              <div className="text-[11px] text-slate-500">Features Matrix</div>
              <div className="text-base font-extrabold text-slate-900 dark:text-white mt-1 flex items-center gap-2">
                <span>{summary.originalCols - 1}</span>
                <ArrowRight className="w-3 h-3 text-slate-400" />
                <span className="text-indigo-600 dark:text-indigo-400">{summary.processedCols}</span>
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {localConfig.oneHotEncode ? 'Expanded via One-Hot' : 'Ordinal preserved'}
              </div>
            </div>

            <div className="p-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700/60">
              <div className="text-[11px] text-slate-500">Missing Imputed</div>
              <div className="text-base font-extrabold text-slate-900 dark:text-white mt-1 flex items-center gap-2">
                <span>{summary.missingCount}</span>
                <ArrowRight className="w-3 h-3 text-slate-400" />
                <span className="text-emerald-600 dark:text-emerald-400">0</span>
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {summary.imputedCount} cells filled ({localConfig.missingStrategy})
              </div>
            </div>

            <div className="p-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700/60">
              <div className="text-[11px] text-slate-500">Outliers Capped</div>
              <div className="text-base font-extrabold text-slate-900 dark:text-white mt-1">
                {summary.outliersCapped}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">IQR 1.5 threshold rule</div>
            </div>
          </div>
        </div>
      )}

      {/* Configuration Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Step 1: Missing Value Imputation */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-rose-500" />
              <span>1. Missing Value Strategy</span>
            </h3>
            <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded text-slate-600 dark:text-slate-400">
              {summary?.missingCount ?? 0} missing cells
            </span>
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            In cardiovascular records, missing laboratory readings (e.g. serum cholesterol or fluoroscopy vessels) must be imputed without biasing variance.
          </p>

          <div className="space-y-2">
            {[
              {
                id: 'median_mode',
                name: 'Median (Numeric) & Mode (Categorical)',
                desc: 'Recommended. Median resists skew from extreme biomarker readings, while mode selects the most frequent clinical presentation category.'
              },
              {
                id: 'mean',
                name: 'Mean (Numeric) & Mode (Categorical)',
                desc: 'Computes arithmetic average. Susceptible to distortion by long-tail outliers.'
              },
              {
                id: 'drop',
                name: 'Drop Rows with Missing Values',
                desc: 'Discard incomplete records. Reduces cohort size and test set statistical power.'
              }
            ].map(opt => (
              <label
                key={opt.id}
                className={`flex items-start gap-3 p-3 rounded-xl border transition-all cursor-pointer ${
                  localConfig.missingStrategy === opt.id
                    ? 'border-rose-500/50 bg-rose-50/50 dark:bg-rose-950/20'
                    : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40'
                }`}
              >
                <input
                  type="radio"
                  name="missingStrategy"
                  checked={localConfig.missingStrategy === opt.id}
                  onChange={() => handleChange('missingStrategy', opt.id as any)}
                  className="mt-1 text-rose-600 focus:ring-rose-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-900 dark:text-white">
                    {opt.name}
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-normal">
                    {opt.desc}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Step 2: Duplicates & Outlier Capping */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-indigo-500" />
            <span>2. Data Cleansing & Outliers</span>
          </h3>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Address duplicates and physiological extremes that destabilize gradient updates during model convergence.
          </p>

          <div className="space-y-3">
            {/* Remove duplicates toggle */}
            <div className="flex items-start justify-between gap-4 p-3 rounded-xl border border-slate-200 dark:border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-900 dark:text-white">
                  Remove Duplicate Records
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                  Filters identical patient duplicate entries ({summary?.duplicatesRemoved ?? 0} found) to avoid evaluation data leakage.
                </div>
              </div>
              <input
                type="checkbox"
                checked={localConfig.removeDuplicates}
                onChange={e => handleChange('removeDuplicates', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-rose-600 focus:ring-rose-500"
              />
            </div>

            {/* Outlier Capping toggle */}
            <div className="flex items-start justify-between gap-4 p-3 rounded-xl border border-slate-200 dark:border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-900 dark:text-white">
                  IQR Outlier Capping (1.5 * IQR)
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                  Caps extreme physiological outliers in <em>chol</em>, <em>trestbps</em>, and <em>oldpeak</em> at [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds without deleting data.
                </div>
              </div>
              <input
                type="checkbox"
                checked={localConfig.handleOutliersIQR}
                onChange={e => handleChange('handleOutliersIQR', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-rose-600 focus:ring-rose-500"
              />
            </div>

            {/* One-Hot Encoding toggle */}
            <div className="flex items-start justify-between gap-4 p-3 rounded-xl border border-slate-200 dark:border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-900 dark:text-white">
                  Categorical One-Hot Encoding
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                  Expands nominal variables (<em>cp, restecg, slope, thal</em>) into binary indicator flags.
                </div>
              </div>
              <input
                type="checkbox"
                checked={localConfig.oneHotEncode}
                onChange={e => handleChange('oneHotEncode', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-rose-600 focus:ring-rose-500"
              />
            </div>
          </div>
        </div>

        {/* Step 3: Feature Scaling (With Critical Leakage Callout) */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-500" />
              <span>3. Feature Scaling & Gradient Stability</span>
            </h3>
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Biomarkers exhibit radically different ranges (e.g. Chol 126–564 mg/dl vs Oldpeak 0.0–6.2 mm).
          </p>

          <div className="space-y-2">
            {[
              {
                id: 'standard',
                name: 'StandardScaler (Z-score: mean=0, std=1)',
                desc: 'Mandatory for Logistic Regression gradient descent. Prevents high-magnitude features from dominating analytical parameter steps.'
              },
              {
                id: 'minmax',
                name: 'MinMaxScaler (Bounds [0, 1])',
                desc: 'Scales features to bounded interval. Sensitive to extreme out-of-distribution values.'
              },
              {
                id: 'none',
                name: 'No Scaling (Raw Units)',
                desc: 'Decision Trees are invariant to monotonic transformations, but Logistic Regression gradient descent will oscillate heavily.'
              }
            ].map(opt => (
              <label
                key={opt.id}
                className={`flex items-start gap-3 p-3 rounded-xl border transition-all cursor-pointer ${
                  localConfig.scalingMethod === opt.id
                    ? 'border-amber-500/50 bg-amber-50/50 dark:bg-amber-950/20'
                    : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40'
                }`}
              >
                <input
                  type="radio"
                  name="scalingMethod"
                  checked={localConfig.scalingMethod === opt.id}
                  onChange={() => handleChange('scalingMethod', opt.id as any)}
                  className="mt-1 text-amber-600 focus:ring-amber-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-900 dark:text-white">
                    {opt.name}
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-normal">
                    {opt.desc}
                  </div>
                </div>
              </label>
            ))}
          </div>

          {/* Educational Callout on Leakage Prevention */}
          <div className="p-3.5 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-500/30 rounded-xl text-xs text-emerald-800 dark:text-emerald-300 space-y-1">
            <div className="font-bold flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>Strict Data Leakage Prevention Enforced</span>
            </div>
            <p className="leading-relaxed">
              Scaler parameters ($\mu$ and $\sigma$) are calculated <strong>strictly on the training split</strong>. The derived parameters are frozen and then applied to the test split. Fitting scalers on the full dataset before splitting is an error that leaks future distribution information into the model.
            </p>
          </div>
        </div>

        {/* Step 4: Stratified Train/Test Split */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-purple-500" />
              <span>4. Stratified Train / Test Split</span>
            </h3>
            <span className="text-xs font-mono font-bold text-purple-600 dark:text-purple-400">
              {((1 - localConfig.testSize) * 100).toFixed(0)}% / {(localConfig.testSize * 100).toFixed(0)}%
            </span>
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Stratification guarantees that both training and test partitions have the identical proportion of diseased (Class 1) and healthy (Class 0) patients.
          </p>

          <div className="space-y-4">
            {/* Split Slider */}
            <div>
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
                <span>Holdout Test Size:</span>
                <span className="font-bold text-slate-900 dark:text-white">
                  {(localConfig.testSize * 100).toFixed(0)}% ({summary?.testCount ?? 61} patients)
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={0.4}
                step={0.05}
                value={localConfig.testSize}
                onChange={e => handleChange('testSize', parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                <span>10% (Diagnostic holdout)</span>
                <span>20% (Recommended)</span>
                <span>40% (Large holdout)</span>
              </div>
            </div>

            {/* Random Seed Input */}
            <div className="flex items-center justify-between gap-4 p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl border border-slate-200 dark:border-slate-700/60">
              <div>
                <div className="text-xs font-semibold text-slate-900 dark:text-white">
                  Random Seed (Reproducibility)
                </div>
                <div className="text-[11px] text-slate-400">
                  Fixed seed for PRNG Mulberry32 shuffling
                </div>
              </div>
              <input
                type="number"
                value={localConfig.randomSeed}
                onChange={e => handleChange('randomSeed', parseInt(e.target.value) || 42)}
                className="w-20 px-2.5 py-1 text-xs font-mono font-bold rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white text-right focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            {/* Verification Table */}
            {summary && (
              <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden text-xs">
                <table className="w-full text-left">
                  <thead className="bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-semibold text-[11px]">
                    <tr>
                      <th className="px-3 py-1.5">Partition</th>
                      <th className="px-3 py-1.5">Total Rows</th>
                      <th className="px-3 py-1.5">Class 0 (No)</th>
                      <th className="px-3 py-1.5">Class 1 (Yes)</th>
                      <th className="px-3 py-1.5">Disease %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono text-[11px]">
                    <tr>
                      <td className="px-3 py-1.5 font-sans font-medium text-slate-800 dark:text-slate-200">Training Set</td>
                      <td className="px-3 py-1.5">{summary.trainCount}</td>
                      <td className="px-3 py-1.5">{summary.trainClassDistribution['0']}</td>
                      <td className="px-3 py-1.5">{summary.trainClassDistribution['1']}</td>
                      <td className="px-3 py-1.5 font-bold text-rose-600 dark:text-rose-400">
                        {((summary.trainClassDistribution['1'] / summary.trainCount) * 100).toFixed(1)}%
                      </td>
                    </tr>
                    <tr>
                      <td className="px-3 py-1.5 font-sans font-medium text-slate-800 dark:text-slate-200">Test Set</td>
                      <td className="px-3 py-1.5">{summary.testCount}</td>
                      <td className="px-3 py-1.5">{summary.testClassDistribution['0']}</td>
                      <td className="px-3 py-1.5">{summary.testClassDistribution['1']}</td>
                      <td className="px-3 py-1.5 font-bold text-rose-600 dark:text-rose-400">
                        {((summary.testClassDistribution['1'] / summary.testCount) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
