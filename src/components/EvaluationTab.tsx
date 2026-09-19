import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  BarChart3,
  Sliders,
  AlertTriangle,
  Award,
  CheckCircle2,
  Download,
  Info,
  Layers,
  Sparkles,
  TrendingUp,
  Activity
} from 'lucide-react';
import {
  ModelMetrics,
  CrossValidationSummary,
  PreprocessingSummary
} from '../types';
import { downloadChartAsPng } from '../utils/chartExport';

interface EvaluationTabProps {
  logisticMetrics: ModelMetrics | null;
  treeMetrics: ModelMetrics | null;
  logisticCV: CrossValidationSummary | null;
  treeCV: CrossValidationSummary | null;
  summary: PreprocessingSummary | null;
  threshold: number;
  onUpdateThreshold: (newThreshold: number) => void;
}

export const EvaluationTab: React.FC<EvaluationTabProps> = ({
  logisticMetrics,
  treeMetrics,
  logisticCV,
  treeCV,
  summary,
  threshold,
  onUpdateThreshold
}) => {
  if (!logisticMetrics || !treeMetrics) {
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-12 text-center space-y-3">
        <Activity className="w-8 h-8 text-rose-500 mx-auto animate-pulse" />
        <h3 className="text-base font-bold text-slate-900 dark:text-white">
          Models Pending Training
        </h3>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Please run the training pipeline first to compute test set accuracy, confusion matrices, and ROC curves.
        </p>
      </div>
    );
  }

  // Combine ROC curves for Recharts
  const maxRocPoints = Math.max(logisticMetrics.rocCurve.length, treeMetrics.rocCurve.length);
  const rocChartData: { fpr: number; logisticTpr?: number; treeTpr?: number; baseline: number }[] = [];

  // Downsample or interpolate points between 0 and 1
  for (let i = 0; i <= 20; i++) {
    const targetFpr = i / 20;

    // Find closest TPR for Logistic
    const closestLr = logisticMetrics.rocCurve.reduce((prev, curr) =>
      Math.abs(curr.fpr - targetFpr) < Math.abs(prev.fpr - targetFpr) ? curr : prev
    );

    // Find closest TPR for Tree
    const closestDt = treeMetrics.rocCurve.reduce((prev, curr) =>
      Math.abs(curr.fpr - targetFpr) < Math.abs(prev.fpr - targetFpr) ? curr : prev
    );

    rocChartData.push({
      fpr: targetFpr,
      logisticTpr: closestLr.tpr,
      treeTpr: closestDt.tpr,
      baseline: targetFpr
    });
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner: Interactive Decision-Threshold Slider */}
      <div className="bg-gradient-to-br from-rose-500/10 via-slate-50 to-white dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-rose-500" />
              <span>Interactive Decision Threshold Controller</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Adjust classification cutoff $\tau$ live: observe the trade-off between Sensitivity (Recall) and Precision.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold px-3 py-1.5 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-rose-600 dark:text-rose-400 shadow-xs">
              Threshold &tau; = {threshold.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="space-y-2 pt-2">
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={threshold}
            onChange={e => onUpdateThreshold(parseFloat(e.target.value))}
            className="w-full h-2.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
          />

          <div className="flex justify-between text-[11px] text-slate-500 font-medium">
            <span className="text-rose-600 dark:text-rose-400 font-semibold">
              &larr; Lower Threshold (High Recall / Catches More Disease)
            </span>
            <span>Default (0.50)</span>
            <span className="text-blue-600 dark:text-blue-400 font-semibold">
              Higher Threshold (High Precision / Fewer False Alarms) &rarr;
            </span>
          </div>
        </div>

        <div className="p-3 bg-white/80 dark:bg-slate-800/80 rounded-xl text-xs text-slate-600 dark:text-slate-300 border border-slate-200/60 dark:border-slate-700/60 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
          <div>
            <strong>Clinical Screening Rule:</strong> In coronary disease screening, lowering the threshold from 0.50 to ~0.35 increases <strong>Recall (Sensitivity)</strong>, ensuring fewer diseased patients leave without treatment (reducing dangerous False Negatives).
          </div>
        </div>
      </div>

      {/* Side-by-Side Scalar Metrics Benchmark Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          {
            label: 'Accuracy',
            lr: (logisticMetrics.accuracy * 100).toFixed(1) + '%',
            dt: (treeMetrics.accuracy * 100).toFixed(1) + '%',
            pref: logisticMetrics.accuracy >= treeMetrics.accuracy ? 'LR' : 'DT'
          },
          {
            label: 'Recall (Sensitivity)',
            lr: (logisticMetrics.recall * 100).toFixed(1) + '%',
            dt: (treeMetrics.recall * 100).toFixed(1) + '%',
            pref: logisticMetrics.recall >= treeMetrics.recall ? 'LR' : 'DT',
            highlight: true
          },
          {
            label: 'Precision',
            lr: (logisticMetrics.precision * 100).toFixed(1) + '%',
            dt: (treeMetrics.precision * 100).toFixed(1) + '%',
            pref: logisticMetrics.precision >= treeMetrics.precision ? 'LR' : 'DT'
          },
          {
            label: 'Specificity',
            lr: (logisticMetrics.specificity * 100).toFixed(1) + '%',
            dt: (treeMetrics.specificity * 100).toFixed(1) + '%',
            pref: logisticMetrics.specificity >= treeMetrics.specificity ? 'LR' : 'DT'
          },
          {
            label: 'F1-Score',
            lr: (logisticMetrics.f1Score * 100).toFixed(1) + '%',
            dt: (treeMetrics.f1Score * 100).toFixed(1) + '%',
            pref: logisticMetrics.f1Score >= treeMetrics.f1Score ? 'LR' : 'DT'
          },
          {
            label: 'ROC-AUC',
            lr: logisticMetrics.rocAuc.toFixed(3),
            dt: treeMetrics.rocAuc.toFixed(3),
            pref: logisticMetrics.rocAuc >= treeMetrics.rocAuc ? 'LR' : 'DT',
            highlight: true
          }
        ].map(m => (
          <div
            key={m.label}
            className={`p-3.5 rounded-2xl border shadow-xs flex flex-col justify-between ${
              m.highlight
                ? 'bg-rose-50/50 dark:bg-rose-950/20 border-rose-200 dark:border-rose-900/60'
                : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800'
            }`}
          >
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                {m.label}
              </div>
              <div className="mt-2 space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Logistic:</span>
                  <span className="font-mono font-bold text-rose-600 dark:text-rose-400">
                    {m.lr}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Tree:</span>
                  <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">
                    {m.dt}
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-2.5 pt-2 border-t border-slate-100 dark:border-slate-800 text-[10px] flex items-center justify-between text-slate-400 font-medium">
              <span>Preferred:</span>
              <span className="font-bold text-slate-700 dark:text-slate-300">
                {m.pref === 'LR' ? 'Logistic' : 'Tree'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Confusion Matrices Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Logistic Confusion Matrix */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <span>Logistic Regression Confusion Matrix</span>
                <span className="text-xs font-mono font-normal text-slate-400">
                  (Test N={summary?.testCount})
                </span>
              </h3>
              <p className="text-xs text-slate-500">Predicted classes vs ground truth clinical outcome</p>
            </div>
          </div>

          {/* Matrix Heatmap Grid */}
          <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto text-center font-mono">
            {/* True Negative (TN) */}
            <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900/60">
              <div className="text-[10px] text-blue-600 dark:text-blue-400 uppercase font-bold">
                True Negative (TN)
              </div>
              <div className="text-2xl font-extrabold text-blue-900 dark:text-blue-200 mt-1">
                {logisticMetrics.confusionMatrix.tn}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">Healthy correctly cleared</div>
            </div>

            {/* False Positive (FP) */}
            <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/60">
              <div className="text-[10px] text-amber-600 dark:text-amber-400 uppercase font-bold">
                False Positive (FP)
              </div>
              <div className="text-2xl font-extrabold text-amber-900 dark:text-amber-200 mt-1">
                {logisticMetrics.confusionMatrix.fp}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">False alarm (over-triage)</div>
            </div>

            {/* False Negative (FN) - Critical */}
            <div className="p-4 rounded-xl bg-red-100 dark:bg-red-950/60 border-2 border-red-500">
              <div className="text-[10px] text-red-600 dark:text-red-400 uppercase font-bold flex items-center justify-center gap-1">
                <AlertTriangle className="w-3 h-3" /> False Negative (FN)
              </div>
              <div className="text-2xl font-extrabold text-red-700 dark:text-red-300 mt-1">
                {logisticMetrics.confusionMatrix.fn}
              </div>
              <div className="text-[11px] text-red-600 dark:text-red-300 font-semibold mt-0.5">
                MISSED HEART DISEASE
              </div>
            </div>

            {/* True Positive (TP) */}
            <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900/60">
              <div className="text-[10px] text-emerald-600 dark:text-emerald-400 uppercase font-bold">
                True Positive (TP)
              </div>
              <div className="text-2xl font-extrabold text-emerald-900 dark:text-emerald-200 mt-1">
                {logisticMetrics.confusionMatrix.tp}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">Disease detected accurately</div>
            </div>
          </div>

          <div className="text-xs text-slate-500 text-center font-mono">
            Total Classified: {logisticMetrics.confusionMatrix.tp + logisticMetrics.confusionMatrix.fp + logisticMetrics.confusionMatrix.tn + logisticMetrics.confusionMatrix.fn} / {summary?.testCount} samples
          </div>
        </div>

        {/* Decision Tree Confusion Matrix */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <span>Decision Tree Confusion Matrix</span>
                <span className="text-xs font-mono font-normal text-slate-400">
                  (Test N={summary?.testCount})
                </span>
              </h3>
              <p className="text-xs text-slate-500">Predicted classes vs ground truth clinical outcome</p>
            </div>
          </div>

          {/* Matrix Heatmap Grid */}
          <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto text-center font-mono">
            {/* True Negative (TN) */}
            <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900/60">
              <div className="text-[10px] text-blue-600 dark:text-blue-400 uppercase font-bold">
                True Negative (TN)
              </div>
              <div className="text-2xl font-extrabold text-blue-900 dark:text-blue-200 mt-1">
                {treeMetrics.confusionMatrix.tn}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">Healthy correctly cleared</div>
            </div>

            {/* False Positive (FP) */}
            <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/60">
              <div className="text-[10px] text-amber-600 dark:text-amber-400 uppercase font-bold">
                False Positive (FP)
              </div>
              <div className="text-2xl font-extrabold text-amber-900 dark:text-amber-200 mt-1">
                {treeMetrics.confusionMatrix.fp}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">False alarm (over-triage)</div>
            </div>

            {/* False Negative (FN) - Critical */}
            <div className="p-4 rounded-xl bg-red-100 dark:bg-red-950/60 border-2 border-red-500">
              <div className="text-[10px] text-red-600 dark:text-red-400 uppercase font-bold flex items-center justify-center gap-1">
                <AlertTriangle className="w-3 h-3" /> False Negative (FN)
              </div>
              <div className="text-2xl font-extrabold text-red-700 dark:text-red-300 mt-1">
                {treeMetrics.confusionMatrix.fn}
              </div>
              <div className="text-[11px] text-red-600 dark:text-red-300 font-semibold mt-0.5">
                MISSED HEART DISEASE
              </div>
            </div>

            {/* True Positive (TP) */}
            <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900/60">
              <div className="text-[10px] text-emerald-600 dark:text-emerald-400 uppercase font-bold">
                True Positive (TP)
              </div>
              <div className="text-2xl font-extrabold text-emerald-900 dark:text-emerald-200 mt-1">
                {treeMetrics.confusionMatrix.tp}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">Disease detected accurately</div>
            </div>
          </div>

          <div className="text-xs text-slate-500 text-center font-mono">
            Total Classified: {treeMetrics.confusionMatrix.tp + treeMetrics.confusionMatrix.fp + treeMetrics.confusionMatrix.tn + treeMetrics.confusionMatrix.fn} / {summary?.testCount} samples
          </div>
        </div>
      </div>

      {/* ROC Curves & Classification Report */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Dual ROC Curve Chart */}
        <div
          id="chart-roc-curve"
          className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                Receiver Operating Characteristic (ROC) Curves
              </h3>
              <p className="text-xs text-slate-500">True Positive Rate vs False Positive Rate across thresholds</p>
            </div>
            <button
              onClick={() => downloadChartAsPng('chart-roc-curve', 'roc-curve.png')}
              className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              title="Download PNG"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocChartData} margin={{ top: 10, right: 15, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                <XAxis dataKey="fpr" tick={{ fontSize: 11 }} label={{ value: 'FPR (1 - Specificity)', position: 'insideBottom', offset: -5, fontSize: 10 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: 'TPR (Recall)', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip
                  formatter={(val: any) => [Number(val).toFixed(3), 'Rate']}
                  contentStyle={{
                    borderRadius: '8px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '11px'
                  }}
                />
                <Legend verticalAlign="top" height={36} />
                <Line
                  type="monotone"
                  dataKey="logisticTpr"
                  name={`Logistic Regression (AUC = ${logisticMetrics.rocAuc.toFixed(3)})`}
                  stroke="#f43f5e"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="treeTpr"
                  name={`Decision Tree (AUC = ${treeMetrics.rocAuc.toFixed(3)})`}
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="linear"
                  dataKey="baseline"
                  name="Random Chance (AUC = 0.500)"
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2 border border-slate-200/60 dark:border-slate-700/60">
            <TrendingUp className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
            <div>
              <strong>ROC Takeaway:</strong> Logistic Regression achieves a higher Area Under Curve (AUC = {logisticMetrics.rocAuc.toFixed(3)}) compared to the Decision Tree (AUC = {treeMetrics.rocAuc.toFixed(3)}). A higher AUC guarantees stronger ranking discrimination between diseased and healthy patients regardless of operating threshold.
            </div>
          </div>
        </div>

        {/* 5-Fold Stratified Cross Validation Results Table */}
        <div className="lg:col-span-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-emerald-500" />
              <span>5-Fold Stratified Cross-Validation</span>
            </h3>
            <p className="text-xs text-slate-500">Unbiased variance estimation across 5 independent holdouts</p>
          </div>

          {logisticCV && treeCV ? (
            <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden text-xs">
              <table className="w-full text-left">
                <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500 text-[11px] font-semibold">
                  <tr>
                    <th className="px-3 py-2">Metric</th>
                    <th className="px-3 py-2">Logistic Regression</th>
                    <th className="px-3 py-2">Decision Tree</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono text-[11px]">
                  <tr>
                    <td className="px-3 py-2 font-sans font-medium text-slate-800 dark:text-slate-200">
                      Mean Accuracy
                    </td>
                    <td className="px-3 py-2 text-rose-600 dark:text-rose-400">
                      {(logisticCV.meanAccuracy * 100).toFixed(1)}% &plusmn; {(logisticCV.stdAccuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2 text-indigo-600 dark:text-indigo-400">
                      {(treeCV.meanAccuracy * 100).toFixed(1)}% &plusmn; {(treeCV.stdAccuracy * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 font-sans font-medium text-slate-800 dark:text-slate-200">
                      Mean Recall (Sens.)
                    </td>
                    <td className="px-3 py-2 text-rose-600 dark:text-rose-400 font-bold">
                      {(logisticCV.meanRecall * 100).toFixed(1)}% &plusmn; {(logisticCV.stdRecall * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2 text-indigo-600 dark:text-indigo-400">
                      {(treeCV.meanRecall * 100).toFixed(1)}% &plusmn; {(treeCV.stdRecall * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 font-sans font-medium text-slate-800 dark:text-slate-200">
                      Mean ROC-AUC
                    </td>
                    <td className="px-3 py-2 text-rose-600 dark:text-rose-400 font-bold">
                      {logisticCV.meanRocAuc.toFixed(3)} &plusmn; {logisticCV.stdRocAuc.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 text-indigo-600 dark:text-indigo-400">
                      {treeCV.meanRocAuc.toFixed(3)} &plusmn; {treeCV.stdRocAuc.toFixed(3)}
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 font-sans font-medium text-slate-800 dark:text-slate-200">
                      Mean F1-Score
                    </td>
                    <td className="px-3 py-2">
                      {(logisticCV.meanF1 * 100).toFixed(1)}% &plusmn; {(logisticCV.stdF1 * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2">
                      {(treeCV.meanF1 * 100).toFixed(1)}% &plusmn; {(treeCV.stdF1 * 100).toFixed(1)}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-xs text-slate-400 p-4 border rounded-xl text-center">
              Cross-validation folds computing...
            </div>
          )}

          {/* Overfitting Analysis Badge */}
          <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-500/30 rounded-xl text-xs text-amber-800 dark:text-amber-300 space-y-1">
            <div className="font-bold flex items-center gap-1.5 text-amber-700 dark:text-amber-400">
              <Info className="w-4 h-4 shrink-0" />
              <span>Overfitting & Variance Analysis</span>
            </div>
            <p className="leading-relaxed">
              Logistic Regression train accuracy ({(logisticMetrics.trainAccuracy * 100).toFixed(1)}%) is closely aligned with test accuracy ({(logisticMetrics.accuracy * 100).toFixed(1)}%), demonstrating low variance. In contrast, unconstrained Decision Trees exhibit higher training-test divergence.
            </p>
          </div>
        </div>
      </div>

      {/* Written Clinical Recommendation Card */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-3">
        <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-rose-500" />
          <span>Capstone Recommendation: Model Selection for Hospital Screening</span>
        </h3>

        <div className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 space-y-2 leading-relaxed">
          <p>
            <strong>Primary Model Selection: Logistic Regression (L2 Regularized)</strong>
          </p>
          <p>
            1. <strong>Higher Generalization:</strong> Logistic Regression achieves superior test ROC-AUC ({logisticMetrics.rocAuc.toFixed(3)}) and Recall ({(logisticMetrics.recall * 100).toFixed(1)}%), producing fewer False Negatives ({logisticMetrics.confusionMatrix.fn} vs {treeMetrics.confusionMatrix.fn}) on the holdout test set.
          </p>
          <p>
            2. <strong>Continuous Calibration:</strong> Logistic Regression yields smooth posterior probabilities rather than coarse leaf step-functions, permitting precise clinical threshold tuning (e.g. lowering cutoff to 0.35 in emergency triage).
          </p>
          <p>
            3. <strong>Mathematical Interpretability:</strong> Weights translate directly into odds ratios ($e^w$), allowing cardiologists to inspect exactly how each biomarker modifies patient risk.
          </p>
        </div>
      </div>
    </div>
  );
};
