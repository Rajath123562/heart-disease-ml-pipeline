import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine
} from 'recharts';
import {
  Stethoscope,
  TrendingUp,
  TrendingDown,
  Download,
  Info,
  Award,
  Layers,
  Sparkles
} from 'lucide-react';
import {
  LogisticModelArtifact,
  DecisionTreeModelArtifact
} from '../types';
import { getLogisticCoefficientsSummary } from '../ml/logistic';
import { downloadChartAsPng } from '../utils/chartExport';

interface InterpretabilityTabProps {
  logisticModel: LogisticModelArtifact | null;
  treeModel: DecisionTreeModelArtifact | null;
}

export const InterpretabilityTab: React.FC<InterpretabilityTabProps> = ({
  logisticModel,
  treeModel
}) => {
  if (!logisticModel || !treeModel) {
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-12 text-center text-slate-500 space-y-2">
        <Stethoscope className="w-8 h-8 mx-auto text-rose-500 animate-pulse" />
        <h3 className="font-bold text-slate-900 dark:text-white">Models Not Yet Trained</h3>
        <p className="text-xs">Run the pipeline first to extract feature coefficients and tree importances.</p>
      </div>
    );
  }

  // Logistic coefficients and odds ratios
  const coefficientsData = useMemo(() => {
    return getLogisticCoefficientsSummary(logisticModel);
  }, [logisticModel]);

  // Tree feature importances
  const treeImportances = useMemo(() => {
    return treeModel.featureImportances.slice(0, 10);
  }, [treeModel]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Overview Header */}
      <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm space-y-2">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-rose-500" />
          <span>Biomarker Interpretability & Feature Significance</span>
        </h2>
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed max-w-4xl">
          Clinical decision support tools require transparent, mathematically explainable reasoning.
          Here we inspect the <strong>Logistic Regression Odds Ratios</strong> ($e^w$) and <strong>Decision Tree Gini Impurity Gains</strong> to reveal exactly which physiological biomarkers drive heart disease predictions.
        </p>
      </div>

      {/* Row 1: Logistic Regression Coefficients Bar Chart */}
      <div
        id="chart-logistic-coeffs"
        className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-rose-500" />
              <span>Logistic Regression Feature Weights & Directionality</span>
            </h3>
            <p className="text-xs text-slate-500">
              Positive weights (Red) increase heart disease risk; Negative weights (Green) represent protective factors.
            </p>
          </div>
          <button
            onClick={() => downloadChartAsPng('chart-logistic-coeffs', 'logistic-coefficients.png')}
            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition self-start sm:self-auto"
            title="Download PNG"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={coefficientsData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis
                dataKey="feature"
                type="category"
                tick={{ fontSize: 11, fontWeight: 'bold' }}
                width={70}
              />
              <Tooltip
                formatter={(val: any, _name: any, item: any) => [
                  `Weight: ${Number(val).toFixed(3)} | Odds Ratio: ${item?.payload?.oddsRatio ?? ''}`,
                  item?.payload?.direction ?? ''
                ]}
                contentStyle={{
                  borderRadius: '8px',
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  color: '#f8fafc',
                  fontSize: '11px'
                }}
              />
              <ReferenceLine x={0} stroke="#64748b" strokeWidth={1.5} />
              <Bar dataKey="weight" radius={[4, 4, 4, 4]}>
                {coefficientsData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.weight > 0 ? '#f43f5e' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Odds Ratio Explanatory Table */}
        <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
          <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
            <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500 font-semibold text-[11px]">
              <tr>
                <th className="px-3.5 py-2">Biomarker</th>
                <th className="px-3.5 py-2">Weight ($w$)</th>
                <th className="px-3.5 py-2">Odds Ratio ($e^w$)</th>
                <th className="px-3.5 py-2">Clinical Interpretation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono text-[11px]">
              {coefficientsData.slice(0, 7).map(c => (
                <tr key={c.feature} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                  <td className="px-3.5 py-2 font-bold text-slate-900 dark:text-white uppercase">
                    {c.feature}
                  </td>
                  <td
                    className={`px-3.5 py-2 font-bold ${
                      c.weight > 0 ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'
                    }`}
                  >
                    {c.weight > 0 ? `+${c.weight}` : c.weight}
                  </td>
                  <td className="px-3.5 py-2 font-bold text-slate-900 dark:text-slate-100">
                    {c.oddsRatio}x
                  </td>
                  <td className="px-3.5 py-2 font-sans text-slate-600 dark:text-slate-300">
                    {c.weight > 0
                      ? `Each +1 standard deviation increases disease odds by ${((c.oddsRatio - 1) * 100).toFixed(0)}%.`
                      : `Protective factor: higher values decrease coronary disease probability.`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 2: Decision Tree Feature Importances */}
      <div
        id="chart-tree-importances"
        className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-500" />
              <span>Decision Tree (CART) Normalized Feature Importances</span>
            </h3>
            <p className="text-xs text-slate-500">
              Total weighted Gini impurity reduction accumulated across all tree split nodes.
            </p>
          </div>
          <button
            onClick={() => downloadChartAsPng('chart-tree-importances', 'tree-importances.png')}
            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition self-start sm:self-auto"
            title="Download PNG"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={treeImportances} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis dataKey="feature" tick={{ fontSize: 11, fontWeight: 'bold' }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(val: any) => [`${(Number(val) * 100).toFixed(1)}%`, 'Importance']}
                contentStyle={{
                  borderRadius: '8px',
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  color: '#f8fafc',
                  fontSize: '11px'
                }}
              />
              <Bar dataKey="importance" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Synthesis: Detailed Clinical Explanation of Top Drivers */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
        <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-500" />
          <span>Clinical Consensus: Top 5 Diagnostic Biomarkers</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
          <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-1">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              1. Chest Pain Type (cp)
            </h4>
            <p>
              Asymptomatic angina (type 3) and atypical presentations paradoxically correlate strongly with advanced multivessel coronary artery occlusion, as patients fail to seek care until obstruction is severe.
            </p>
          </div>

          <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-1">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              2. Maximum Heart Rate (thalach)
            </h4>
            <p>
              Serves as a robust protective biomarker. Patients capable of achieving higher chronotropic stress during treadmill exercise exhibit greater myocardial oxygen reserve and lower ischemic burden.
            </p>
          </div>

          <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-1">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              3. ST Depression (oldpeak)
            </h4>
            <p>
              Electrocardiographic ST-segment depression relative to rest is a classic hallmark of exercise-induced subendocardial ischemia. Higher millimeter values exponentially increase the odds of severe coronary stenosis.
            </p>
          </div>

          <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-1">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              4. Major Fluoroscopy Vessels (ca) & Thallium (thal)
            </h4>
            <p>
              The number of major colored vessels visualized during fluoroscopy (0–3) provides direct physical confirmation of arterial plaque. Reversible thallium defects indicate hypoperfused cardiac tissue.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
