import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import {
  Download,
  Database,
  Layers,
  AlertCircle,
  Copy,
  Activity,
  BarChart2,
  TrendingUp,
  Info
} from 'lucide-react';
import { DatasetMeta, PreprocessingSummary } from '../types';
import { downloadChartAsPng } from '../utils/chartExport';

interface EdaTabProps {
  meta: DatasetMeta;
  summary?: PreprocessingSummary | null;
  rawRows?: Record<string, any>[];
  data?: Record<string, any>[];
}

export const EdaTab: React.FC<EdaTabProps> = ({ meta, summary, rawRows, data }) => {
  const rows = rawRows || data || [];
  const [selectedNumericFeature, setSelectedNumericFeature] = useState<string>('age');

  // Compute class balance
  const classStats = useMemo(() => {
    let count0 = 0;
    let count1 = 0;
    const targetCol = meta.targetColumn;

    for (const r of rows) {
      const val = Number(r[targetCol]);
      if (val === 1 || val > 0) count1++;
      else count0++;
    }

    const total = rows.length || 1;
    return {
      count0,
      count1,
      pct0: ((count0 / total) * 100).toFixed(1),
      pct1: ((count1 / total) * 100).toFixed(1),
      total
    };
  }, [rows, meta.targetColumn]);

  const pieData = [
    { name: 'Class 0: No Disease', value: classStats.count0, color: '#3b82f6' },
    { name: 'Class 1: Heart Disease', value: classStats.count1, color: '#f43f5e' }
  ];

  // Numeric features list
  const numericFeatures = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'];

  // Histogram binning calculation for selected feature
  const histogramData = useMemo(() => {
    const targetCol = meta.targetColumn;
    const vals0: number[] = [];
    const vals1: number[] = [];

    for (const r of rows) {
      const v = Number(r[selectedNumericFeature]);
      if (!isNaN(v)) {
        const t = Number(r[targetCol]) > 0 ? 1 : 0;
        if (t === 1) vals1.push(v);
        else vals0.push(v);
      }
    }

    if (vals0.length === 0 && vals1.length === 0) return [];

    const allVals = [...vals0, ...vals1];
    const min = Math.min(...allVals);
    const max = Math.max(...allVals);
    const numBins = 8;
    const step = (max - min) / numBins || 1;

    const bins = Array.from({ length: numBins }, (_, i) => {
      const start = min + i * step;
      const end = start + step;
      const label = `${start.toFixed(1)}-${end.toFixed(1)}`;
      return { bin: label, class0: 0, class1: 0 };
    });

    for (const v of vals0) {
      let bIdx = Math.floor((v - min) / step);
      if (bIdx >= numBins) bIdx = numBins - 1;
      if (bIdx >= 0) bins[bIdx].class0++;
    }

    for (const v of vals1) {
      let bIdx = Math.floor((v - min) / step);
      if (bIdx >= numBins) bIdx = numBins - 1;
      if (bIdx >= 0) bins[bIdx].class1++;
    }

    return bins;
  }, [rows, selectedNumericFeature, meta.targetColumn]);

  // Compute Pearson Correlation Matrix between numeric features and target
  const correlationMatrix = useMemo(() => {
    const targetCol = meta.targetColumn;
    const features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', targetCol];

    // Helper: compute Pearson r between two arrays
    function pearson(x: number[], y: number[]): number {
      const n = x.length;
      if (n === 0) return 0;
      let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
      for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
      }
      const num = n * sumXY - sumX * sumY;
      const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
      return den === 0 ? 0 : num / den;
    }

    // Extract columns
    const cols: Record<string, number[]> = {};
    for (const f of features) {
      cols[f] = rows.map(r => {
        const v = Number(r[f]);
        return isNaN(v) ? 0 : (f === targetCol ? (v > 0 ? 1 : 0) : v);
      });
    }

    const matrix: { f1: string; f2: string; corr: number }[] = [];
    for (const f1 of features) {
      for (const f2 of features) {
        const corr = pearson(cols[f1], cols[f2]);
        matrix.push({ f1, f2, corr: Number(corr.toFixed(2)) });
      }
    }

    return { features, matrix };
  }, [rows, meta.targetColumn]);

  // Quantile comparison for key numeric features (target 0 vs target 1)
  const quantileStats = useMemo(() => {
    const targetCol = meta.targetColumn;

    function getStats(vals: number[]) {
      if (vals.length === 0) return { min: 0, q1: 0, median: 0, q3: 0, max: 0 };
      const sorted = [...vals].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const median = sorted[Math.floor(sorted.length * 0.5)];
      const q1 = sorted[Math.floor(sorted.length * 0.25)];
      const q3 = sorted[Math.floor(sorted.length * 0.75)];
      return { min, q1, median, q3, max };
    }

    return numericFeatures.map(feat => {
      const v0: number[] = [];
      const v1: number[] = [];
      for (const r of rows) {
        const v = Number(r[feat]);
        if (!isNaN(v)) {
          const t = Number(r[targetCol]) > 0 ? 1 : 0;
          if (t === 1) v1.push(v);
          else v0.push(v);
        }
      }
      return {
        feature: feat,
        class0: getStats(v0),
        class1: getStats(v1)
      };
    });
  }, [rows, meta.targetColumn]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-2xl shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 text-xs">
            <span className="font-semibold uppercase">Total Samples</span>
            <Database className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {classStats.total}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Patient records</div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-2xl shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 text-xs">
            <span className="font-semibold uppercase">Features</span>
            <Layers className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {meta.columnCount - 1}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">+1 Target column</div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-2xl shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 text-xs">
            <span className="font-semibold uppercase">Missing Cells</span>
            <AlertCircle className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {summary?.missingCount ?? 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Imputed median/mode</div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-2xl shadow-sm">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 text-xs">
            <span className="font-semibold uppercase">Duplicates</span>
            <Copy className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
            {summary?.duplicatesRemoved ?? 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Removed on load</div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-2xl shadow-sm col-span-2 sm:col-span-1">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 text-xs">
            <span className="font-semibold uppercase">Class Balance</span>
            <Activity className="w-4 h-4 text-rose-500" />
          </div>
          <div className="text-xl font-extrabold text-slate-900 dark:text-white mt-1">
            {classStats.pct0}% / {classStats.pct1}%
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Balanced cohort</div>
        </div>
      </div>

      {/* Row 1: Target Balance + Feature Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Class Balance Donut */}
        <div
          id="chart-class-balance"
          className="lg:col-span-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                Diagnostic Class Balance
              </h3>
              <p className="text-xs text-slate-500">Distribution of patient target outcomes</p>
            </div>
            <button
              onClick={() => downloadChartAsPng('chart-class-balance', 'class-balance.png')}
              className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              title="Download PNG"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(val: any, name: any) => [
                    `${val} patients (${((Number(val) / classStats.total) * 100).toFixed(1)}%)`,
                    String(name)
                  ]}
                  contentStyle={{
                    borderRadius: '8px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '12px'
                  }}
                />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2 border border-slate-200/60 dark:border-slate-700/60">
            <Info className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
            <div>
              <strong>Takeaway:</strong> The dataset exhibits an approximately balanced target distribution ({classStats.count0} healthy vs {classStats.count1} disease-positive). This eliminates the requirement for extreme synthetic oversampling (e.g. SMOTE), though stratified splitting remains crucial.
            </div>
          </div>
        </div>

        {/* Numeric Feature Distribution by Class */}
        <div
          id="chart-feature-dist"
          className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                Feature Distribution by Outcome
              </h3>
              <p className="text-xs text-slate-500">Histogram comparison between healthy & disease classes</p>
            </div>
            <div className="flex items-center gap-2">
              <select
                value={selectedNumericFeature}
                onChange={e => setSelectedNumericFeature(e.target.value)}
                className="py-1 px-2.5 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value="age">Age (years)</option>
                <option value="thalach">Max Heart Rate (thalach)</option>
                <option value="oldpeak">ST Depression (oldpeak)</option>
                <option value="trestbps">Resting BP (trestbps)</option>
                <option value="chol">Serum Chol (chol)</option>
              </select>

              <button
                onClick={() => downloadChartAsPng('chart-feature-dist', `${selectedNumericFeature}-distribution.png`)}
                className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                title="Download PNG"
              >
                <Download className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: '8px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '12px'
                  }}
                />
                <Legend verticalAlign="top" height={30} />
                <Bar dataKey="class0" name="No Disease (0)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="class1" name="Heart Disease (1)" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2 border border-slate-200/60 dark:border-slate-700/60">
            <TrendingUp className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
            <div>
              <strong>Takeaway ({selectedNumericFeature}):</strong>{' '}
              {selectedNumericFeature === 'thalach' &&
                'Diseased patients show lower peak heart rates during stress testing (chronotropic incompetence), separating them markedly from healthy patients.'}
              {selectedNumericFeature === 'oldpeak' &&
                'ST-segment depression > 1.5 mm is heavily concentrated in disease-positive cases, representing exercise-induced myocardial ischemia.'}
              {selectedNumericFeature === 'age' &&
                'Disease incidence trends higher beyond age 55, reflecting cumulative coronary arteriosclerosis risks.'}
              {selectedNumericFeature === 'chol' &&
                'Serum cholesterol is broadly elevated across the cohort, exhibiting subtle right-skew that benefits from outlier capping.'}
              {selectedNumericFeature === 'trestbps' &&
                'Resting blood pressure demonstrates consistent moderate elevation in older subsets with arterial stiffness.'}
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Correlation Heatmap Matrix */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-purple-500" />
              <span>Pearson Correlation Heatmap Matrix</span>
            </h3>
            <p className="text-xs text-slate-500">Pairwise linear correlation coefficients between biomarkers and target</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <div className="grid grid-cols-7 gap-1 text-center text-xs font-mono">
              <div className="p-2 font-bold text-slate-400">Feature</div>
              {correlationMatrix.features.map(f => (
                <div key={f} className="p-2 font-bold text-slate-700 dark:text-slate-300 uppercase truncate">
                  {f}
                </div>
              ))}

              {correlationMatrix.features.map(f1 => (
                <React.Fragment key={f1}>
                  <div className="p-2 font-bold text-slate-700 dark:text-slate-300 text-left uppercase truncate">
                    {f1}
                  </div>
                  {correlationMatrix.features.map(f2 => {
                    const item = correlationMatrix.matrix.find(m => m.f1 === f1 && m.f2 === f2);
                    const val = item ? item.corr : 0;

                    // Color based on Pearson r: [-1, 1]
                    let bg = 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400';
                    if (val === 1) {
                      bg = 'bg-slate-200 dark:bg-slate-700 font-bold text-slate-800 dark:text-slate-100';
                    } else if (val > 0.3) {
                      bg = 'bg-rose-500/80 text-white font-bold';
                    } else if (val > 0.1) {
                      bg = 'bg-rose-500/30 text-rose-800 dark:text-rose-200';
                    } else if (val < -0.3) {
                      bg = 'bg-blue-600/80 text-white font-bold';
                    } else if (val < -0.1) {
                      bg = 'bg-blue-500/30 text-blue-800 dark:text-blue-200';
                    }

                    return (
                      <div
                        key={`${f1}-${f2}`}
                        className={`p-2 rounded-md flex items-center justify-center transition-transform hover:scale-105 ${bg}`}
                        title={`${f1} vs ${f2}: r = ${val}`}
                      >
                        {val.toFixed(2)}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2 border border-slate-200/60 dark:border-slate-700/60">
          <Info className="w-4 h-4 text-purple-500 shrink-0 mt-0.5" />
          <div>
            <strong>Correlation Takeaway:</strong> Maximum heart rate (<em>thalach</em>) exhibits strong negative correlation with the disease target (r ~ -0.42), making it a major protective biomarker. In contrast, ST depression (<em>oldpeak</em>) shows strong positive correlation (r ~ +0.43), indicating ischemic hazard.
          </div>
        </div>
      </div>

      {/* Row 3: Quantile / Boxplot Comparison Table */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white">
            Biomarker Quantile Summaries by Class (Median & Interquartile Ranges)
          </h3>
          <p className="text-xs text-slate-500">Direct statistical comparison of distributions between healthy and diseased subjects</p>
        </div>

        <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
          <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
            <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-semibold">
              <tr>
                <th className="px-4 py-2.5">Biomarker</th>
                <th className="px-4 py-2.5">Class 0 (No Disease) Median [Q1 - Q3]</th>
                <th className="px-4 py-2.5">Class 1 (Heart Disease) Median [Q1 - Q3]</th>
                <th className="px-4 py-2.5">Distribution Shift</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800 bg-white dark:bg-slate-900">
              {quantileStats.map(stat => (
                <tr key={stat.feature} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                  <td className="px-4 py-2.5 font-mono font-bold text-slate-900 dark:text-slate-100">
                    {stat.feature}
                  </td>
                  <td className="px-4 py-2.5 font-mono">
                    <span className="font-semibold text-blue-600 dark:text-blue-400">
                      {stat.class0.median.toFixed(1)}
                    </span>{' '}
                    <span className="text-slate-400">
                      [{stat.class0.q1.toFixed(1)} - {stat.class0.q3.toFixed(1)}]
                    </span>
                  </td>
                  <td className="px-4 py-2.5 font-mono">
                    <span className="font-semibold text-rose-600 dark:text-rose-400">
                      {stat.class1.median.toFixed(1)}
                    </span>{' '}
                    <span className="text-slate-400">
                      [{stat.class1.q1.toFixed(1)} - {stat.class1.q3.toFixed(1)}]
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-slate-600 dark:text-slate-300">
                    {stat.class1.median > stat.class0.median
                      ? 'Elevated in cardiac patients'
                      : 'Depressed in cardiac patients'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
