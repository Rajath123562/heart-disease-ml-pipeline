import React, { useState, useMemo } from 'react';
import {
  Activity,
  ArrowRight,
  Database,
  Filter,
  Search,
  CheckCircle2,
  FileSpreadsheet,
  Sliders,
  Cpu,
  BarChart3,
  Stethoscope,
  Info
} from 'lucide-react';
import { HEART_DATA_DICTIONARY } from '../data/heartData';
import { DatasetMeta } from '../types';

interface OverviewTabProps {
  meta: DatasetMeta;
  data?: any[];
  summary?: any;
  onNavigate?: (tabId: string) => void;
  onSelectTab?: (tabId: string) => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({ meta, onSelectTab, onNavigate }) => {
  const handleNav = onNavigate || onSelectTab || (() => {});
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // Pipeline flowchart nodes
  const pipelineNodes = [
    {
      id: 'eda',
      title: '1. Ingest & EDA',
      desc: 'Load CSV, check schema & distribution',
      icon: Database,
      tab: 'eda',
      accent: 'border-blue-500/40 text-blue-600 dark:text-blue-400'
    },
    {
      id: 'cleaning',
      title: '2. Clean & Impute',
      desc: 'IQR outlier caps & median/mode filling',
      icon: Filter,
      tab: 'cleaning',
      accent: 'border-cyan-500/40 text-cyan-600 dark:text-cyan-400'
    },
    {
      id: 'split',
      title: '3. Stratified Split',
      desc: '80% Train / 20% Test, preserved ratios',
      icon: Sliders,
      tab: 'cleaning',
      accent: 'border-indigo-500/40 text-indigo-600 dark:text-indigo-400'
    },
    {
      id: 'scaling',
      title: '4. Fit Scaler (Train Only)',
      desc: 'Standardize features without data leakage',
      icon: CheckCircle2,
      tab: 'cleaning',
      accent: 'border-emerald-500/40 text-emerald-600 dark:text-emerald-400'
    },
    {
      id: 'training',
      title: '5. Train Scratch ML',
      desc: 'Logistic Regression & CART Decision Tree',
      icon: Cpu,
      tab: 'training',
      accent: 'border-purple-500/40 text-purple-600 dark:text-purple-400'
    },
    {
      id: 'evaluation',
      title: '6. Evaluate & ROC',
      desc: 'Clinical Recall, Confusion Matrix & AUC',
      icon: BarChart3,
      tab: 'evaluation',
      accent: 'border-rose-500/40 text-rose-600 dark:text-rose-400'
    },
    {
      id: 'predict',
      title: '7. Patient Inference',
      desc: 'Interactive clinical risk calculator',
      icon: Stethoscope,
      tab: 'predict',
      accent: 'border-amber-500/40 text-amber-600 dark:text-amber-400'
    }
  ];

  // Categories list
  const categories = [
    { id: 'all', label: 'All Categories' },
    { id: 'demographic', label: 'Demographics' },
    { id: 'symptom', label: 'Symptoms & History' },
    { id: 'vital_sign', label: 'Vital Signs' },
    { id: 'lab_test', label: 'Lab Tests' },
    { id: 'ecg', label: 'Electrocardiogram (ECG)' },
    { id: 'angiography', label: 'Fluoroscopy & Imaging' }
  ];

  // Filtered dictionary
  const filteredFeatures = useMemo(() => {
    return HEART_DATA_DICTIONARY.filter((f: any) => {
      const matchCategory = selectedCategory === 'all' || f.category === selectedCategory;
      const matchSearch =
        searchTerm === '' ||
        f.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (f.fullName && f.fullName.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (f.label && f.label.toLowerCase().includes(searchTerm.toLowerCase())) ||
        f.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        f.clinicalRelevance.toLowerCase().includes(searchTerm.toLowerCase());
      return matchCategory && matchSearch;
    });
  }, [selectedCategory, searchTerm]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Hero / Problem Statement Card */}
      <div className="relative overflow-hidden bg-gradient-to-br from-rose-500/10 via-slate-50 to-white dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 sm:p-8 shadow-sm">
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-100 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300 text-xs font-bold uppercase tracking-wider border border-rose-200 dark:border-rose-800">
            <Activity className="w-3.5 h-3.5" /> University Capstone: Week 4 Deliverable
          </div>

          <h2 className="text-2xl sm:text-4xl font-extrabold text-slate-900 dark:text-white tracking-tight">
            Heart Disease Prediction: End-to-End ML Pipeline
          </h2>

          <p className="text-slate-600 dark:text-slate-300 text-sm sm:text-base leading-relaxed">
            Cardiovascular diseases (CVDs) are the leading cause of death globally. Early triage can save lives.
            This single-page application implements an entire Machine Learning pipeline <strong>completely from scratch in TypeScript</strong>—without external ML libraries, Python servers, or cloud API keys.
            It performs exploratory data analysis, leak-free data preprocessing, Batch Gradient Descent Logistic Regression, CART Decision Tree training, clinical evaluation, and patient risk simulation.
          </p>

          <div className="pt-2 flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400">
            <div className="flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              <span>Strict Leakage Prevention (Train-Only Fit)</span>
            </div>
            <div className="flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              <span>Clinical Asymmetry: Prioritizing Recall</span>
            </div>
            <div className="flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              <span>100% In-Browser & Fully Offline</span>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Pipeline Diagram */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <span>Interactive Pipeline Architecture</span>
              <span className="text-xs font-normal text-slate-500">(Click any node to jump to that stage)</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              End-to-end data flow from raw CSV ingestion through zero-leakage training to patient diagnosis.
            </p>
          </div>
        </div>

        {/* Nodes Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {pipelineNodes.map((node, index) => {
            const Icon = node.icon;
            return (
              <button
                key={node.id}
                onClick={() => handleNav(node.tab)}
                className={`group relative p-3 rounded-xl border bg-slate-50/50 dark:bg-slate-800/40 hover:bg-white dark:hover:bg-slate-800 hover:shadow-md transition-all text-left flex flex-col justify-between cursor-pointer ${node.accent}`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="p-2 rounded-lg bg-white dark:bg-slate-700 shadow-xs group-hover:scale-110 transition-transform">
                      <Icon className="w-4 h-4" />
                    </div>
                    {index < pipelineNodes.length - 1 && (
                      <ArrowRight className="w-3.5 h-3.5 text-slate-300 dark:text-slate-600 hidden lg:block" />
                    )}
                  </div>
                  <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 leading-tight">
                    {node.title}
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                    {node.desc}
                  </p>
                </div>
                <span className="text-[10px] font-semibold text-rose-600 dark:text-rose-400 opacity-0 group-hover:opacity-100 transition-opacity mt-2 flex items-center gap-1">
                  View step &rarr;
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Clinical Data Dictionary */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <FileSpreadsheet className="w-4 h-4 text-rose-500" />
              <span>Clinical Data Dictionary & Feature Semantics</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Detailed clinical background, units, normal physiological ranges, and cardiac relevance for each biomarker.
            </p>
          </div>

          {/* Search & Filter Controls */}
          <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
            {/* Search */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search biomarker..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="w-full sm:w-48 pl-8 pr-3 py-1.5 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              />
            </div>

            {/* Category Filter */}
            <select
              value={selectedCategory}
              onChange={e => setSelectedCategory(e.target.value)}
              className="py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
            >
              {categories.map(c => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
          <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
            <thead className="bg-slate-50 dark:bg-slate-800/80 text-slate-500 dark:text-slate-400 font-semibold border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="px-3.5 py-2.5">Feature</th>
                <th className="px-3.5 py-2.5">Clinical Name</th>
                <th className="px-3.5 py-2.5">Type & Category</th>
                <th className="px-3.5 py-2.5">Normal / Values</th>
                <th className="px-3.5 py-2.5">Clinical Relevance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60 bg-white dark:bg-slate-900">
              {filteredFeatures.map((f: any) => (
                <tr key={f.name} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40 transition">
                  <td className="px-3.5 py-2.5 font-mono font-bold text-rose-600 dark:text-rose-400 whitespace-nowrap">
                    {f.name}
                  </td>
                  <td className="px-3.5 py-2.5 font-semibold text-slate-900 dark:text-white whitespace-nowrap">
                    {f.fullName || f.label}
                  </td>
                  <td className="px-3.5 py-2.5 whitespace-nowrap">
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                      {f.type} • {f.category}
                    </span>
                  </td>
                  <td className="px-3.5 py-2.5 whitespace-nowrap font-mono text-[11px] text-slate-600 dark:text-slate-400">
                    {f.normalRange} {f.unit ? `(${f.unit})` : ''}
                  </td>
                  <td className="px-3.5 py-2.5 text-slate-600 dark:text-slate-300 max-w-md leading-relaxed">
                    {f.clinicalRelevance}
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
