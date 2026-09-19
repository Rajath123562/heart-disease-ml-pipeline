import React, { useState, useMemo } from 'react';
import {
  Stethoscope,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  Sparkles,
  Info,
  ArrowRight,
  ShieldAlert,
  GitFork,
  Activity
} from 'lucide-react';
import {
  LogisticModelArtifact,
  DecisionTreeModelArtifact,
  PreprocessingSummary,
  PreprocessingConfig
} from '../types';
import { sigmoid } from '../ml/logistic';
import { predictSampleProbability } from '../ml/decisionTree';

interface PredictTabProps {
  logisticModel: LogisticModelArtifact | null;
  treeModel: DecisionTreeModelArtifact | null;
  summary: PreprocessingSummary | null;
  config: PreprocessingConfig;
  threshold: number;
}

export const PredictTab: React.FC<PredictTabProps> = ({
  logisticModel,
  treeModel,
  summary,
  config,
  threshold
}) => {
  // 13 clinical biomarkers state
  const [patientData, setPatientData] = useState<Record<string, number>>({
    age: 54,
    sex: 1, // 1=male, 0=female
    cp: 0, // 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
    trestbps: 130,
    chol: 245,
    fbs: 0,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 1.0,
    slope: 1,
    ca: 0,
    thal: 2
  });

  const [activeModel, setActiveModel] = useState<'both' | 'logistic' | 'tree'>('both');

  // Preset Patient Profiles
  const presets = [
    {
      name: 'Low-Risk Healthy Patient',
      desc: 'Younger adult, high exercise capacity, no angina or ST depression',
      data: {
        age: 42,
        sex: 0,
        cp: 1,
        trestbps: 115,
        chol: 185,
        fbs: 0,
        restecg: 0,
        thalach: 175,
        exang: 0,
        oldpeak: 0.0,
        slope: 2,
        ca: 0,
        thal: 2
      }
    },
    {
      name: 'High-Risk Patient with Angina',
      desc: 'Older adult with exercise angina, significant ST depression, and arterial stenosis',
      data: {
        age: 64,
        sex: 1,
        cp: 0,
        trestbps: 155,
        chol: 295,
        fbs: 1,
        restecg: 1,
        thalach: 112,
        exang: 1,
        oldpeak: 2.8,
        slope: 1,
        ca: 2,
        thal: 3
      }
    },
    {
      name: 'Borderline Patient',
      desc: 'Intermediate age with moderate cholesterol and mild exercise response',
      data: {
        age: 55,
        sex: 1,
        cp: 2,
        trestbps: 138,
        chol: 242,
        fbs: 0,
        restecg: 0,
        thalach: 142,
        exang: 0,
        oldpeak: 1.2,
        slope: 1,
        ca: 1,
        thal: 2
      }
    }
  ];

  const handleApplyPreset = (presetData: Record<string, number>) => {
    setPatientData({ ...presetData });
  };

  const handleInputChange = (field: string, value: number) => {
    setPatientData(prev => ({ ...prev, [field]: value }));
  };

  // Preprocess patient vector using frozen pipeline parameters
  const { scaledVector, rawVector, featureContributions } = useMemo(() => {
    if (!summary || !logisticModel) {
      return { scaledVector: [], rawVector: [], featureContributions: [] };
    }

    const rawVec: number[] = [];
    const scaledVec: number[] = [];

    // Helper: encode / scale according to summary feature order
    for (const fName of summary.featureNames) {
      // Check if one-hot column like cp_1
      if (fName.includes('_')) {
        const [base, catVal] = fName.split('_');
        const currVal = patientData[base];
        const isMatch = currVal === Number(catVal) ? 1 : 0;
        rawVec.push(isMatch);
        scaledVec.push(isMatch);
      } else {
        const rawVal = patientData[fName] ?? 0;
        rawVec.push(rawVal);

        // Apply training scaler parameters
        const means = summary.scalerParams.means || summary.scalerParams.mean || {};
        const stds = summary.scalerParams.stds || summary.scalerParams.std || {};
        const mins = summary.scalerParams.mins || summary.scalerParams.min || {};
        const maxs = summary.scalerParams.maxs || summary.scalerParams.max || {};

        const mean = means[fName] ?? 0;
        const std = stds[fName] || 1;
        const min = mins[fName] ?? 0;
        const max = maxs[fName] || 1;

        if (config.scalingMethod === 'standard') {
          scaledVec.push((rawVal - mean) / std);
        } else if (config.scalingMethod === 'minmax') {
          scaledVec.push((rawVal - min) / (max - min || 1));
        } else {
          scaledVec.push(rawVal);
        }
      }
    }

    // Compute feature contributions for Logistic Regression
    const contributions = summary.featureNames.map((name, i) => {
      const w = logisticModel.weights[i] ?? 0;
      const x = scaledVec[i] ?? 0;
      const impact = w * x;
      return {
        feature: name,
        impact,
        absImpact: Math.abs(impact),
        direction: impact > 0 ? 'Risk Driver' : 'Protective'
      };
    }).sort((a, b) => b.absImpact - a.absImpact);

    return {
      scaledVector: scaledVec,
      rawVector: rawVec,
      featureContributions: contributions
    };
  }, [patientData, summary, logisticModel, config]);

  // Logistic Regression Prediction
  const logisticPrediction = useMemo(() => {
    if (!logisticModel || scaledVector.length === 0) return null;

    let z = logisticModel.bias;
    for (let i = 0; i < scaledVector.length; i++) {
      z += scaledVector[i] * (logisticModel.weights[i] ?? 0);
    }

    const prob = sigmoid(z);
    const isDisease = prob >= threshold;

    return {
      prob: Number(prob.toFixed(4)),
      pct: (prob * 100).toFixed(1),
      isDisease,
      riskLevel: prob >= 0.7 ? 'High Risk' : prob >= threshold ? 'Moderate Risk' : 'Low Risk'
    };
  }, [logisticModel, scaledVector, threshold]);

  // Decision Tree Prediction
  const treePrediction = useMemo(() => {
    if (!treeModel || rawVector.length === 0) return null;

    const { prob, path } = predictSampleProbability(treeModel.root, rawVector);
    const isDisease = prob >= threshold;

    return {
      prob: Number(prob.toFixed(4)),
      pct: (prob * 100).toFixed(1),
      isDisease,
      path,
      riskLevel: prob >= 0.7 ? 'High Risk' : prob >= threshold ? 'Moderate Risk' : 'Low Risk'
    };
  }, [treeModel, rawVector, threshold]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Prominent Medical Disclaimer Banner */}
      <div className="p-4 bg-amber-500/15 border border-amber-500/40 rounded-2xl flex items-start gap-3 text-amber-800 dark:text-amber-300 text-xs sm:text-sm">
        <ShieldAlert className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold uppercase tracking-wider block text-xs mb-0.5 text-amber-900 dark:text-amber-200">
            Regulatory Medical Disclaimer
          </span>
          This tool is an educational machine learning demonstration engineered for a university capstone project.
          It is <strong>not an FDA-cleared diagnostic medical device</strong> and must not be used to direct clinical patient care, prescribe pharmaceuticals, or substitute for professional physician evaluation.
        </div>
      </div>

      {/* Quick Presets Bar */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-rose-500" />
            <span>Clinical Vignette Presets</span>
          </h3>
          <span className="text-[11px] text-slate-400">Click to autofill patient profile</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {presets.map(p => (
            <button
              key={p.name}
              onClick={() => handleApplyPreset(p.data)}
              className="p-3 text-left rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-850 hover:border-rose-300 dark:hover:border-rose-700 transition cursor-pointer group"
            >
              <div className="text-xs font-bold text-slate-900 dark:text-white group-hover:text-rose-600 dark:group-hover:text-rose-400 flex items-center justify-between">
                <span>{p.name}</span>
                <ArrowRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                {p.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Form Inputs & Prediction Outcome */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: 13 Biomarkers Patient Input Form */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-rose-500" />
                <span>Patient Clinical Profile (13 Parameters)</span>
              </h3>
              <p className="text-xs text-slate-500">Modify physiological values to run live risk scoring</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Age */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Age (years)</span>
                <span className="font-mono font-bold text-slate-900 dark:text-white">{patientData.age}</span>
              </div>
              <input
                type="range"
                min={25}
                max={85}
                step={1}
                value={patientData.age}
                onChange={e => handleInputChange('age', parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Sex */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Biological Sex
              </label>
              <select
                value={patientData.sex}
                onChange={e => handleInputChange('sex', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={1}>Male (1)</option>
                <option value={0}>Female (0)</option>
              </select>
            </div>

            {/* Chest Pain Type (cp) */}
            <div className="sm:col-span-2">
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Chest Pain Type (cp)
              </label>
              <select
                value={patientData.cp}
                onChange={e => handleInputChange('cp', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={0}>Typical Angina (0) - Exertional substernal pressure</option>
                <option value={1}>Atypical Angina (1) - Non-classical pain duration</option>
                <option value={2}>Non-Anginal Pain (2) - Sharp or pleuritic pain</option>
                <option value={3}>Asymptomatic (3) - High correlation with advanced plaque</option>
              </select>
            </div>

            {/* Resting Blood Pressure (trestbps) */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Resting BP (mm Hg)</span>
                <span className="font-mono font-bold text-slate-900 dark:text-white">{patientData.trestbps}</span>
              </div>
              <input
                type="range"
                min={90}
                max={200}
                step={2}
                value={patientData.trestbps}
                onChange={e => handleInputChange('trestbps', parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Serum Cholesterol (chol) */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Cholesterol (mg/dl)</span>
                <span className="font-mono font-bold text-slate-900 dark:text-white">{patientData.chol}</span>
              </div>
              <input
                type="range"
                min={120}
                max={400}
                step={5}
                value={patientData.chol}
                onChange={e => handleInputChange('chol', parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Fasting Blood Sugar (fbs) */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Fasting Blood Sugar &gt; 120 mg/dl
              </label>
              <select
                value={patientData.fbs}
                onChange={e => handleInputChange('fbs', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={0}>No (0) &le; 120 mg/dl</option>
                <option value={1}>Yes (1) &gt; 120 mg/dl (Diabetic / Prediabetic)</option>
              </select>
            </div>

            {/* Resting ECG (restecg) */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Resting Electrocardiogram (restecg)
              </label>
              <select
                value={patientData.restecg}
                onChange={e => handleInputChange('restecg', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={0}>Normal (0)</option>
                <option value={1}>ST-T wave abnormality (1)</option>
                <option value={2}>Left ventricular hypertrophy (2)</option>
              </select>
            </div>

            {/* Max Heart Rate (thalach) */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Max Heart Rate (bpm)</span>
                <span className="font-mono font-bold text-slate-900 dark:text-white">{patientData.thalach}</span>
              </div>
              <input
                type="range"
                min={70}
                max={210}
                step={2}
                value={patientData.thalach}
                onChange={e => handleInputChange('thalach', parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Exercise-Induced Angina (exang) */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Exercise-Induced Angina (exang)
              </label>
              <select
                value={patientData.exang}
                onChange={e => handleInputChange('exang', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={0}>No (0)</option>
                <option value={1}>Yes (1) - Exercise triggers angina</option>
              </select>
            </div>

            {/* ST Depression (oldpeak) */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">ST Depression (mm)</span>
                <span className="font-mono font-bold text-slate-900 dark:text-white">{patientData.oldpeak.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={0.0}
                max={6.0}
                step={0.1}
                value={patientData.oldpeak}
                onChange={e => handleInputChange('oldpeak', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* ST Slope (slope) */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Peak Exercise ST Slope
              </label>
              <select
                value={patientData.slope}
                onChange={e => handleInputChange('slope', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={0}>Upsloping (0) - Favorable</option>
                <option value={1}>Flat (1) - Equivocal / Ischemia risk</option>
                <option value={2}>Downsloping (2) - Severe subendocardial ischemia</option>
              </select>
            </div>

            {/* Major Fluoroscopy Vessels (ca) */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Stenosed Vessels (ca: 0-3)</span>
                <span className="font-mono font-bold text-slate-900 dark:text-white">{patientData.ca}</span>
              </div>
              <input
                type="range"
                min={0}
                max={3}
                step={1}
                value={patientData.ca}
                onChange={e => handleInputChange('ca', parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Thallium Stress Defect (thal) */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Thallium Scintigraphy (thal)
              </label>
              <select
                value={patientData.thal}
                onChange={e => handleInputChange('thal', parseInt(e.target.value))}
                className="w-full py-1.5 px-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <option value={1}>Fixed Defect (1) - Infarcted non-viable tissue</option>
                <option value={2}>Normal Blood Flow (2)</option>
                <option value={3}>Reversible Defect (3) - Severe viable ischemia</option>
              </select>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Score Cards & Feature Contribution */}
        <div className="lg:col-span-5 space-y-6">
          {/* Model Selector Pill */}
          <div className="flex rounded-xl bg-slate-100 dark:bg-slate-800 p-1 text-xs font-semibold">
            <button
              onClick={() => setActiveModel('both')}
              className={`flex-1 py-1.5 rounded-lg transition ${
                activeModel === 'both'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-xs'
                  : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveModel('logistic')}
              className={`flex-1 py-1.5 rounded-lg transition ${
                activeModel === 'logistic'
                  ? 'bg-white dark:bg-slate-700 text-rose-600 dark:text-rose-400 shadow-xs'
                  : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              Logistic
            </button>
            <button
              onClick={() => setActiveModel('tree')}
              className={`flex-1 py-1.5 rounded-lg transition ${
                activeModel === 'tree'
                  ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-xs'
                  : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              Decision Tree
            </button>
          </div>

          {/* Logistic Output Card */}
          {(activeModel === 'both' || activeModel === 'logistic') && logisticPrediction && (
            <div
              className={`p-6 rounded-2xl border shadow-sm transition-all ${
                logisticPrediction.isDisease
                  ? 'bg-gradient-to-br from-rose-500/10 via-rose-50/50 to-white dark:from-rose-950/30 dark:via-slate-900 dark:to-slate-900 border-rose-300 dark:border-rose-800'
                  : 'bg-gradient-to-br from-emerald-500/10 via-emerald-50/50 to-white dark:from-emerald-950/30 dark:via-slate-900 dark:to-slate-900 border-emerald-300 dark:border-emerald-800'
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold uppercase tracking-wider text-slate-500">
                  Logistic Regression
                </span>
                <span
                  className={`px-2.5 py-0.5 rounded-full font-extrabold uppercase text-[10px] ${
                    logisticPrediction.isDisease
                      ? 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300'
                      : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                  }`}
                >
                  {logisticPrediction.riskLevel}
                </span>
              </div>

              <div className="my-3">
                <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white flex items-center gap-2">
                  {logisticPrediction.isDisease ? (
                    <>
                      <AlertTriangle className="w-6 h-6 text-rose-600 dark:text-rose-400 shrink-0" />
                      <span>Heart Disease Likely</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400 shrink-0" />
                      <span>Low Cardiac Risk</span>
                    </>
                  )}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Threshold cutoff: &tau; &ge; {threshold.toFixed(2)}
                </div>
              </div>

              {/* Risk Gauge Bar */}
              <div className="space-y-1.5 pt-2">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-slate-600 dark:text-slate-400">Posterior Probability:</span>
                  <span className="font-mono font-extrabold text-slate-900 dark:text-white">
                    {logisticPrediction.pct}%
                  </span>
                </div>
                <div className="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      logisticPrediction.isDisease ? 'bg-rose-600' : 'bg-emerald-500'
                    }`}
                    style={{ width: `${logisticPrediction.pct}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Tree Output Card */}
          {(activeModel === 'both' || activeModel === 'tree') && treePrediction && (
            <div
              className={`p-6 rounded-2xl border shadow-sm transition-all ${
                treePrediction.isDisease
                  ? 'bg-gradient-to-br from-indigo-500/10 via-indigo-50/50 to-white dark:from-indigo-950/30 dark:via-slate-900 dark:to-slate-900 border-indigo-300 dark:border-indigo-800'
                  : 'bg-gradient-to-br from-emerald-500/10 via-emerald-50/50 to-white dark:from-emerald-950/30 dark:via-slate-900 dark:to-slate-900 border-emerald-300 dark:border-emerald-800'
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold uppercase tracking-wider text-slate-500">
                  Decision Tree (CART)
                </span>
                <span
                  className={`px-2.5 py-0.5 rounded-full font-extrabold uppercase text-[10px] ${
                    treePrediction.isDisease
                      ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300'
                      : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                  }`}
                >
                  {treePrediction.riskLevel}
                </span>
              </div>

              <div className="my-3">
                <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white flex items-center gap-2">
                  {treePrediction.isDisease ? (
                    <>
                      <AlertTriangle className="w-6 h-6 text-indigo-600 dark:text-indigo-400 shrink-0" />
                      <span>Heart Disease Likely</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400 shrink-0" />
                      <span>Low Cardiac Risk</span>
                    </>
                  )}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Leaf distribution probability: {treePrediction.pct}%
                </div>
              </div>

              {/* Decision Path Trace */}
              {treePrediction.path.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-200/60 dark:border-slate-800">
                  <div className="text-[11px] font-bold text-slate-700 dark:text-slate-300 mb-1 flex items-center gap-1">
                    <GitFork className="w-3 h-3" /> Decision Tree Path:
                  </div>
                  <div className="space-y-1 text-[10px] font-mono text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/80 p-2.5 rounded-lg">
                    {treePrediction.path.map((step, idx) => (
                      <div key={idx} className="flex items-center gap-1.5 truncate">
                        <span className="text-indigo-500 font-bold">&rarr;</span>
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Top Contributing Biomarkers for this Patient */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-rose-500" />
              <span>Primary Risk Drivers for this Patient</span>
            </h4>

            <div className="space-y-2 text-xs">
              {featureContributions.slice(0, 4).map(fc => (
                <div
                  key={fc.feature}
                  className="flex items-center justify-between p-2 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200/60 dark:border-slate-700/60"
                >
                  <span className="font-bold text-slate-800 dark:text-slate-200 uppercase">
                    {fc.feature}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">Impact: {fc.impact.toFixed(2)}</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        fc.impact > 0
                          ? 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300'
                          : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                      }`}
                    >
                      {fc.direction}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
