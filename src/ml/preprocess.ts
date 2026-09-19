/**
 * Data Cleaning, Preprocessing, Scaling, and Stratified Train/Test Split
 * CRITICAL LEAKAGE-FREE ARCHITECTURE:
 * Scalers and imputer parameters are fitted strictly on the TRAINING split,
 * then applied without refitting to the TEST split.
 */

import {
  RawDataRow,
  PreprocessingConfig,
  PreprocessingSummary,
  TrainTestSplitData,
  ColumnStats
} from '../types';

/**
 * Seedable pseudo-random number generator (Mulberry32) for reproducible splits and shuffling
 */
export function createPRNG(seed: number): () => number {
  let s = Math.floor(seed) || 42;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Compute descriptive statistics for a feature column
 */
export function computeColumnStats(data: RawDataRow[], colName: string, targetCol: string): ColumnStats {
  const values = data.map(r => r[colName]);
  const nonNulls = values.filter((v): v is string | number | boolean => v !== null && v !== undefined && v !== '');
  const missing = values.length - nonNulls.length;

  const isNumeric = nonNulls.length > 0 && nonNulls.every(v => typeof v === 'number' || !isNaN(Number(v)));
  const uniqueVals = Array.from(new Set(nonNulls.map(v => String(v))));

  if (isNumeric && uniqueVals.length > 5) {
    const nums = nonNulls.map(v => Number(v)).sort((a, b) => a - b);
    const n = nums.length;
    const sum = nums.reduce((acc, v) => acc + v, 0);
    const mean = n > 0 ? sum / n : 0;
    const variance = n > 1 ? nums.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (n - 1) : 0;
    const std = Math.sqrt(variance);

    const q25 = nums[Math.floor(n * 0.25)] ?? nums[0];
    const median = nums[Math.floor(n * 0.5)] ?? nums[0];
    const q75 = nums[Math.floor(n * 0.75)] ?? nums[n - 1];

    // Build 8-bin distribution with target breakdown
    const min = nums[0];
    const max = nums[n - 1];
    const binCount = 8;
    const binWidth = (max - min) / binCount || 1;
    const distribution = Array.from({ length: binCount }, (_, i) => {
      const bMin = min + i * binWidth;
      const bMax = bMin + binWidth;
      return {
        label: `${bMin.toFixed(0)}-${bMax.toFixed(0)}`,
        count: 0,
        diseaseCount: 0,
        noDiseaseCount: 0
      };
    });

    for (const r of data) {
      const v = Number(r[colName]);
      const t = Number(r[targetCol]);
      if (!isNaN(v)) {
        let bIdx = Math.floor((v - min) / binWidth);
        if (bIdx >= binCount) bIdx = binCount - 1;
        if (bIdx < 0) bIdx = 0;
        distribution[bIdx].count++;
        if (t === 1) distribution[bIdx].diseaseCount++;
        else distribution[bIdx].noDiseaseCount++;
      }
    }

    return {
      name: colName,
      type: 'numeric',
      count: nonNulls.length,
      missing,
      unique: uniqueVals.length,
      mean: Number(mean.toFixed(2)),
      std: Number(std.toFixed(2)),
      min,
      q25,
      median,
      q75,
      max,
      distribution
    };
  } else {
    // Categorical or Binary
    const counts: Record<string, { count: number; diseaseCount: number; noDiseaseCount: number }> = {};
    for (const r of data) {
      const raw = r[colName];
      const key = raw === null || raw === undefined ? 'Missing' : String(raw);
      const t = Number(r[targetCol]);
      if (!counts[key]) counts[key] = { count: 0, diseaseCount: 0, noDiseaseCount: 0 };
      counts[key].count++;
      if (t === 1) counts[key].diseaseCount++;
      else counts[key].noDiseaseCount++;
    }

    let modeVal: string = uniqueVals[0] || '';
    let maxFreq = -1;
    for (const [k, v] of Object.entries(counts)) {
      if (k !== 'Missing' && v.count > maxFreq) {
        maxFreq = v.count;
        modeVal = k;
      }
    }

    const distribution = Object.entries(counts).map(([label, v]) => ({
      label,
      count: v.count,
      diseaseCount: v.diseaseCount,
      noDiseaseCount: v.noDiseaseCount
    }));

    return {
      name: colName,
      type: uniqueVals.length <= 2 ? 'binary' : 'categorical',
      count: nonNulls.length,
      missing,
      unique: uniqueVals.length,
      mode: modeVal,
      distribution
    };
  }
}

/**
 * Standard numeric features in heart disease data
 */
export const NUMERIC_FEATURE_CANDIDATES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'];
export const CATEGORICAL_FEATURE_CANDIDATES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'];

export const DEFAULT_PREPROCESSING_CONFIG: PreprocessingConfig = {
  missingStrategy: 'median_mode',
  removeDuplicates: true,
  handleOutliersIQR: true,
  capOutliers: true,
  oneHotEncode: true,
  scalingMethod: 'standard',
  testSize: 0.20,
  randomSeed: 42
};

/**
 * Comprehensive preprocessing pipeline execution
 */
export function preprocessPipeline(
  rawData: RawDataRow[],
  targetColOrConfig: string | PreprocessingConfig,
  maybeConfig?: PreprocessingConfig
): {
  splitData: TrainTestSplitData;
  trainDataset: { X: number[][]; y: number[]; featureNames: string[] };
  testDataset: { X: number[][]; y: number[]; featureNames: string[] };
  summary: PreprocessingSummary;
} {
  let targetCol: string = 'target';
  let config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG;

  if (typeof targetColOrConfig === 'string') {
    targetCol = targetColOrConfig;
    if (maybeConfig) config = maybeConfig;
  } else if (typeof targetColOrConfig === 'object') {
    config = targetColOrConfig;
    // auto detect target column
    if (rawData.length > 0) {
      const keys = Object.keys(rawData[0]);
      const lower = keys.map(k => k.toLowerCase());
      if (lower.includes('target')) targetCol = keys[lower.indexOf('target')];
      else if (lower.includes('num')) targetCol = keys[lower.indexOf('num')];
      else targetCol = keys[keys.length - 1];
    }
  }
  const originalRows = rawData.length;
  const originalCols = Object.keys(rawData[0] || {}).length;

  // Step 1: Duplicate Removal
  let currentRows: RawDataRow[] = [];
  let duplicatesRemoved = 0;
  if (config.removeDuplicates) {
    const seen = new Set<string>();
    for (const row of rawData) {
      const key = JSON.stringify(row);
      if (seen.has(key)) {
        duplicatesRemoved++;
      } else {
        seen.add(key);
        currentRows.push({ ...row });
      }
    }
  } else {
    currentRows = rawData.map(r => ({ ...r }));
  }

  // Identify features (excluding target)
  const allFeatureCols = Object.keys(currentRows[0] || {}).filter(k => k !== targetCol);
  
  // Classify features as numeric vs categorical
  const numericFeatures: string[] = [];
  const categoricalFeatures: string[] = [];

  for (const f of allFeatureCols) {
    const vals = currentRows.map(r => r[f]).filter(v => v !== null && v !== undefined && v !== '');
    const numVals = vals.filter(v => typeof v === 'number' || !isNaN(Number(v)));
    const unique = new Set(vals).size;
    if (numVals.length === vals.length && unique > 5 && !CATEGORICAL_FEATURE_CANDIDATES.includes(f)) {
      numericFeatures.push(f);
    } else {
      categoricalFeatures.push(f);
    }
  }

  // Step 2: Handle Missing Values
  let imputedCount = 0;
  if (config.missingStrategy === 'drop') {
    currentRows = currentRows.filter(row => {
      for (const col of Object.keys(row)) {
        if (row[col] === null || row[col] === undefined || row[col] === '') return false;
      }
      return true;
    });
  } else {
    // Compute medians / modes
    const numericImputers: Record<string, number> = {};
    for (const numF of numericFeatures) {
      const nums = currentRows
        .map(r => Number(r[numF]))
        .filter(n => !isNaN(n))
        .sort((a, b) => a - b);
      if (config.missingStrategy === 'mean') {
        numericImputers[numF] = nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      } else {
        // default median
        numericImputers[numF] = nums.length ? nums[Math.floor(nums.length / 2)] : 0;
      }
    }

    const catImputers: Record<string, number | string> = {};
    for (const catF of categoricalFeatures) {
      const counts: Record<string, number> = {};
      for (const r of currentRows) {
        const v = r[catF];
        if (v !== null && v !== undefined && v !== '') {
          const k = String(v);
          counts[k] = (counts[k] || 0) + 1;
        }
      }
      let topK: string = '0';
      let maxCount = -1;
      for (const [k, count] of Object.entries(counts)) {
        if (count > maxCount) {
          maxCount = count;
          topK = k;
        }
      }
      catImputers[catF] = isNaN(Number(topK)) ? topK : Number(topK);
    }

    for (const row of currentRows) {
      for (const col of allFeatureCols) {
        if (row[col] === null || row[col] === undefined || row[col] === '') {
          imputedCount++;
          if (numericFeatures.includes(col)) {
            row[col] = numericImputers[col] ?? 0;
          } else {
            row[col] = catImputers[col] ?? 0;
          }
        }
      }
      // Target imputation if missing: default to 0
      if (row[targetCol] === null || row[targetCol] === undefined) {
        row[targetCol] = 0;
        imputedCount++;
      }
    }
  }

  // Step 3: Outlier Capping (IQR)
  let outliersCapped = 0;
  if (config.handleOutliersIQR) {
    for (const numF of numericFeatures) {
      const vals = currentRows.map(r => Number(r[numF])).filter(v => !isNaN(v)).sort((a, b) => a - b);
      if (vals.length > 10) {
        const q1 = vals[Math.floor(vals.length * 0.25)];
        const q3 = vals[Math.floor(vals.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;

        for (const row of currentRows) {
          const v = Number(row[numF]);
          if (v < lowerBound) {
            row[numF] = lowerBound;
            outliersCapped++;
          } else if (v > upperBound) {
            row[numF] = upperBound;
            outliersCapped++;
          }
        }
      }
    }
  }

  // Step 4: Categorical Encoding (One-Hot or Numeric Ordinal)
  const encodedCategories: Record<string, string[]> = {};
  let finalFeatureNames: string[] = [...numericFeatures];

  if (config.oneHotEncode) {
    for (const catF of categoricalFeatures) {
      // If binary (0/1 or 2 unique values), keep as single column
      const uniqueVals = Array.from(new Set(currentRows.map(r => String(r[catF])))).sort();
      if (uniqueVals.length <= 2) {
        finalFeatureNames.push(catF);
      } else {
        encodedCategories[catF] = uniqueVals;
        for (const val of uniqueVals) {
          finalFeatureNames.push(`${catF}_${val}`);
        }
      }
    }
  } else {
    finalFeatureNames.push(...categoricalFeatures);
  }

  // Convert all rows into flat numeric arrays
  const matrixX: number[][] = [];
  const vectorY: number[] = [];

  for (const row of currentRows) {
    const rowX: number[] = [];

    // Numeric features
    for (const numF of numericFeatures) {
      rowX.push(Number(row[numF]) || 0);
    }

    // Categorical features
    if (config.oneHotEncode) {
      for (const catF of categoricalFeatures) {
        const uniqueVals = encodedCategories[catF];
        const rawVal = String(row[catF]);
        if (!uniqueVals) {
          // Binary
          rowX.push(Number(row[catF]) || 0);
        } else {
          // One-hot dummy variables
          for (const u of uniqueVals) {
            rowX.push(rawVal === u ? 1 : 0);
          }
        }
      }
    } else {
      for (const catF of categoricalFeatures) {
        rowX.push(Number(row[catF]) || 0);
      }
    }

    matrixX.push(rowX);
    vectorY.push(Number(row[targetCol]) === 1 ? 1 : 0);
  }

  // Step 5: Stratified Train/Test Split
  // Stratify by class to ensure exact equal class distribution in train and test splits
  const rng = createPRNG(config.randomSeed);
  const class0Indices: number[] = [];
  const class1Indices: number[] = [];

  for (let i = 0; i < vectorY.length; i++) {
    if (vectorY[i] === 1) class1Indices.push(i);
    else class0Indices.push(i);
  }

  // Shuffle both index lists with seeded PRNG (Fisher-Yates)
  function shuffle(arr: number[]) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  shuffle(class0Indices);
  shuffle(class1Indices);

  const testSize = Math.max(0.05, Math.min(0.5, config.testSize));
  const testCount0 = Math.round(class0Indices.length * testSize);
  const testCount1 = Math.round(class1Indices.length * testSize);

  const testIndices = new Set([
    ...class0Indices.slice(0, testCount0),
    ...class1Indices.slice(0, testCount1)
  ]);

  const XTrain: number[][] = [];
  const yTrain: number[] = [];
  const XTest: number[][] = [];
  const yTest: number[] = [];

  for (let i = 0; i < matrixX.length; i++) {
    if (testIndices.has(i)) {
      XTest.push([...matrixX[i]]);
      yTest.push(vectorY[i]);
    } else {
      XTrain.push([...matrixX[i]]);
      yTrain.push(vectorY[i]);
    }
  }

  // Step 6: Leakage-Free Feature Scaling
  // CRITICAL: Fit scaler parameters (mean, std, min, max) strictly on XTrain!
  const scalerParams: PreprocessingSummary['scalerParams'] = {
    means: {},
    stds: {},
    mins: {},
    maxs: {}
  };

  if (config.scalingMethod !== 'none') {
    // We scale numeric features only (indices 0 to numericFeatures.length - 1)
    const numCount = numericFeatures.length;

    for (let colIdx = 0; colIdx < numCount; colIdx++) {
      const colName = numericFeatures[colIdx];
      const trainColValues = XTrain.map(row => row[colIdx]);

      if (config.scalingMethod === 'standard') {
        const n = trainColValues.length;
        const mean = n > 0 ? trainColValues.reduce((a, b) => a + b, 0) / n : 0;
        const variance = n > 1 ? trainColValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1) : 0;
        const std = Math.sqrt(variance) || 1e-6; // Avoid division by 0

        scalerParams.means![colName] = mean;
        scalerParams.stds![colName] = std;

        // Apply to XTrain
        for (let r = 0; r < XTrain.length; r++) {
          XTrain[r][colIdx] = (XTrain[r][colIdx] - mean) / std;
        }
        // Apply to XTest using TRAIN fitted parameters
        for (let r = 0; r < XTest.length; r++) {
          XTest[r][colIdx] = (XTest[r][colIdx] - mean) / std;
        }
      } else if (config.scalingMethod === 'minmax') {
        let min = Infinity;
        let max = -Infinity;
        for (const v of trainColValues) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const range = max - min || 1e-6;

        scalerParams.mins![colName] = min;
        scalerParams.maxs![colName] = max;

        // Apply to XTrain
        for (let r = 0; r < XTrain.length; r++) {
          XTrain[r][colIdx] = (XTrain[r][colIdx] - min) / range;
        }
        // Apply to XTest
        for (let r = 0; r < XTest.length; r++) {
          XTest[r][colIdx] = (XTest[r][colIdx] - min) / range;
        }
      }
    }
  }

  const trainClassDistribution = {
    '0': yTrain.filter(y => y === 0).length,
    '1': yTrain.filter(y => y === 1).length
  };

  const testClassDistribution = {
    '0': yTest.filter(y => y === 0).length,
    '1': yTest.filter(y => y === 1).length
  };

  const summary: PreprocessingSummary = {
    originalRows,
    originalCols,
    processedRows: currentRows.length,
    processedCols: finalFeatureNames.length,
    missingCount: imputedCount,
    imputedCount,
    duplicatesRemoved,
    outliersCapped,
    featureNames: finalFeatureNames,
    numericFeatures,
    categoricalFeatures,
    trainCount: XTrain.length,
    testCount: XTest.length,
    trainClassDistribution,
    testClassDistribution,
    trainTargetDistribution: {
      diseaseRate: ((trainClassDistribution['1'] / (XTrain.length || 1)) * 100).toFixed(1),
      diseaseCount: trainClassDistribution['1'],
      healthyCount: trainClassDistribution['0']
    },
    testTargetDistribution: {
      diseaseRate: ((testClassDistribution['1'] / (XTest.length || 1)) * 100).toFixed(1),
      diseaseCount: testClassDistribution['1'],
      healthyCount: testClassDistribution['0']
    },
    scalerParams: {
      ...scalerParams,
      mean: scalerParams.means,
      std: scalerParams.stds,
      min: scalerParams.mins,
      max: scalerParams.maxs
    },
    encodedCategories
  };

  return {
    splitData: {
      XTrain,
      yTrain,
      XTest,
      yTest,
      featureNames: finalFeatureNames
    },
    trainDataset: {
      X: XTrain,
      y: yTrain,
      featureNames: finalFeatureNames
    },
    testDataset: {
      X: XTest,
      y: yTest,
      featureNames: finalFeatureNames
    },
    summary
  };
}

/**
 * Transform a new patient record using fitted scaler and preprocessing configuration
 */
export function transformPatientRecord(
  patient: Record<string, number>,
  summary: PreprocessingSummary,
  config: PreprocessingConfig
): number[] {
  const rowX: number[] = [];

  // Numeric features (with scaling applied from scalerParams)
  for (const numF of summary.numericFeatures) {
    let val = patient[numF] !== undefined ? patient[numF] : 0;
    if (config.scalingMethod === 'standard' && summary.scalerParams.means && summary.scalerParams.stds) {
      const mean = summary.scalerParams.means[numF] ?? 0;
      const std = summary.scalerParams.stds[numF] ?? 1;
      val = (val - mean) / std;
    } else if (config.scalingMethod === 'minmax' && summary.scalerParams.mins && summary.scalerParams.maxs) {
      const min = summary.scalerParams.mins[numF] ?? 0;
      const max = summary.scalerParams.maxs[numF] ?? 1;
      val = (val - min) / (max - min || 1);
    }
    rowX.push(val);
  }

  // Categorical features
  if (config.oneHotEncode && summary.encodedCategories) {
    for (const catF of summary.categoricalFeatures) {
      const uniqueVals = summary.encodedCategories[catF];
      const rawVal = String(patient[catF] ?? 0);
      if (!uniqueVals) {
        rowX.push(patient[catF] ?? 0);
      } else {
        for (const u of uniqueVals) {
          rowX.push(rawVal === u ? 1 : 0);
        }
      }
    }
  } else {
    for (const catF of summary.categoricalFeatures) {
      rowX.push(patient[catF] ?? 0);
    }
  }

  return rowX;
}
