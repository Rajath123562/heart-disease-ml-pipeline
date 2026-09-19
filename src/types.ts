/**
 * Core type definitions for Heart Disease ML Pipeline
 */

export interface RawDataRow {
  [key: string]: string | number | boolean | null | undefined;
}

export interface DatasetMeta {
  name: string;
  source: 'uci_authentic' | 'uploaded' | 'synthetic';
  rowCount: number;
  columnCount: number;
  columns: string[];
  targetColumn: string;
  hasMissing: boolean;
  missingCount: number;
  duplicateCount: number;
  isSynthetic: boolean;
}

export interface ColumnStats {
  name: string;
  type: 'numeric' | 'categorical' | 'binary';
  count: number;
  missing: number;
  unique: number;
  mean?: number;
  std?: number;
  min?: number;
  q25?: number;
  median?: number;
  q75?: number;
  max?: number;
  mode?: string | number;
  distribution?: { label: string; count: number; diseaseCount: number; noDiseaseCount: number }[];
}

export interface PreprocessingConfig {
  missingStrategy: 'median_mode' | 'drop' | 'mean';
  removeDuplicates: boolean;
  handleOutliersIQR: boolean;
  capOutliers?: boolean;
  oneHotEncode: boolean;
  scalingMethod: 'standard' | 'minmax' | 'none';
  testSize: number; // e.g. 0.20
  randomSeed: number;
}

export interface PreprocessingSummary {
  originalRows: number;
  originalCols: number;
  processedRows: number;
  processedCols: number;
  missingCount: number;
  imputedCount: number;
  duplicatesRemoved: number;
  outliersCapped: number;
  featureNames: string[];
  numericFeatures: string[];
  categoricalFeatures: string[];
  trainCount: number;
  testCount: number;
  trainClassDistribution: { '0': number; '1': number };
  testClassDistribution: { '0': number; '1': number };
  trainTargetDistribution?: { diseaseRate: string; diseaseCount: number; healthyCount: number };
  testTargetDistribution?: { diseaseRate: string; diseaseCount: number; healthyCount: number };
  scalerParams: {
    means?: Record<string, number>;
    stds?: Record<string, number>;
    mins?: Record<string, number>;
    maxs?: Record<string, number>;
    mean?: Record<string, number>;
    std?: Record<string, number>;
    min?: Record<string, number>;
    max?: Record<string, number>;
  };
  encodedCategories?: Record<string, string[]>;
}

export interface TrainTestSplitData {
  XTrain: number[][];
  yTrain: number[];
  XTest: number[][];
  yTest: number[];
  featureNames: string[];
}

export interface ProcessedDataset {
  X: number[][];
  y: number[];
  featureNames: string[];
}

export interface LogisticHyperparams {
  learningRate: number;
  epochs: number;
  regularization: number; // L2 lambda
}

export interface LogisticModelArtifact {
  weights: number[];
  bias: number;
  lossHistory: { epoch: number; loss: number }[];
  trainTimeMs: number;
  featureNames: string[];
  hyperparams: LogisticHyperparams;
}

export interface DecisionTreeHyperparams {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  criterion: 'gini' | 'entropy';
}

export interface TreeNode {
  id: string;
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  isLeaf: boolean;
  predictedClass?: number;
  probability?: number;
  impurity: number;
  samples: number;
  value: [number, number]; // [count0, count1]
  depth: number;
}

export interface DecisionTreeModelArtifact {
  root: TreeNode;
  featureImportances: { feature: string; importance: number }[];
  trainTimeMs: number;
  hyperparams: DecisionTreeHyperparams;
  maxAchievedDepth: number;
  totalNodes: number;
  leafNodes: number;
}

export interface ConfusionMatrix {
  tp: number; // True Positive (Disease predicted as Disease)
  fp: number; // False Positive (Healthy predicted as Disease)
  tn: number; // True Negative (Healthy predicted as Healthy)
  fn: number; // False Negative (Disease missed as Healthy - CRITICAL)
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number; // Sensitivity
  specificity: number;
  f1Score: number;
  rocAuc: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: { fpr: number; tpr: number; threshold: number }[];
  prCurve: { precision: number; recall: number; threshold: number }[];
  trainAccuracy: number;
  classificationReport: {
    class0: { precision: number; recall: number; f1: number; support: number };
    class1: { precision: number; recall: number; f1: number; support: number };
    macroAvg: { precision: number; recall: number; f1: number; support: number };
    weightedAvg: { precision: number; recall: number; f1: number; support: number };
  };
}

export interface CrossValidationFoldResult {
  fold: number;
  accuracy: number;
  recall: number;
  rocAuc: number;
  f1Score: number;
}

export interface CrossValidationSummary {
  folds: CrossValidationFoldResult[];
  meanAccuracy: number;
  stdAccuracy: number;
  meanRecall: number;
  stdRecall: number;
  meanRocAuc: number;
  stdRocAuc: number;
  meanF1: number;
  stdF1: number;
}

export interface GridSearchResult {
  params: Record<string, number | string>;
  meanAccuracy: number;
  meanRecall: number;
  meanRocAuc: number;
  stdAccuracy: number;
}

export interface PatientInput {
  age: number;
  sex: number; // 1: male, 0: female
  cp: number; // 0: typical angina, 1: atypical, 2: non-anginal, 3: asymptomatic
  trestbps: number; // resting blood pressure
  chol: number; // serum cholesterol
  fbs: number; // 0 or 1
  restecg: number; // 0, 1, 2
  thalach: number; // max heart rate
  exang: number; // 0 or 1
  oldpeak: number; // ST depression
  slope: number; // 0: upsloping, 1: flat, 2: downsloping
  ca: number; // 0 to 3
  thal: number; // 0: normal, 1: fixed defect, 2: reversible
}

export interface PredictionOutput {
  modelName: 'Logistic Regression' | 'Decision Tree';
  probability: number; // 0.0 to 1.0
  predictedClass: 0 | 1;
  threshold: number;
  riskLevel: 'Low' | 'Moderate' | 'High';
  topFactors: { feature: string; contribution: string; impact: 'increases_risk' | 'decreases_risk' | 'neutral' }[];
  treeDecisionPath?: string[];
}

export interface ExportedModelBundle {
  version: string;
  timestamp: string;
  metadata: DatasetMeta;
  preprocessingConfig: PreprocessingConfig;
  preprocessingSummary: PreprocessingSummary;
  logisticModel: LogisticModelArtifact | null;
  decisionTreeModel: DecisionTreeModelArtifact | null;
  logisticMetrics: ModelMetrics | null;
  decisionTreeMetrics: ModelMetrics | null;
  threshold: number;
  models?: {
    logistic?: LogisticModelArtifact | null;
    tree?: DecisionTreeModelArtifact | null;
  };
  metrics?: {
    logistic?: ModelMetrics | null;
    tree?: ModelMetrics | null;
  };
}
