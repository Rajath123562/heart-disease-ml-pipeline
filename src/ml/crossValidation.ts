/**
 * 5-Fold Stratified Cross-Validation and Hyperparameter Grid Search
 *
 * Ensures exact class proportions across all validation folds, providing
 * unbiased variance estimates for Accuracy, Recall, and ROC-AUC.
 */

import {
  CrossValidationFoldResult,
  CrossValidationSummary,
  GridSearchResult,
  LogisticHyperparams,
  DecisionTreeHyperparams
} from '../types';
import { createPRNG } from './preprocess';
import { trainLogisticRegression, predictProbability } from './logistic';
import { trainDecisionTree, predictTreeProbability } from './decisionTree';
import { computeMetricsFromPredictions } from './metrics';

/**
 * Generate stratified K folds of indices
 */
export function generateStratifiedFolds(y: number[], k: number = 5, seed: number = 42): { trainIdx: number[]; valIdx: number[] }[] {
  const rng = createPRNG(seed);
  const class0: number[] = [];
  const class1: number[] = [];

  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1) class1.push(i);
    else class0.push(i);
  }

  // Shuffle both class index sets
  for (let i = class0.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [class0[i], class0[j]] = [class0[j], class0[i]];
  }
  for (let i = class1.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [class1[i], class1[j]] = [class1[j], class1[i]];
  }

  // Split into k buckets
  const foldBuckets: number[][] = Array.from({ length: k }, () => []);

  for (let i = 0; i < class0.length; i++) {
    foldBuckets[i % k].push(class0[i]);
  }
  for (let i = 0; i < class1.length; i++) {
    foldBuckets[i % k].push(class1[i]);
  }

  // Build train/val index pairs
  const folds: { trainIdx: number[]; valIdx: number[] }[] = [];

  for (let f = 0; f < k; f++) {
    const valIdx = foldBuckets[f];
    const trainIdx: number[] = [];
    for (let other = 0; other < k; other++) {
      if (other !== f) {
        trainIdx.push(...foldBuckets[other]);
      }
    }
    folds.push({ trainIdx, valIdx });
  }

  return folds;
}

/**
 * Run 5-fold Stratified Cross-Validation on Logistic Regression
 */
export function crossValidateLogistic(
  X: number[][],
  y: number[],
  featureNames: string[],
  hyperparams: LogisticHyperparams,
  k: number = 5
): CrossValidationSummary {
  const folds = generateStratifiedFolds(y, k, 123);
  const results: CrossValidationFoldResult[] = [];

  for (let f = 0; f < k; f++) {
    const { trainIdx, valIdx } = folds[f];
    const XTrainFold = trainIdx.map(i => X[i]);
    const yTrainFold = trainIdx.map(i => y[i]);
    const XValFold = valIdx.map(i => X[i]);
    const yValFold = valIdx.map(i => y[i]);

    const model = trainLogisticRegression(XTrainFold, yTrainFold, featureNames, hyperparams);
    const probs = predictProbability(XValFold, model.weights, model.bias);
    const metrics = computeMetricsFromPredictions(yValFold, probs, 0.5);

    results.push({
      fold: f + 1,
      accuracy: metrics.accuracy,
      recall: metrics.recall,
      rocAuc: metrics.rocAuc,
      f1Score: metrics.f1Score
    });
  }

  return aggregateFoldResults(results);
}

/**
 * Run 5-fold Stratified Cross-Validation on Decision Tree
 */
export function crossValidateDecisionTree(
  X: number[][],
  y: number[],
  featureNames: string[],
  hyperparams: DecisionTreeHyperparams,
  k: number = 5
): CrossValidationSummary {
  const folds = generateStratifiedFolds(y, k, 123);
  const results: CrossValidationFoldResult[] = [];

  for (let f = 0; f < k; f++) {
    const { trainIdx, valIdx } = folds[f];
    const XTrainFold = trainIdx.map(i => X[i]);
    const yTrainFold = trainIdx.map(i => y[i]);
    const XValFold = valIdx.map(i => X[i]);
    const yValFold = valIdx.map(i => y[i]);

    const model = trainDecisionTree(XTrainFold, yTrainFold, featureNames, hyperparams);
    const probs = predictTreeProbability(model.root, XValFold);
    const metrics = computeMetricsFromPredictions(yValFold, probs, 0.5);

    results.push({
      fold: f + 1,
      accuracy: metrics.accuracy,
      recall: metrics.recall,
      rocAuc: metrics.rocAuc,
      f1Score: metrics.f1Score
    });
  }

  return aggregateFoldResults(results);
}

function aggregateFoldResults(results: CrossValidationFoldResult[]): CrossValidationSummary {
  const n = results.length || 1;
  const meanAcc = results.reduce((a, b) => a + b.accuracy, 0) / n;
  const stdAcc = Math.sqrt(results.reduce((a, b) => a + Math.pow(b.accuracy - meanAcc, 2), 0) / n);

  const meanRec = results.reduce((a, b) => a + b.recall, 0) / n;
  const stdRec = Math.sqrt(results.reduce((a, b) => a + Math.pow(b.recall - meanRec, 2), 0) / n);

  const meanAuc = results.reduce((a, b) => a + b.rocAuc, 0) / n;
  const stdAuc = Math.sqrt(results.reduce((a, b) => a + Math.pow(b.rocAuc - meanAuc, 2), 0) / n);

  const meanF1 = results.reduce((a, b) => a + b.f1Score, 0) / n;
  const stdF1 = Math.sqrt(results.reduce((a, b) => a + Math.pow(b.f1Score - meanF1, 2), 0) / n);

  return {
    folds: results,
    meanAccuracy: Number(meanAcc.toFixed(4)),
    stdAccuracy: Number(stdAcc.toFixed(4)),
    meanRecall: Number(meanRec.toFixed(4)),
    stdRecall: Number(stdRec.toFixed(4)),
    meanRocAuc: Number(meanAuc.toFixed(4)),
    stdRocAuc: Number(stdAuc.toFixed(4)),
    meanF1: Number(meanF1.toFixed(4)),
    stdF1: Number(stdF1.toFixed(4))
  };
}

/**
 * Grid Search for Logistic Regression Hyperparameters
 */
export function gridSearchLogistic(
  X: number[][],
  y: number[],
  featureNames: string[]
): { results: GridSearchResult[]; best: GridSearchResult } {
  const learningRates = [0.01, 0.05, 0.1];
  const regularizations = [0.0, 0.01, 0.1];
  const epochs = 150;

  const results: GridSearchResult[] = [];

  for (const lr of learningRates) {
    for (const reg of regularizations) {
      const cv = crossValidateLogistic(
        X,
        y,
        featureNames,
        { learningRate: lr, epochs, regularization: reg },
        3
      );

      results.push({
        params: { lr, reg, epochs },
        meanAccuracy: cv.meanAccuracy,
        meanRecall: cv.meanRecall,
        meanRocAuc: cv.meanRocAuc,
        stdAccuracy: cv.stdAccuracy
      });
    }
  }

  results.sort((a, b) => b.meanRocAuc - a.meanRocAuc);
  return { results, best: results[0] };
}

/**
 * Grid Search for Decision Tree Hyperparameters
 */
export function gridSearchDecisionTree(
  X: number[][],
  y: number[],
  featureNames: string[]
): { results: GridSearchResult[]; best: GridSearchResult } {
  const depths = [2, 3, 4, 5];
  const criteria: ('gini' | 'entropy')[] = ['gini', 'entropy'];

  const results: GridSearchResult[] = [];

  for (const maxDepth of depths) {
    for (const criterion of criteria) {
      const cv = crossValidateDecisionTree(
        X,
        y,
        featureNames,
        { maxDepth, criterion, minSamplesSplit: 2, minSamplesLeaf: 1 },
        3
      );

      results.push({
        params: { maxDepth, criterion },
        meanAccuracy: cv.meanAccuracy,
        meanRecall: cv.meanRecall,
        meanRocAuc: cv.meanRocAuc,
        stdAccuracy: cv.stdAccuracy
      });
    }
  }

  results.sort((a, b) => b.meanRocAuc - a.meanRocAuc);
  return { results, best: results[0] };
}

/**
 * Universal cross-validation runner
 */
export function runCrossValidation(
  modelType: 'logistic' | 'tree',
  X: number[][],
  y: number[],
  k: number = 5,
  lrParams?: LogisticHyperparams,
  dtParams?: DecisionTreeHyperparams,
  featureNames: string[] = []
): CrossValidationSummary {
  const fNames = featureNames.length ? featureNames : Array.from({ length: X[0]?.length || 0 }, (_, i) => `f_${i}`);
  if (modelType === 'logistic') {
    return crossValidateLogistic(X, y, fNames, lrParams || { learningRate: 0.05, epochs: 150, regularization: 0.01 }, k);
  } else {
    return crossValidateDecisionTree(X, y, fNames, dtParams || { maxDepth: 4, criterion: 'gini', minSamplesSplit: 2, minSamplesLeaf: 1 }, k);
  }
}

/**
 * Grid search runner for logistic regression
 */
export function runGridSearchLogistic(
  X: number[][],
  y: number[],
  learningRates: number[] = [0.01, 0.05, 0.1],
  regularizations: number[] = [0.0, 0.02, 0.1],
  epochs: number = 150,
  featureNames: string[] = []
): { results: GridSearchResult[]; best: GridSearchResult } {
  const fNames = featureNames.length ? featureNames : Array.from({ length: X[0]?.length || 0 }, (_, i) => `f_${i}`);
  const results: GridSearchResult[] = [];
  for (const lr of learningRates) {
    for (const reg of regularizations) {
      const cv = crossValidateLogistic(
        X,
        y,
        fNames,
        { learningRate: lr, epochs, regularization: reg },
        3
      );
      results.push({
        params: { lr, reg, epochs },
        meanAccuracy: cv.meanAccuracy,
        meanRecall: cv.meanRecall,
        meanRocAuc: cv.meanRocAuc,
        stdAccuracy: cv.stdAccuracy
      });
    }
  }
  results.sort((a, b) => b.meanRocAuc - a.meanRocAuc);
  return { results, best: results[0] };
}

/**
 * Grid search runner for decision tree
 */
export function runGridSearchTree(
  X: number[][],
  y: number[],
  featureNames: string[] = [],
  depths: number[] = [2, 3, 4, 5],
  criteria: ('gini' | 'entropy')[] = ['gini', 'entropy']
): { results: GridSearchResult[]; best: GridSearchResult } {
  const fNames = featureNames.length ? featureNames : Array.from({ length: X[0]?.length || 0 }, (_, i) => `f_${i}`);
  const results: GridSearchResult[] = [];
  for (const maxDepth of depths) {
    for (const criterion of criteria) {
      const cv = crossValidateDecisionTree(
        X,
        y,
        fNames,
        { maxDepth, criterion, minSamplesSplit: 2, minSamplesLeaf: 1 },
        3
      );
      results.push({
        params: { maxDepth, criterion },
        meanAccuracy: cv.meanAccuracy,
        meanRecall: cv.meanRecall,
        meanRocAuc: cv.meanRocAuc,
        stdAccuracy: cv.stdAccuracy
      });
    }
  }
  results.sort((a, b) => b.meanRocAuc - a.meanRocAuc);
  return { results, best: results[0] };
}
