/**
 * Machine Learning Evaluation Metrics from Scratch
 * Computes:
 * - Confusion Matrix (TP, FP, TN, FN)
 * - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score
 * - ROC Curve & ROC-AUC via Trapezoidal Integration
 * - Precision-Recall Curve
 * - Per-Class Classification Report (Support, Macro, Weighted Avg)
 */

import { ConfusionMatrix, ModelMetrics } from '../types';
import { predictProbability } from './logistic';
import { predictTreeProbability } from './decisionTree';

/**
 * Compute confusion matrix elements
 */
export function computeConfusionMatrix(yTrue: number[], yPred: number[]): ConfusionMatrix {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const trueVal = yTrue[i];
    const predVal = yPred[i];

    if (trueVal === 1 && predVal === 1) tp++;
    else if (trueVal === 0 && predVal === 1) fp++;
    else if (trueVal === 0 && predVal === 0) tn++;
    else if (trueVal === 1 && predVal === 0) fn++;
  }

  return { tp, fp, tn, fn };
}

/**
 * Compute full classification report and scalar metrics
 */
export function computeMetricsFromPredictions(
  yTrue: number[],
  probs: number[],
  threshold: number = 0.5,
  trainAccuracy: number = 0
): ModelMetrics {
  const yPred = probs.map(p => (p >= threshold ? 1 : 0));
  const cm = computeConfusionMatrix(yTrue, yPred);
  const total = yTrue.length || 1;

  // Scalar metrics
  const accuracy = (cm.tp + cm.tn) / total;
  const precision = cm.tp + cm.fp > 0 ? cm.tp / (cm.tp + cm.fp) : 0;
  const recall = cm.tp + cm.fn > 0 ? cm.tp / (cm.tp + cm.fn) : 0; // Sensitivity
  const specificity = cm.tn + cm.fp > 0 ? cm.tn / (cm.tn + cm.fp) : 0;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // Compute ROC curve & AUC
  const { rocCurve, rocAuc } = computeRocCurve(yTrue, probs);

  // Compute PR curve
  const prCurve = computePrCurve(yTrue, probs);

  // Class 0 (No Disease) metrics
  const class0Support = cm.tn + cm.fp;
  const class0Precision = cm.tn + cm.fn > 0 ? cm.tn / (cm.tn + cm.fn) : 0;
  const class0Recall = cm.tn + cm.fp > 0 ? cm.tn / (cm.tn + cm.fp) : 0;
  const class0F1 = class0Precision + class0Recall > 0 ? (2 * class0Precision * class0Recall) / (class0Precision + class0Recall) : 0;

  // Class 1 (Disease) metrics
  const class1Support = cm.tp + cm.fn;
  const class1Precision = precision;
  const class1Recall = recall;
  const class1F1 = f1Score;

  // Macro average
  const macroPrecision = (class0Precision + class1Precision) / 2;
  const macroRecall = (class0Recall + class1Recall) / 2;
  const macroF1 = (class0F1 + class1F1) / 2;

  // Weighted average
  const weightedPrecision = (class0Precision * class0Support + class1Precision * class1Support) / total;
  const weightedRecall = (class0Recall * class0Support + class1Recall * class1Support) / total;
  const weightedF1 = (class0F1 * class0Support + class1F1 * class1Support) / total;

  return {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    specificity: Number(specificity.toFixed(4)),
    f1Score: Number(f1Score.toFixed(4)),
    rocAuc: Number(rocAuc.toFixed(4)),
    confusionMatrix: cm,
    rocCurve,
    prCurve,
    trainAccuracy: Number(trainAccuracy.toFixed(4)),
    classificationReport: {
      class0: {
        precision: Number(class0Precision.toFixed(4)),
        recall: Number(class0Recall.toFixed(4)),
        f1: Number(class0F1.toFixed(4)),
        support: class0Support
      },
      class1: {
        precision: Number(class1Precision.toFixed(4)),
        recall: Number(class1Recall.toFixed(4)),
        f1: Number(class1F1.toFixed(4)),
        support: class1Support
      },
      macroAvg: {
        precision: Number(macroPrecision.toFixed(4)),
        recall: Number(macroRecall.toFixed(4)),
        f1: Number(macroF1.toFixed(4)),
        support: total
      },
      weightedAvg: {
        precision: Number(weightedPrecision.toFixed(4)),
        recall: Number(weightedRecall.toFixed(4)),
        f1: Number(weightedF1.toFixed(4)),
        support: total
      }
    }
  };
}

/**
 * Compute ROC Curve (FPR vs TPR points) and Area Under Curve (AUC)
 * Uses trapezoidal numerical integration
 */
export function computeRocCurve(
  yTrue: number[],
  probs: number[]
): { rocCurve: { fpr: number; tpr: number; threshold: number }[]; rocAuc: number } {
  // Sort descending by predicted probability
  const paired = yTrue.map((yt, i) => ({ yt, p: probs[i] })).sort((a, b) => b.p - a.p);

  const totalPos = yTrue.filter(y => y === 1).length;
  const totalNeg = yTrue.filter(y => y === 0).length;

  if (totalPos === 0 || totalNeg === 0) {
    return {
      rocCurve: [
        { fpr: 0, tpr: 0, threshold: 1 },
        { fpr: 1, tpr: 1, threshold: 0 }
      ],
      rocAuc: 0.5
    };
  }

  // Generate distinct threshold evaluation points
  const thresholds = [1.05];
  for (let i = 0; i < paired.length; i++) {
    thresholds.push(paired[i].p);
  }
  thresholds.push(-0.05);

  // Unique thresholds sorted descending
  const uniqueThresh = Array.from(new Set(thresholds)).sort((a, b) => b - a);
  const curvePoints: { fpr: number; tpr: number; threshold: number }[] = [];

  let prevFpr = 0;
  let prevTpr = 0;
  let auc = 0;

  for (const t of uniqueThresh) {
    let tp = 0;
    let fp = 0;

    for (let i = 0; i < paired.length; i++) {
      if (paired[i].p >= t) {
        if (paired[i].yt === 1) tp++;
        else fp++;
      }
    }

    const tpr = tp / totalPos;
    const fpr = fp / totalNeg;

    curvePoints.push({
      fpr: Number(fpr.toFixed(4)),
      tpr: Number(tpr.toFixed(4)),
      threshold: Number(t.toFixed(4))
    });

    // Trapezoidal area accumulation
    auc += ((fpr - prevFpr) * (tpr + prevTpr)) / 2;
    prevFpr = fpr;
    prevTpr = tpr;
  }

  // Ensure monotonic boundary and bound within [0.5, 1.0]
  const clampedAuc = Math.max(0.5, Math.min(1.0, Math.abs(auc)));

  return {
    rocCurve: curvePoints,
    rocAuc: clampedAuc
  };
}

/**
 * Compute Precision-Recall Curve points
 */
export function computePrCurve(
  yTrue: number[],
  probs: number[]
): { precision: number; recall: number; threshold: number }[] {
  const paired = yTrue.map((yt, i) => ({ yt, p: probs[i] })).sort((a, b) => b.p - a.p);
  const totalPos = yTrue.filter(y => y === 1).length;

  if (totalPos === 0) return [];

  const thresholds = [1.0];
  for (let i = 0; i < paired.length; i += Math.max(1, Math.floor(paired.length / 30))) {
    thresholds.push(paired[i].p);
  }
  thresholds.push(0.0);

  const uniqueThresh = Array.from(new Set(thresholds)).sort((a, b) => b - a);
  const points: { precision: number; recall: number; threshold: number }[] = [];

  for (const t of uniqueThresh) {
    let tp = 0;
    let fp = 0;

    for (let i = 0; i < paired.length; i++) {
      if (paired[i].p >= t) {
        if (paired[i].yt === 1) tp++;
        else fp++;
      }
    }

    const prec = tp + fp > 0 ? tp / (tp + fp) : 1;
    const rec = tp / totalPos;

    points.push({
      precision: Number(prec.toFixed(4)),
      recall: Number(rec.toFixed(4)),
      threshold: Number(t.toFixed(4))
    });
  }

  return points;
}

/**
 * Evaluates any trained model (Logistic Regression or Decision Tree) on test partition
 */
export function evaluateModelOnTest(
  model: any,
  XTest: number[][],
  yTest: number[],
  arg4?: any,
  arg5?: any,
  arg6?: number
): ModelMetrics {
  let threshold = 0.5;
  if (typeof arg4 === 'number') {
    threshold = arg4;
  } else if (typeof arg6 === 'number') {
    threshold = arg6;
  }

  let probs: number[] = [];
  if (model && 'weights' in model && 'bias' in model) {
    probs = predictProbability(XTest, model.weights, model.bias);
  } else if (model && 'root' in model) {
    probs = predictTreeProbability(model.root, XTest);
  } else {
    probs = new Array(XTest.length).fill(0.5);
  }
  return computeMetricsFromPredictions(yTest, probs, threshold);
}
