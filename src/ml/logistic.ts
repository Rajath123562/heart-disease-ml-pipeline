/**
 * Logistic Regression from Scratch in TypeScript
 *
 * Algorithm Overview:
 * 1. Linear combination: z = X * w + b
 * 2. Sigmoid activation: p_hat = 1 / (1 + exp(-z))
 * 3. Regularized Cross-Entropy (Log-Loss):
 *    J(w, b) = -(1/m) * sum[ y * log(p_hat) + (1-y) * log(1 - p_hat) ] + (lambda / (2*m)) * sum(w_j^2)
 * 4. Analytical Gradient Computation:
 *    grad_w = (1/m) * X^T * (p_hat - y) + (lambda / m) * w
 *    grad_b = (1/m) * sum(p_hat - y)
 * 5. Batch Gradient Descent Parameter Updates:
 *    w = w - alpha * grad_w
 *    b = b - alpha * grad_b
 */

import { LogisticHyperparams, LogisticModelArtifact } from '../types';

/**
 * Numerically stable Sigmoid function
 * Clips extreme values to prevent floating-point overflow
 */
export function sigmoid(z: number): number {
  if (z >= 45) return 1;
  if (z <= -45) return 0;
  return 1 / (1 + Math.exp(-z));
}

/**
 * Predict raw probabilities for an array of feature vectors
 */
export function predictProbability(X: number[][], weights: number[], bias: number): number[] {
  const m = X.length;
  const n = weights.length;
  const probs: number[] = new Array(m);

  for (let i = 0; i < m; i++) {
    const row = X[i];
    let z = bias;
    for (let j = 0; j < n; j++) {
      z += row[j] * weights[j];
    }
    probs[i] = sigmoid(z);
  }

  return probs;
}

/**
 * Predict binary classes (0 or 1) using a decision threshold
 */
export function predictClass(
  X: number[][],
  weights: number[],
  bias: number,
  threshold: number = 0.5
): number[] {
  const probs = predictProbability(X, weights, bias);
  return probs.map(p => (p >= threshold ? 1 : 0));
}

/**
 * Compute binary cross-entropy loss with L2 regularization
 */
export function computeLoss(
  X: number[][],
  y: number[],
  weights: number[],
  bias: number,
  lambdaL2: number
): number {
  const m = X.length;
  if (m === 0) return 0;
  const probs = predictProbability(X, weights, bias);
  const eps = 1e-15; // Numerical stability epsilon to prevent log(0)

  let logLossSum = 0;
  for (let i = 0; i < m; i++) {
    const p = Math.max(eps, Math.min(1 - eps, probs[i]));
    const target = y[i];
    logLossSum += target * Math.log(p) + (1 - target) * Math.log(1 - p);
  }

  let l2Penalty = 0;
  if (lambdaL2 > 0) {
    for (let j = 0; j < weights.length; j++) {
      l2Penalty += weights[j] * weights[j];
    }
    l2Penalty = (lambdaL2 / (2 * m)) * l2Penalty;
  }

  return -(logLossSum / m) + l2Penalty;
}

/**
 * Train Logistic Regression model using Batch Gradient Descent
 */
export function trainLogisticRegression(
  X: number[][],
  y: number[],
  arg3: string[] | number,
  arg4?: LogisticHyperparams | number,
  arg5?: number | string[],
  arg6?: string[],
  onEpochProgress?: (epoch: number, loss: number) => void
): LogisticModelArtifact {
  let featureNames: string[] = [];
  let hyperparams: LogisticHyperparams = { learningRate: 0.05, epochs: 200, regularization: 0.01 };

  if (Array.isArray(arg3)) {
    featureNames = arg3;
    if (typeof arg4 === 'object' && arg4 !== null) {
      hyperparams = arg4 as LogisticHyperparams;
    }
  } else if (typeof arg3 === 'number') {
    const lr = arg3;
    const ep = typeof arg4 === 'number' ? arg4 : 200;
    const reg = typeof arg5 === 'number' ? arg5 : 0.01;
    if (Array.isArray(arg6)) {
      featureNames = arg6;
    } else if (Array.isArray(arg5)) {
      featureNames = arg5;
    }
    hyperparams = { learningRate: lr, epochs: ep, regularization: reg };
  }

  if (featureNames.length === 0 && X.length > 0) {
    featureNames = Array.from({ length: X[0].length }, (_, i) => `f_${i}`);
  }

  const startTime = performance.now();
  const m = X.length; // Number of training instances
  const n = featureNames.length; // Number of features

  // Initialize weights with small Xavier-like random values or zeros
  const weights: number[] = new Array(n).fill(0);
  let bias = 0;

  const lossHistory: { epoch: number; loss: number }[] = [];
  const { learningRate, epochs, regularization } = hyperparams;

  // Track initial loss at epoch 0
  const initialLoss = computeLoss(X, y, weights, bias, regularization);
  lossHistory.push({ epoch: 0, loss: initialLoss });

  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Forward pass: compute predictions
    const probs = predictProbability(X, weights, bias);

    // Compute error vector (p_hat - y)
    const errors = new Array(m);
    let gradBias = 0;
    for (let i = 0; i < m; i++) {
      const err = probs[i] - y[i];
      errors[i] = err;
      gradBias += err;
    }
    gradBias = gradBias / m;

    // Compute feature weight gradients with L2 regularization
    for (let j = 0; j < n; j++) {
      let gradWeight = 0;
      for (let i = 0; i < m; i++) {
        gradWeight += errors[i] * X[i][j];
      }
      gradWeight = gradWeight / m + (regularization / m) * weights[j];

      // Gradient descent step
      weights[j] -= learningRate * gradWeight;
    }

    // Bias update (unregularized)
    bias -= learningRate * gradBias;

    // Record loss for monitoring and plotting
    if (epoch % Math.max(1, Math.floor(epochs / 50)) === 0 || epoch === epochs) {
      const currentLoss = computeLoss(X, y, weights, bias, regularization);
      lossHistory.push({ epoch, loss: Number(currentLoss.toFixed(4)) });
      if (onEpochProgress) {
        onEpochProgress(epoch, currentLoss);
      }
    }
  }

  const trainTimeMs = Math.round(performance.now() - startTime);

  return {
    weights,
    bias,
    lossHistory,
    trainTimeMs,
    featureNames,
    hyperparams
  };
}

/**
 * Extract odds ratios and interpretable feature coefficients
 */
export function getLogisticCoefficientsSummary(
  model: LogisticModelArtifact
): { feature: string; weight: number; absWeight: number; oddsRatio: number; direction: 'Risk Factor' | 'Protective' }[] {
  return model.featureNames.map((name, i) => {
    const w = model.weights[i] ?? 0;
    const dir: 'Risk Factor' | 'Protective' = w > 0 ? 'Risk Factor' : 'Protective';
    return {
      feature: name,
      weight: Number(w.toFixed(4)),
      absWeight: Math.abs(w),
      oddsRatio: Number(Math.exp(w).toFixed(4)),
      direction: dir
    };
  }).sort((a, b) => b.absWeight - a.absWeight);
}
