/**
 * Classification and Regression Trees (CART) Decision Tree from Scratch
 * Supports:
 * - Gini Impurity or Shannon Entropy splitting criteria
 * - Adjustable maxDepth, minSamplesSplit, minSamplesLeaf
 * - Feature importance accumulation by weighted impurity reduction
 * - Tree structure serialization and path-tracing for interpretability
 */

import { TreeNode, DecisionTreeHyperparams, DecisionTreeModelArtifact } from '../types';

/**
 * Calculate impurity of a binary label distribution
 */
export function calculateImpurity(
  labels: number[],
  criterion: 'gini' | 'entropy'
): { impurity: number; count0: number; count1: number } {
  const n = labels.length;
  if (n === 0) return { impurity: 0, count0: 0, count1: 0 };

  let count1 = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] === 1) count1++;
  }
  const count0 = n - count1;

  const p0 = count0 / n;
  const p1 = count1 / n;

  if (criterion === 'gini') {
    // Gini = 1 - (p0^2 + p1^2)
    const gini = 1 - (p0 * p0 + p1 * p1);
    return { impurity: gini, count0, count1 };
  } else {
    // Entropy = - p0*log2(p0) - p1*log2(p1)
    let ent = 0;
    if (p0 > 0) ent -= p0 * Math.log2(p0);
    if (p1 > 0) ent -= p1 * Math.log2(p1);
    return { impurity: ent, count0, count1 };
  }
}

interface BestSplit {
  featureIndex: number;
  threshold: number;
  gain: number;
  leftIndices: number[];
  rightIndices: number[];
}

/**
 * Find best splitting feature and threshold across all candidate splits
 */
function findBestSplit(
  X: number[][],
  y: number[],
  indices: number[],
  criterion: 'gini' | 'entropy',
  minSamplesLeaf: number
): BestSplit | null {
  const n = indices.length;
  if (n < 2) return null;

  const currentLabels = indices.map(i => y[i]);
  const parentImpurity = calculateImpurity(currentLabels, criterion).impurity;

  let bestSplit: BestSplit | null = null;
  let maxGain = -1;
  const numFeatures = X[0].length;

  for (let f = 0; f < numFeatures; f++) {
    // Extract unique sorted feature values to evaluate candidate split thresholds
    const uniqueVals = Array.from(new Set(indices.map(i => X[i][f]))).sort((a, b) => a - b);
    if (uniqueVals.length < 2) continue;

    // Evaluate midpoints between consecutive unique values
    for (let k = 0; k < uniqueVals.length - 1; k++) {
      const threshold = (uniqueVals[k] + uniqueVals[k + 1]) / 2;

      const leftIndices: number[] = [];
      const rightIndices: number[] = [];

      for (let i = 0; i < n; i++) {
        const idx = indices[i];
        if (X[idx][f] <= threshold) {
          leftIndices.push(idx);
        } else {
          rightIndices.push(idx);
        }
      }

      // Check minSamplesLeaf constraint
      if (leftIndices.length < minSamplesLeaf || rightIndices.length < minSamplesLeaf) {
        continue;
      }

      // Calculate weighted children impurity
      const leftLabels = leftIndices.map(i => y[i]);
      const rightLabels = rightIndices.map(i => y[i]);

      const leftImp = calculateImpurity(leftLabels, criterion).impurity;
      const rightImp = calculateImpurity(rightLabels, criterion).impurity;

      const weightedChildImpurity =
        (leftIndices.length / n) * leftImp + (rightIndices.length / n) * rightImp;
      const gain = parentImpurity - weightedChildImpurity;

      if (gain > maxGain && gain > 1e-7) {
        maxGain = gain;
        bestSplit = {
          featureIndex: f,
          threshold,
          gain,
          leftIndices,
          rightIndices
        };
      }
    }
  }

  return bestSplit;
}

/**
 * Recursive tree builder
 */
function buildTreeRecursive(
  X: number[][],
  y: number[],
  indices: number[],
  depth: number,
  featureNames: string[],
  hyperparams: DecisionTreeHyperparams,
  nodeId: string,
  featureGains: number[]
): TreeNode {
  const currentLabels = indices.map(i => y[i]);
  const { impurity, count0, count1 } = calculateImpurity(currentLabels, hyperparams.criterion);
  const samples = indices.length;
  const prob1 = samples > 0 ? count1 / samples : 0;
  const predictedClass = prob1 >= 0.5 ? 1 : 0;

  // Stopping conditions:
  // 1. Reached max depth
  // 2. Insufficient samples to split
  // 3. Pure node (impurity == 0)
  const isStoppingConditionMet =
    depth >= hyperparams.maxDepth ||
    samples < hyperparams.minSamplesSplit ||
    impurity <= 1e-6;

  if (isStoppingConditionMet) {
    return {
      id: nodeId,
      isLeaf: true,
      predictedClass,
      probability: Number(prob1.toFixed(4)),
      impurity: Number(impurity.toFixed(4)),
      samples,
      value: [count0, count1],
      depth
    };
  }

  // Find optimal split
  const bestSplit = findBestSplit(X, y, indices, hyperparams.criterion, hyperparams.minSamplesLeaf);

  if (!bestSplit) {
    return {
      id: nodeId,
      isLeaf: true,
      predictedClass,
      probability: Number(prob1.toFixed(4)),
      impurity: Number(impurity.toFixed(4)),
      samples,
      value: [count0, count1],
      depth
    };
  }

  // Accumulate feature importance: node_samples * gain
  featureGains[bestSplit.featureIndex] += samples * bestSplit.gain;

  const leftChild = buildTreeRecursive(
    X,
    y,
    bestSplit.leftIndices,
    depth + 1,
    featureNames,
    hyperparams,
    `${nodeId}L`,
    featureGains
  );

  const rightChild = buildTreeRecursive(
    X,
    y,
    bestSplit.rightIndices,
    depth + 1,
    featureNames,
    hyperparams,
    `${nodeId}R`,
    featureGains
  );

  return {
    id: nodeId,
    isLeaf: false,
    featureIndex: bestSplit.featureIndex,
    featureName: featureNames[bestSplit.featureIndex],
    threshold: Number(bestSplit.threshold.toFixed(3)),
    left: leftChild,
    right: rightChild,
    impurity: Number(impurity.toFixed(4)),
    samples,
    value: [count0, count1],
    predictedClass,
    probability: Number(prob1.toFixed(4)),
    depth
  };
}

/**
 * Count total tree nodes, leaf nodes, and calculate max achieved depth
 */
function getTreeStats(root: TreeNode): { totalNodes: number; leafNodes: number; maxAchievedDepth: number } {
  let totalNodes = 0;
  let leafNodes = 0;
  let maxAchievedDepth = 0;

  function traverse(node: TreeNode) {
    totalNodes++;
    if (node.depth > maxAchievedDepth) maxAchievedDepth = node.depth;
    if (node.isLeaf) {
      leafNodes++;
    } else {
      if (node.left) traverse(node.left);
      if (node.right) traverse(node.right);
    }
  }

  traverse(root);
  return { totalNodes, leafNodes, maxAchievedDepth };
}

/**
 * Train CART Decision Tree from scratch
 */
export function trainDecisionTree(
  X: number[][],
  y: number[],
  featureNames: string[],
  hyperparams: DecisionTreeHyperparams
): DecisionTreeModelArtifact {
  const startTime = performance.now();
  const allIndices = Array.from({ length: X.length }, (_, i) => i);
  const featureGains = new Array(featureNames.length).fill(0);

  const root = buildTreeRecursive(
    X,
    y,
    allIndices,
    0,
    featureNames,
    hyperparams,
    'root',
    featureGains
  );

  // Compute normalized feature importances
  const totalGain = featureGains.reduce((a, b) => a + b, 0) || 1;
  const featureImportances = featureNames
    .map((name, i) => ({
      feature: name,
      importance: Number((featureGains[i] / totalGain).toFixed(4))
    }))
    .sort((a, b) => b.importance - a.importance);

  const { totalNodes, leafNodes, maxAchievedDepth } = getTreeStats(root);
  const trainTimeMs = Math.round(performance.now() - startTime);

  return {
    root,
    featureImportances,
    trainTimeMs,
    hyperparams,
    maxAchievedDepth,
    totalNodes,
    leafNodes
  };
}

/**
 * Predict class probability for a single instance using Decision Tree
 */
export function predictSampleProbability(root: TreeNode, sample: number[]): { prob: number; path: string[] } {
  let curr: TreeNode = root;
  const path: string[] = [];

  while (!curr.isLeaf) {
    const fIdx = curr.featureIndex!;
    const fVal = sample[fIdx];
    const thresh = curr.threshold!;
    const fName = curr.featureName || `Feature ${fIdx}`;

    if (fVal <= thresh) {
      path.push(`${fName} = ${fVal.toFixed(2)} <= ${thresh.toFixed(2)} [True -> Left]`);
      if (!curr.left) break;
      curr = curr.left;
    } else {
      path.push(`${fName} = ${fVal.toFixed(2)} > ${thresh.toFixed(2)} [False -> Right]`);
      if (!curr.right) break;
      curr = curr.right;
    }
  }

  const prob = curr.probability !== undefined ? curr.probability : (curr.predictedClass ?? 0);
  return { prob, path };
}

/**
 * Predict probabilities for a matrix of instances
 */
export function predictTreeProbability(root: TreeNode, X: number[][]): number[] {
  return X.map(row => predictSampleProbability(root, row).prob);
}

/**
 * Predict classes for a matrix of instances given a decision threshold
 */
export function predictTreeClass(root: TreeNode, X: number[][], threshold: number = 0.5): number[] {
  const probs = predictTreeProbability(root, X);
  return probs.map(p => (p >= threshold ? 1 : 0));
}
