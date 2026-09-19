import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend
} from 'recharts';
import {
  Cpu,
  Play,
  Settings2,
  GitFork,
  ChevronRight,
  ChevronDown,
  Search,
  CheckCircle2,
  Clock,
  Sparkles,
  TrendingDown,
  Info
} from 'lucide-react';
import {
  LogisticHyperparams,
  DecisionTreeHyperparams,
  LogisticModelArtifact,
  DecisionTreeModelArtifact,
  TreeNode,
  GridSearchResult
} from '../types';

interface TrainingTabProps {
  logisticHyperparams: LogisticHyperparams;
  treeHyperparams: DecisionTreeHyperparams;
  logisticModel: LogisticModelArtifact | null;
  treeModel: DecisionTreeModelArtifact | null;
  isTraining: boolean;
  onUpdateLogisticParams: (params: LogisticHyperparams) => void;
  onUpdateTreeParams: (params: DecisionTreeHyperparams) => void;
  onTrainModels: () => void;
  onRunGridSearch: () => void;
  isGridSearching: boolean;
  gridSearchResults: {
    logistic?: { results: GridSearchResult[]; best: GridSearchResult };
    tree?: { results: GridSearchResult[]; best: GridSearchResult };
  } | null;
}

/**
 * Recursive Tree Node Component for Expandable Visualization
 */
const TreeNodeView: React.FC<{ node: TreeNode; depth?: number }> = ({ node, depth = 0 }) => {
  const [isExpanded, setIsExpanded] = useState(depth < 2);

  if (node.isLeaf) {
    const isDisease = node.predictedClass === 1;
    return (
      <div className="ml-4 my-1.5 p-2 rounded-lg border text-xs bg-slate-50 dark:bg-slate-800/80 border-slate-200 dark:border-slate-700/80 inline-flex items-center gap-2 shadow-xs">
        <span
          className={`px-2 py-0.5 rounded-full font-bold uppercase text-[10px] ${
            isDisease
              ? 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300 border border-rose-300 dark:border-rose-800'
              : 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 border border-blue-300 dark:border-blue-800'
          }`}
        >
          {isDisease ? 'Leaf: Heart Disease (1)' : 'Leaf: No Disease (0)'}
        </span>
        <span className="font-mono text-slate-500 text-[11px]">
          prob={node.probability?.toFixed(2)} | n={node.samples} (c0:{node.value[0]}, c1:{node.value[1]})
        </span>
      </div>
    );
  }

  return (
    <div className="ml-4 my-1 text-xs">
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="inline-flex items-center gap-2 p-2 rounded-lg border bg-white dark:bg-slate-850 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 transition cursor-pointer shadow-xs"
      >
        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
        )}
        <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">
          [{node.featureName} &le; {node.threshold}]
        </span>
        <span className="text-[11px] text-slate-400 font-mono">
          impurity={node.impurity} | samples={node.samples}
        </span>
      </div>

      {isExpanded && (
        <div className="border-l-2 border-slate-200 dark:border-slate-700/80 ml-2 pl-2 space-y-1">
          {node.left && (
            <div>
              <span className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 ml-2">
                True (Left):
              </span>
              <TreeNodeView node={node.left} depth={depth + 1} />
            </div>
          )}
          {node.right && (
            <div>
              <span className="text-[10px] font-bold text-rose-600 dark:text-rose-400 ml-2">
                False (Right):
              </span>
              <TreeNodeView node={node.right} depth={depth + 1} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const TrainingTab: React.FC<TrainingTabProps> = ({
  logisticHyperparams,
  treeHyperparams,
  logisticModel,
  treeModel,
  isTraining,
  onUpdateLogisticParams,
  onUpdateTreeParams,
  onTrainModels,
  onRunGridSearch,
  isGridSearching,
  gridSearchResults
}) => {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner & Primary Execution */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-rose-500" />
            <span>Algorithm Training from Scratch</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Batch Gradient Descent Logistic Regression with L2 Regularization & CART Decision Tree.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onRunGridSearch}
            disabled={isGridSearching || isTraining}
            className="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-semibold text-xs rounded-xl border border-slate-300 dark:border-slate-700 transition flex items-center gap-2 cursor-pointer"
          >
            {isGridSearching ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                <span>Searching Grid...</span>
              </>
            ) : (
              <>
                <Search className="w-3.5 h-3.5" />
                <span>5-Fold Grid Search</span>
              </>
            )}
          </button>

          <button
            onClick={onTrainModels}
            disabled={isTraining}
            className="px-5 py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs rounded-xl shadow-md shadow-rose-600/20 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer"
          >
            {isTraining ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Optimizing Models...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Train Both Models</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Model Training Statistics Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-blue-500" /> Logistic Train Time
          </div>
          <div className="text-xl font-extrabold text-slate-900 dark:text-white mt-1">
            {logisticModel ? `${logisticModel.trainTimeMs} ms` : 'Not trained'}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            {logisticHyperparams.epochs} epochs gradient descent
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-indigo-500" /> Tree Train Time
          </div>
          <div className="text-xl font-extrabold text-slate-900 dark:text-white mt-1">
            {treeModel ? `${treeModel.trainTimeMs} ms` : 'Not trained'}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            {treeModel ? `${treeModel.totalNodes} total nodes created` : 'CART recursive split'}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase flex items-center gap-1.5">
            <GitFork className="w-3.5 h-3.5 text-emerald-500" /> Achieved Depth
          </div>
          <div className="text-xl font-extrabold text-slate-900 dark:text-white mt-1">
            {treeModel ? `Depth ${treeModel.maxAchievedDepth}` : 'N/A'}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            {treeModel ? `${treeModel.leafNodes} terminal leaf leaves` : 'Max Depth Cap'}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase flex items-center gap-1.5">
            <TrendingDown className="w-3.5 h-3.5 text-rose-500" /> Final Log-Loss
          </div>
          <div className="text-xl font-extrabold text-rose-600 dark:text-rose-400 mt-1">
            {logisticModel?.lossHistory?.length
              ? logisticModel.lossHistory[logisticModel.lossHistory.length - 1].loss.toFixed(4)
              : 'N/A'}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">Regularized binary loss</div>
        </div>
      </div>

      {/* Model Hyperparameters & Visualizers Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model 1: Logistic Regression Controls & Live Loss Curve */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-rose-500" />
                <span>Logistic Regression Hyperparameters</span>
              </h3>
              <p className="text-xs text-slate-500">Continuous linear combination with Sigmoid activation</p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300">
              L2 Regularized
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                Learning Rate (&alpha;): {logisticHyperparams.learningRate}
              </label>
              <input
                type="range"
                min={0.005}
                max={0.2}
                step={0.005}
                value={logisticHyperparams.learningRate}
                onChange={e =>
                  onUpdateLogisticParams({
                    ...logisticHyperparams,
                    learningRate: parseFloat(e.target.value)
                  })
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            <div>
              <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                Epochs: {logisticHyperparams.epochs}
              </label>
              <input
                type="range"
                min={50}
                max={500}
                step={25}
                value={logisticHyperparams.epochs}
                onChange={e =>
                  onUpdateLogisticParams({
                    ...logisticHyperparams,
                    epochs: parseInt(e.target.value)
                  })
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            <div>
              <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                L2 Penalty (&lambda;): {logisticHyperparams.regularization}
              </label>
              <input
                type="range"
                min={0}
                max={0.2}
                step={0.01}
                value={logisticHyperparams.regularization}
                onChange={e =>
                  onUpdateLogisticParams({
                    ...logisticHyperparams,
                    regularization: parseFloat(e.target.value)
                  })
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>
          </div>

          {/* Loss Curve Chart */}
          <div>
            <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
              <span className="font-semibold text-slate-700 dark:text-slate-300">
                Gradient Descent Loss Convergence Curve (Log-Loss vs Epoch)
              </span>
            </div>

            <div className="h-48 w-full bg-slate-50 dark:bg-slate-800/40 rounded-xl p-2 border border-slate-200/60 dark:border-slate-800">
              {logisticModel?.lossHistory ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={logisticModel.lossHistory} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                    <Tooltip
                      formatter={(val: any) => [Number(val).toFixed(4), 'Log-Loss']}
                      contentStyle={{
                        borderRadius: '8px',
                        backgroundColor: '#0f172a',
                        border: '1px solid #334155',
                        color: '#f8fafc',
                        fontSize: '11px'
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-xs text-slate-400">
                  Click 'Train Both Models' to plot convergence.
                </div>
              )}
            </div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2 border border-slate-200/60 dark:border-slate-700/60">
            <Info className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
            <div>
              <strong>Mathematical Form:</strong> Sigmoid activation &ycirc; = &sigma;(w<sup>T</sup>x + b). The L2 regularization penalty (&lambda; / 2m) &sum; w<sub>j</sub><sup>2</sup> prevents coefficient explosion on correlated clinical biomarkers.
            </div>
          </div>
        </div>

        {/* Model 2: Decision Tree Controls & Expandable Visualization */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <GitFork className="w-4 h-4 text-indigo-500" />
                <span>Decision Tree Hyperparameters (CART)</span>
              </h3>
              <p className="text-xs text-slate-500">Recursive binary partitioning based on impurity reduction</p>
            </div>
            <div className="flex gap-1">
              <button
                onClick={() => onUpdateTreeParams({ ...treeHyperparams, criterion: 'gini' })}
                className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase transition ${
                  treeHyperparams.criterion === 'gini'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                }`}
              >
                Gini
              </button>
              <button
                onClick={() => onUpdateTreeParams({ ...treeHyperparams, criterion: 'entropy' })}
                className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase transition ${
                  treeHyperparams.criterion === 'entropy'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                }`}
              >
                Entropy
              </button>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                Max Depth: {treeHyperparams.maxDepth}
              </label>
              <input
                type="range"
                min={2}
                max={8}
                step={1}
                value={treeHyperparams.maxDepth}
                onChange={e =>
                  onUpdateTreeParams({
                    ...treeHyperparams,
                    maxDepth: parseInt(e.target.value)
                  })
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            <div>
              <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                Min Split: {treeHyperparams.minSamplesSplit}
              </label>
              <input
                type="range"
                min={2}
                max={20}
                step={1}
                value={treeHyperparams.minSamplesSplit}
                onChange={e =>
                  onUpdateTreeParams({
                    ...treeHyperparams,
                    minSamplesSplit: parseInt(e.target.value)
                  })
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            <div>
              <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                Min Leaf: {treeHyperparams.minSamplesLeaf}
              </label>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={treeHyperparams.minSamplesLeaf}
                onChange={e =>
                  onUpdateTreeParams({
                    ...treeHyperparams,
                    minSamplesLeaf: parseInt(e.target.value)
                  })
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>
          </div>

          {/* Interactive Tree Structure Container */}
          <div>
            <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
              <span className="font-semibold text-slate-700 dark:text-slate-300">
                Interactive Hierarchical Tree Structure
              </span>
              <span className="text-[10px] text-slate-400">Click nodes to expand/collapse</span>
            </div>

            <div className="h-48 w-full overflow-y-auto bg-slate-50 dark:bg-slate-800/40 rounded-xl p-3 border border-slate-200/60 dark:border-slate-800">
              {treeModel?.root ? (
                <TreeNodeView node={treeModel.root} />
              ) : (
                <div className="h-full flex items-center justify-center text-xs text-slate-400">
                  Click 'Train Both Models' to visualize the decision tree.
                </div>
              )}
            </div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2 border border-slate-200/60 dark:border-slate-700/60">
            <Info className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
            <div>
              <strong>Impurity Formulation:</strong> Evaluates all candidate midpoints &theta;, picking the split maximizing Gini reduction &Delta;I = I(parent) - [(N<sub>L</sub> / N) I<sub>L</sub> + (N<sub>R</sub> / N) I<sub>R</sub>]. Capping tree depth prevents variance overfitting.
            </div>
          </div>
        </div>
      </div>

      {/* Grid Search Results Table (If Executed) */}
      {gridSearchResults && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-500" />
              <span>5-Fold Stratified Cross-Validation Hyperparameter Optimization (Grid Search)</span>
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Logistic Grid Results */}
            {gridSearchResults.logistic && (
              <div className="space-y-2">
                <div className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                  <span>Logistic Regression Search</span>
                  <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
                    Best: &alpha;={gridSearchResults.logistic.best.params.lr}, &lambda;={gridSearchResults.logistic.best.params.reg}
                  </span>
                </div>

                <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500 text-[11px]">
                      <tr>
                        <th className="px-3 py-2">Params (LR / L2)</th>
                        <th className="px-3 py-2">Mean CV AUC</th>
                        <th className="px-3 py-2">Mean Recall</th>
                        <th className="px-3 py-2">Mean Acc</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono text-[11px]">
                      {gridSearchResults.logistic.results.slice(0, 5).map((r, i) => (
                        <tr key={i} className={i === 0 ? 'bg-emerald-50/50 dark:bg-emerald-950/20 font-bold' : ''}>
                          <td className="px-3 py-1.5 font-sans">
                            &alpha;={r.params.lr} | &lambda;={r.params.reg}
                          </td>
                          <td className="px-3 py-1.5 text-rose-600 dark:text-rose-400">{r.meanRocAuc.toFixed(3)}</td>
                          <td className="px-3 py-1.5">{r.meanRecall.toFixed(3)}</td>
                          <td className="px-3 py-1.5">
                            {r.meanAccuracy.toFixed(3)} &plusmn; {r.stdAccuracy.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Tree Grid Results */}
            {gridSearchResults.tree && (
              <div className="space-y-2">
                <div className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                  <span>Decision Tree Search</span>
                  <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
                    Best: depth={gridSearchResults.tree.best.params.maxDepth}, crit={gridSearchResults.tree.best.params.criterion}
                  </span>
                </div>

                <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500 text-[11px]">
                      <tr>
                        <th className="px-3 py-2">Params (Depth / Crit)</th>
                        <th className="px-3 py-2">Mean CV AUC</th>
                        <th className="px-3 py-2">Mean Recall</th>
                        <th className="px-3 py-2">Mean Acc</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono text-[11px]">
                      {gridSearchResults.tree.results.slice(0, 5).map((r, i) => (
                        <tr key={i} className={i === 0 ? 'bg-emerald-50/50 dark:bg-emerald-950/20 font-bold' : ''}>
                          <td className="px-3 py-1.5 font-sans">
                            depth={r.params.maxDepth} | {r.params.criterion}
                          </td>
                          <td className="px-3 py-1.5 text-indigo-600 dark:text-indigo-400">{r.meanRocAuc.toFixed(3)}</td>
                          <td className="px-3 py-1.5">{r.meanRecall.toFixed(3)}</td>
                          <td className="px-3 py-1.5">
                            {r.meanAccuracy.toFixed(3)} &plusmn; {r.stdAccuracy.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
