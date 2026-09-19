import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  DatasetMeta,
  RawDataRow,
  PreprocessingConfig,
  PreprocessingSummary,
  ProcessedDataset,
  LogisticHyperparams,
  DecisionTreeHyperparams,
  LogisticModelArtifact,
  DecisionTreeModelArtifact,
  ModelMetrics,
  CrossValidationSummary,
  GridSearchResult,
  ExportedModelBundle
} from './types';
import {
  loadDefaultDataset,
  parseCsvString,
  getInitialDatasetMeta
} from './ml/dataLoader';
import { generateSyntheticDataset } from './data/heartData';
import {
  preprocessPipeline,
  DEFAULT_PREPROCESSING_CONFIG
} from './ml/preprocess';
import { trainLogisticRegression } from './ml/logistic';
import { trainDecisionTree } from './ml/decisionTree';
import { evaluateModelOnTest } from './ml/metrics';
import {
  runCrossValidation,
  runGridSearchLogistic,
  runGridSearchTree
} from './ml/crossValidation';

// UI Components
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { OverviewTab } from './components/OverviewTab';
import { EdaTab } from './components/EdaTab';
import { CleaningTab } from './components/CleaningTab';
import { TrainingTab } from './components/TrainingTab';
import { EvaluationTab } from './components/EvaluationTab';
import { InterpretabilityTab } from './components/InterpretabilityTab';
import { PredictTab } from './components/PredictTab';
import { SaveLoadTab } from './components/SaveLoadTab';
import { PythonTab } from './components/PythonTab';
import { ReportTab } from './components/ReportTab';
import { PresentationTab } from './components/PresentationTab';
import { AboutModal } from './components/AboutModal';

export default function App() {
  // Theme state (dark/light)
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('heart_ml_theme');
    return saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)
      ? 'dark'
      : 'light';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('heart_ml_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Active Tab
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [aboutOpen, setAboutOpen] = useState<boolean>(false);

  // Raw Dataset & Metadata
  const [rawDataset, setRawDataset] = useState<RawDataRow[]>(() => loadDefaultDataset().data);
  const [meta, setMeta] = useState<DatasetMeta>(() => getInitialDatasetMeta());

  // Preprocessing Config
  const [config, setConfig] = useState<PreprocessingConfig>(DEFAULT_PREPROCESSING_CONFIG);

  // Preprocessed Train & Test Datasets
  const [summary, setSummary] = useState<PreprocessingSummary | null>(null);
  const [processedTrain, setProcessedTrain] = useState<ProcessedDataset | null>(null);
  const [processedTest, setProcessedTest] = useState<ProcessedDataset | null>(null);

  // Model Hyperparameters
  const [logisticHyperparams, setLogisticHyperparams] = useState<LogisticHyperparams>({
    learningRate: 0.05,
    epochs: 250,
    regularization: 0.02
  });

  const [treeHyperparams, setTreeHyperparams] = useState<DecisionTreeHyperparams>({
    maxDepth: 4,
    minSamplesSplit: 4,
    minSamplesLeaf: 2,
    criterion: 'gini'
  });

  // Trained Model Artifacts
  const [logisticModel, setLogisticModel] = useState<LogisticModelArtifact | null>(null);
  const [treeModel, setTreeModel] = useState<DecisionTreeModelArtifact | null>(null);

  // Threshold (Default 0.50)
  const [threshold, setThreshold] = useState<number>(0.5);

  // Test Evaluation Metrics
  const [logisticMetrics, setLogisticMetrics] = useState<ModelMetrics | null>(null);
  const [treeMetrics, setTreeMetrics] = useState<ModelMetrics | null>(null);

  // 5-Fold Stratified Cross-Validation Summaries
  const [logisticCV, setLogisticCV] = useState<CrossValidationSummary | null>(null);
  const [treeCV, setTreeCV] = useState<CrossValidationSummary | null>(null);

  // Grid Search State
  const [isGridSearching, setIsGridSearching] = useState<boolean>(false);
  const [gridSearchResults, setGridSearchResults] = useState<{
    logistic?: { results: GridSearchResult[]; best: GridSearchResult };
    tree?: { results: GridSearchResult[]; best: GridSearchResult };
  } | null>(null);

  // Loading / Training Status
  const [isTraining, setIsTraining] = useState<boolean>(false);

  // 1. Run Preprocessing Pipeline
  const runPreprocessing = useCallback(
    (data: RawDataRow[], cfg: PreprocessingConfig) => {
      const result = preprocessPipeline(data, cfg);
      setSummary(result.summary);
      setProcessedTrain(result.trainDataset);
      setProcessedTest(result.testDataset);
      return result;
    },
    []
  );

  // 2. Train Models and Evaluate
  const executeTraining = useCallback(
    (
      trainData: ProcessedDataset,
      testData: ProcessedDataset,
      lrParams: LogisticHyperparams,
      dtParams: DecisionTreeHyperparams,
      cutoff: number
    ) => {
      setIsTraining(true);

      // Execute synchronous ML in microtask / setTimeout so UI renders loading state
      setTimeout(() => {
        // Train Logistic Regression
        const lrArtifact = trainLogisticRegression(
          trainData.X,
          trainData.y,
          lrParams.learningRate,
          lrParams.epochs,
          lrParams.regularization,
          trainData.featureNames
        );

        // Train Decision Tree
        const dtArtifact = trainDecisionTree(
          trainData.X,
          trainData.y,
          trainData.featureNames,
          dtParams
        );

        // Evaluate on Holdout Test Set
        const lrTestEval = evaluateModelOnTest(
          lrArtifact,
          testData.X,
          testData.y,
          trainData.X,
          trainData.y,
          cutoff
        );

        const dtTestEval = evaluateModelOnTest(
          dtArtifact,
          testData.X,
          testData.y,
          trainData.X,
          trainData.y,
          cutoff
        );

        // Run 5-fold Stratified Cross Validation
        const lrCvSummary = runCrossValidation(
          'logistic',
          trainData.X,
          trainData.y,
          5,
          lrParams
        );

        const dtCvSummary = runCrossValidation(
          'tree',
          trainData.X,
          trainData.y,
          5,
          undefined,
          dtParams
        );

        setLogisticModel(lrArtifact);
        setTreeModel(dtArtifact);
        setLogisticMetrics(lrTestEval);
        setTreeMetrics(dtTestEval);
        setLogisticCV(lrCvSummary);
        setTreeCV(dtCvSummary);
        setIsTraining(false);
      }, 30);
    },
    []
  );

  // Initial Load: Preprocess & Train automatically so user sees complete populated dashboard immediately
  useEffect(() => {
    const prep = runPreprocessing(rawDataset, config);
    executeTraining(
      prep.trainDataset,
      prep.testDataset,
      logisticHyperparams,
      treeHyperparams,
      threshold
    );
  }, []); // Run once on startup

  // Handle Threshold Change
  const handleUpdateThreshold = (newThreshold: number) => {
    setThreshold(newThreshold);
    if (logisticModel && treeModel && processedTest && processedTrain) {
      const lrEval = evaluateModelOnTest(
        logisticModel,
        processedTest.X,
        processedTest.y,
        processedTrain.X,
        processedTrain.y,
        newThreshold
      );
      const dtEval = evaluateModelOnTest(
        treeModel,
        processedTest.X,
        processedTest.y,
        processedTrain.X,
        processedTrain.y,
        newThreshold
      );
      setLogisticMetrics(lrEval);
      setTreeMetrics(dtEval);
    }
  };

  // Re-run pipeline when cleaning config is updated
  const handleApplyConfig = (newConfig: PreprocessingConfig) => {
    setConfig(newConfig);
    const prep = runPreprocessing(rawDataset, newConfig);
    executeTraining(
      prep.trainDataset,
      prep.testDataset,
      logisticHyperparams,
      treeHyperparams,
      threshold
    );
  };

  // Train button handler
  const handleTrainModels = () => {
    if (!processedTrain || !processedTest) {
      const prep = runPreprocessing(rawDataset, config);
      executeTraining(
        prep.trainDataset,
        prep.testDataset,
        logisticHyperparams,
        treeHyperparams,
        threshold
      );
    } else {
      executeTraining(
        processedTrain,
        processedTest,
        logisticHyperparams,
        treeHyperparams,
        threshold
      );
    }
  };

  // 5-Fold Grid Search Handler
  const handleRunGridSearch = () => {
    if (!processedTrain) return;
    setIsGridSearching(true);

    setTimeout(() => {
      const lrGrid = runGridSearchLogistic(
        processedTrain.X,
        processedTrain.y,
        [0.01, 0.05, 0.1],
        [0.0, 0.02, 0.1],
        200
      );

      const dtGrid = runGridSearchTree(
        processedTrain.X,
        processedTrain.y,
        processedTrain.featureNames,
        [3, 4, 5, 6],
        ['gini', 'entropy']
      );

      setGridSearchResults({
        logistic: lrGrid,
        tree: dtGrid
      });
      setIsGridSearching(false);
    }, 50);
  };

  // CSV Upload Handler
  const handleUploadCsv = (csvText: string, filename: string) => {
    try {
      const parsed = parseCsvString(csvText, filename);
      setRawDataset(parsed.data);
      setMeta(parsed.meta);
      const prep = runPreprocessing(parsed.data, config);
      executeTraining(
        prep.trainDataset,
        prep.testDataset,
        logisticHyperparams,
        treeHyperparams,
        threshold
      );
    } catch (err: any) {
      alert(`CSV Upload Error: ${err.message || 'Invalid CSV format'}`);
    }
  };

  // Reset to default dataset
  const handleResetDataset = () => {
    const { data: defaultData, meta: defaultMeta } = loadDefaultDataset();
    setRawDataset(defaultData);
    setMeta(defaultMeta);
    const prep = runPreprocessing(defaultData, config);
    executeTraining(
      prep.trainDataset,
      prep.testDataset,
      logisticHyperparams,
      treeHyperparams,
      threshold
    );
  };

  // Load synthetic demo dataset
  const handleLoadSynthetic = () => {
    const csv = generateSyntheticDataset(42);
    const parsed = parseCsvString(csv, 'Synthetic Heart Disease Dataset', 'synthetic');
    setRawDataset(parsed.data);
    setMeta(parsed.meta);
    const prep = runPreprocessing(parsed.data, config);
    executeTraining(
      prep.trainDataset,
      prep.testDataset,
      logisticHyperparams,
      treeHyperparams,
      threshold
    );
  };

  // Import Saved Model JSON Bundle
  const handleImportModel = (bundle: ExportedModelBundle) => {
    if (bundle.preprocessingConfig) {
      setConfig(bundle.preprocessingConfig);
    }
    if (bundle.threshold) {
      setThreshold(bundle.threshold);
    }
    if (bundle.models?.logistic) {
      setLogisticModel(bundle.models.logistic);
    }
    if (bundle.models?.tree) {
      setTreeModel(bundle.models.tree);
    }
    if (bundle.metrics?.logistic) {
      setLogisticMetrics(bundle.metrics.logistic);
    }
    if (bundle.metrics?.tree) {
      setTreeMetrics(bundle.metrics.tree);
    }
    // Switch to predict tab so user can test the restored model immediately
    setActiveTab('predict');
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-200">
      {/* Top Navigation Bar */}
      <Header
        meta={meta}
        isDarkMode={theme === 'dark'}
        theme={theme}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onToggleTheme={toggleTheme}
        onUploadCSV={handleUploadCsv}
        onSelectDatasetType={(t) => {
          if (t === 'synthetic') handleLoadSynthetic();
          else handleResetDataset();
        }}
        onOpenAbout={() => setAboutOpen(true)}
        onOpenPresentation={() => setActiveTab('presentation')}
      />

      {/* Main Layout Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col lg:flex-row gap-6">
        {/* Sidebar Status & Navigation */}
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          summary={summary}
          logisticMetrics={logisticMetrics}
          treeMetrics={treeMetrics}
          isTraining={isTraining}
          threshold={threshold}
          onUploadCsv={handleUploadCsv}
          onResetDataset={handleResetDataset}
          onTrainModels={handleTrainModels}
        />

        {/* Tab Content Area */}
        <main className="flex-1 min-w-0">
          {activeTab === 'overview' && (
            <OverviewTab
              meta={meta}
              data={rawDataset}
              summary={summary}
              onNavigate={setActiveTab}
            />
          )}

          {activeTab === 'eda' && (
            <EdaTab data={rawDataset} meta={meta} summary={summary} />
          )}

          {activeTab === 'cleaning' && (
            <CleaningTab
              config={config}
              summary={summary}
              onApplyConfig={handleApplyConfig}
            />
          )}

          {activeTab === 'training' && (
            <TrainingTab
              logisticHyperparams={logisticHyperparams}
              treeHyperparams={treeHyperparams}
              logisticModel={logisticModel}
              treeModel={treeModel}
              isTraining={isTraining}
              onUpdateLogisticParams={setLogisticHyperparams}
              onUpdateTreeParams={setTreeHyperparams}
              onTrainModels={handleTrainModels}
              onRunGridSearch={handleRunGridSearch}
              isGridSearching={isGridSearching}
              gridSearchResults={gridSearchResults}
            />
          )}

          {activeTab === 'evaluation' && (
            <EvaluationTab
              logisticMetrics={logisticMetrics}
              treeMetrics={treeMetrics}
              logisticCV={logisticCV}
              treeCV={treeCV}
              summary={summary}
              threshold={threshold}
              onUpdateThreshold={handleUpdateThreshold}
            />
          )}

          {activeTab === 'interpretability' && (
            <InterpretabilityTab
              logisticModel={logisticModel}
              treeModel={treeModel}
            />
          )}

          {activeTab === 'predict' && (
            <PredictTab
              logisticModel={logisticModel}
              treeModel={treeModel}
              summary={summary}
              config={config}
              threshold={threshold}
            />
          )}

          {activeTab === 'saveload' && (
            <SaveLoadTab
              meta={meta}
              config={config}
              summary={summary}
              logisticModel={logisticModel}
              treeModel={treeModel}
              logisticMetrics={logisticMetrics}
              treeMetrics={treeMetrics}
              threshold={threshold}
              onImportModel={handleImportModel}
            />
          )}

          {activeTab === 'python' && (
            <PythonTab
              config={config}
              logisticHyperparams={logisticHyperparams}
              treeHyperparams={treeHyperparams}
            />
          )}

          {activeTab === 'report' && (
            <ReportTab
              meta={meta}
              config={config}
              summary={summary}
              logisticMetrics={logisticMetrics}
              treeMetrics={treeMetrics}
              logisticCV={logisticCV}
              treeCV={treeCV}
              threshold={threshold}
            />
          )}

          {activeTab === 'presentation' && (
            <PresentationTab
              meta={meta}
              config={config}
              summary={summary}
              logisticModel={logisticModel}
              treeModel={treeModel}
              logisticMetrics={logisticMetrics}
              treeMetrics={treeMetrics}
              logisticCV={logisticCV}
              treeCV={treeCV}
              threshold={threshold}
            />
          )}
        </main>
      </div>

      {/* About Capstone Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
