/**
 * Model Export/Import Serialization and Python/Jupyter Notebook Code Generators
 */

import {
  DatasetMeta,
  PreprocessingConfig,
  PreprocessingSummary,
  LogisticModelArtifact,
  DecisionTreeModelArtifact,
  ModelMetrics,
  ExportedModelBundle
} from '../types';

/**
 * Creates the complete exportable JSON payload
 */
export function createModelBundle(
  metadata: DatasetMeta,
  preprocessingConfig: PreprocessingConfig,
  preprocessingSummary: PreprocessingSummary,
  logisticModel: LogisticModelArtifact | null,
  decisionTreeModel: DecisionTreeModelArtifact | null,
  logisticMetrics: ModelMetrics | null,
  decisionTreeMetrics: ModelMetrics | null,
  threshold: number
): ExportedModelBundle {
  return {
    version: '1.0.0-capstone',
    timestamp: new Date().toISOString(),
    metadata,
    preprocessingConfig,
    preprocessingSummary,
    logisticModel,
    decisionTreeModel,
    logisticMetrics,
    decisionTreeMetrics,
    threshold
  };
}

/**
 * Trigger browser file download for a string payload
 */
export function downloadFile(filename: string, content: string, contentType: string = 'application/json') {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Validate imported model bundle structure
 */
export function parseModelBundle(jsonString: string): ExportedModelBundle {
  const parsed = JSON.parse(jsonString);
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Invalid JSON format: root is not an object.');
  }
  if (!parsed.preprocessingSummary || !parsed.preprocessingConfig) {
    throw new Error('Missing preprocessing metadata in imported model package.');
  }
  if (!parsed.logisticModel && !parsed.decisionTreeModel) {
    throw new Error('Bundle must contain at least one trained model (Logistic Regression or Decision Tree).');
  }
  return parsed as ExportedModelBundle;
}

/**
 * Generate equivalent standalone Python scikit-learn script (.py)
 */
export function generatePythonScript(
  config: PreprocessingConfig,
  logisticHyperparams: { lr: number; epochs: number; reg: number },
  treeHyperparams: { maxDepth: number; criterion: string }
): string {
  return `"""
Heart Disease Prediction: End-to-End Machine Learning Pipeline
==============================================================
University Machine Learning Capstone - Scikit-Learn Reference Pipeline
Trained on Cleveland Heart Disease UCI Dataset

Demonstrates:
1. Data Cleaning & Imputation (median for numeric, mode for categorical)
2. Stratified Train/Test Split (Avoid Data Leakage)
3. Feature Scaling with StandardScaler fitted ONLY on training split
4. Logistic Regression with L2 Regularization & CART Decision Tree
5. Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
6. Model Persistence with Joblib and Pickle
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

def main():
    print("=" * 60)
    print("Heart Disease Prediction Pipeline (Scikit-Learn)")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. LOAD DATASET
    # -------------------------------------------------------------
    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found locally. Please place Kaggle heart.csv in current dir.")
        print("Using dummy dataset for demonstration...")
        # Create synthetic dataframe if file not present
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(30, 75, size=303),
            'sex': np.random.choice([0, 1], size=303),
            'cp': np.random.choice([0, 1, 2, 3], size=303),
            'trestbps': np.random.randint(100, 180, size=303),
            'chol': np.random.randint(150, 360, size=303),
            'fbs': np.random.choice([0, 1], size=303),
            'restecg': np.random.choice([0, 1, 2], size=303),
            'thalach': np.random.randint(90, 200, size=303),
            'exang': np.random.choice([0, 1], size=303),
            'oldpeak': np.round(np.random.uniform(0.0, 4.0, size=303), 1),
            'slope': np.random.choice([0, 1, 2], size=303),
            'ca': np.random.choice([0, 1, 2, 3], size=303),
            'thal': np.random.choice([1, 2, 3], size=303),
            'target': np.random.choice([0, 1], size=303)
        })
    else:
        df = pd.read_csv(csv_path)

    print(f"Raw dataset shape: {df.shape}")

    # Handle alternate column names and drop identifier columns
    drop_cols = [c for c in df.columns if c.lower() in ['id', 'dataset', 'unnamed: 0']]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped identifiers: {drop_cols}")

    # Identify target column: 'target' or 'num'
    target_col = 'target' if 'target' in df.columns else ('num' if 'num' in df.columns else df.columns[-1])
    if target_col == 'num':
        # Binarize 0 to 4 into 0 (no disease) and 1 (disease)
        df[target_col] = (df[target_col] > 0).astype(int)

    print(f"Target column: '{target_col}' | Class balance: {dict(df[target_col].value_counts())}")

    # -------------------------------------------------------------
    # 2. DATA CLEANING & PREPROCESSING
    # -------------------------------------------------------------
    ${config.removeDuplicates ? "df = df.drop_duplicates()\n    print(f'Duplicates removed. Remaining rows: {len(df)}')" : "# Duplicates kept as configured"}

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Categorize columns
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    if numeric_cols:
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_cols:
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # -------------------------------------------------------------
    # 3. STRATIFIED TRAIN/TEST SPLIT (CRITICAL TO PREVENT LEAKAGE)
    # -------------------------------------------------------------
    test_size = ${config.testSize}
    random_seed = ${config.randomSeed}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    print(f"Train split size: {len(X_train)} | Test split size: {len(X_test)}")

    # -------------------------------------------------------------
    # 4. FEATURE SCALING (FIT ON TRAIN ONLY!)
    # -------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if numeric_cols:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        # Apply fitted scaler to test set without refitting
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # -------------------------------------------------------------
    # 5. MODEL TRAINING
    # -------------------------------------------------------------
    print("\\nTraining Logistic Regression (C=1.0, L2 regularization)...")
    lr_model = LogisticRegression(
        C=${logisticHyperparams.reg > 0 ? (1 / (logisticHyperparams.reg * 10)).toFixed(2) : '1.0'},
        max_iter=${logisticHyperparams.epochs},
        random_state=random_seed
    )
    lr_model.fit(X_train_scaled, y_train)

    print("Training Decision Tree Classifier (CART)...")
    dt_model = DecisionTreeClassifier(
        max_depth=${treeHyperparams.maxDepth},
        criterion='${treeHyperparams.criterion}',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_seed
    )
    dt_model.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 6. EVALUATION ON UNSEEN TEST SET
    # -------------------------------------------------------------
    def evaluate_model(name, model, X_eval, y_true):
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\\n--- {name} Results ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}  (Clinical Sensitivity)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("Confusion Matrix:")
        print(f"  TN: {cm[0,0]}   FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}   TP: {cm[1,1]} (False Negatives missed disease: {cm[1,0]})")
        return acc, prec, rec, f1, auc

    lr_acc, lr_prec, lr_rec, lr_f1, lr_auc = evaluate_model(
        "Logistic Regression", lr_model, X_test_scaled, y_test
    )
    dt_acc, dt_prec, dt_rec, dt_f1, dt_auc = evaluate_model(
        "Decision Tree", dt_model, X_test, y_test
    )

    # -------------------------------------------------------------
    # 7. 5-FOLD STRATIFIED CROSS VALIDATION
    # -------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    print(f"\\nLogistic Regression 5-Fold Cross-Validation AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # -------------------------------------------------------------
    # 8. PERSISTENCE (JOBLIB & PICKLE)
    # -------------------------------------------------------------
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump({'model': lr_model, 'scaler': scaler, 'features': list(X.columns)}, "saved_models/logistic_pipeline.joblib")
    with open("saved_models/decision_tree.pkl", "wb") as f:
        pickle.dump(dt_model, f)

    print("\\nModels successfully saved to 'saved_models/' folder!")
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
`;
}

/**
 * Generate equivalent Jupyter Notebook (.ipynb JSON structure)
 */
export function generateJupyterNotebook(
  config: PreprocessingConfig,
  logisticHyperparams: { lr: number; epochs: number; reg: number },
  treeHyperparams: { maxDepth: number; criterion: string }
): string {
  const pythonScript = generatePythonScript(config, logisticHyperparams, treeHyperparams);

  const notebook = {
    cells: [
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '# Heart Disease Prediction: End-to-End ML Pipeline\n',
          '### University/Internship Machine Learning Capstone (Week 4 Deliverable)\n',
          '---\n',
          'This notebook implements an authentic end-to-end Machine Learning pipeline for coronary heart disease screening using the Cleveland Heart Disease UCI dataset.\n',
          '\n',
          '**Pipeline Workflow:**\n',
          '1. Exploratory Data Analysis & Schema Detection\n',
          '2. Preprocessing & Leakage-Free Stratified Train/Test Split\n',
          '3. Feature Scaling (StandardScaler fit on train only)\n',
          '4. Model Training: Logistic Regression & CART Decision Tree\n',
          '5. Model Evaluation: Confusion Matrix, ROC-AUC, Precision-Recall\n',
          '6. Model Interpretability & Deployment Artifacts'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          '# Install dependencies if running on Google Colab / JupyterLab\n',
          '!pip install -q pandas numpy scikit-learn matplotlib seaborn joblib'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 1. Imports & Global Reproducibility Configuration\n',
          'We set a fixed random state for reproducibility across all splits and initializations.'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          'import os\n',
          'import joblib\n',
          'import pickle\n',
          'import numpy as np\n',
          'import pandas as pd\n',
          'import matplotlib.pyplot as plt\n',
          'import seaborn as sns\n',
          '\n',
          'from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n',
          'from sklearn.preprocessing import StandardScaler\n',
          'from sklearn.impute import SimpleImputer\n',
          'from sklearn.linear_model import LogisticRegression\n',
          'from sklearn.tree import DecisionTreeClassifier, plot_tree\n',
          'from sklearn.metrics import (\n',
          '    accuracy_score, precision_score, recall_score, f1_score,\n',
          '    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report\n',
          ')\n',
          '\n',
          'RANDOM_SEED = ' + config.randomSeed + '\n',
          'np.random.seed(RANDOM_SEED)\n',
          'plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 2. Dataset Loading & Schema Auto-Detection\n',
          'We load the UCI Cleveland Heart Disease dataset containing 303 patient records and 14 clinical variables.'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          'url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"\n',
          'try:\n',
          '    df = pd.read_csv("heart.csv")\n',
          '    print("Loaded local heart.csv")\n',
          'except:\n',
          '    print("Downloading Cleveland dataset from mirror...")\n',
          '    df = pd.read_csv(url)\n',
          '\n',
          'print(f"Dataset shape: {df.shape}")\n',
          'df.head()'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 3. Leakage-Free Stratified Train/Test Split\n',
          '**Critical ML Concept**: We split the dataset *before* fitting the feature scaler. Fitting scalers on the entire dataset leads to *data leakage*, providing artificially optimistic test metrics.'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          'target_col = "target" if "target" in df.columns else "num"\n',
          'if target_col == "num":\n',
          '    df[target_col] = (df[target_col] > 0).astype(int)\n',
          '\n',
          'X = df.drop(columns=[target_col])\n',
          'y = df[target_col].values\n',
          '\n',
          'X_train, X_test, y_train, y_test = train_test_split(\n',
          '    X, y, test_size=' + config.testSize + ', random_state=RANDOM_SEED, stratify=y\n',
          ')\n',
          '\n',
          '# Feature Scaling: Fit strictly on X_train\n',
          'scaler = StandardScaler()\n',
          'numeric_cols = [c for c in ["age", "trestbps", "chol", "thalach", "oldpeak"] if c in X.columns]\n',
          '\n',
          'X_train_scaled = X_train.copy()\n',
          'X_test_scaled = X_test.copy()\n',
          'X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])\n',
          'X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])\n',
          '\n',
          'print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 4. Model Training: Logistic Regression & Decision Tree\n',
          'We train both models with matched hyperparameters.'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          '# 1. Logistic Regression\n',
          'lr = LogisticRegression(C=1.0, max_iter=' + logisticHyperparams.epochs + ', random_state=RANDOM_SEED)\n',
          'lr.fit(X_train_scaled, y_train)\n',
          '\n',
          '# 2. CART Decision Tree\n',
          'dt = DecisionTreeClassifier(max_depth=' + treeHyperparams.maxDepth + ', criterion="' + treeHyperparams.criterion + '", random_state=RANDOM_SEED)\n',
          'dt.fit(X_train, y_train)\n',
          '\n',
          'print("Both models trained successfully!")'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 5. Evaluation & Clinical Metrics\n',
          'In cardiovascular screening, **Recall (Sensitivity)** is paramount because a False Negative (failing to detect existing coronary disease) is far more dangerous than a False Positive.'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          'lr_probs = lr.predict_proba(X_test_scaled)[:, 1]\n',
          'dt_probs = dt.predict_proba(X_test)[:, 1]\n',
          '\n',
          'lr_preds = (lr_probs >= 0.5).astype(int)\n',
          'dt_preds = (dt_probs >= 0.5).astype(int)\n',
          '\n',
          'print("=== Logistic Regression Report ===")\n',
          'print(classification_report(y_test, lr_preds, target_names=["No Disease", "Disease"]))\n',
          '\n',
          'print("=== Decision Tree Report ===")\n',
          'print(classification_report(y_test, dt_preds, target_names=["No Disease", "Disease"]))'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 6. Visualizations: Confusion Matrix & ROC Curves'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          'fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n',
          '\n',
          '# ROC Curves\n',
          'fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)\n',
          'fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)\n',
          'auc_lr = roc_auc_score(y_test, lr_probs)\n',
          'auc_dt = roc_auc_score(y_test, dt_probs)\n',
          '\n',
          'axes[0].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="#2563eb", lw=2)\n',
          'axes[0].plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.3f})", color="#059669", lw=2)\n',
          'axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")\n',
          'axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")\n',
          'axes[0].set_xlabel("False Positive Rate (1 - Specificity)")\n',
          'axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")\n',
          'axes[0].legend(loc="lower right")\n',
          '\n',
          '# Confusion Matrix Heatmap for Logistic Regression\n',
          'cm = confusion_matrix(y_test, lr_preds)\n',
          'sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False,\n',
          '            xticklabels=["Pred: No", "Pred: Disease"], yticklabels=["Actual: No", "Actual: Disease"])\n',
          'axes[1].set_title("Logistic Regression Confusion Matrix", fontsize=12, fontweight="bold")\n',
          'plt.tight_layout()\n',
          'plt.show()'
        ]
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 7. Model Serialization\n',
          'Exporting the trained pipeline with Joblib for production deployment.'
        ]
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          'os.makedirs("models", exist_ok=True)\n',
          'bundle = {\n',
          '    "model": lr, \n',
          '    "scaler": scaler,\n',
          '    "feature_names": list(X.columns)\n',
          '}\n',
          'joblib.dump(bundle, "models/heart_disease_logistic.joblib")\n',
          'print("Pipeline saved successfully!")'
        ]
      }
    ],
    metadata: {
      language_info: { name: 'python' },
      orig_nbformat: 4
    },
    nbformat: 4,
    nbformat_minor: 2
  };

  return JSON.stringify(notebook, null, 2);
}
