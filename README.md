# Heart Disease Prediction: End-to-End Machine Learning Pipeline

An interactive, university capstone-grade Machine Learning web application built with TypeScript, React, and Tailwind CSS. The machine learning models—**Regularized Logistic Regression** and a **CART Decision Tree**—are implemented entirely from scratch in native TypeScript with zero external ML dependencies or backend calls.

---

## Features

- **Authentic UCI Cleveland Dataset**: Pre-loaded with the gold-standard 303-patient Cleveland Heart Disease dataset, with options to test algorithmic synthetic data or upload custom CSV datasets.
- **Exploratory Data Analysis (EDA)**:
  - Class balance distribution analysis
  - Feature distributions split by diagnosis
  - Pearson correlation matrix heatmap
  - Interquartile summary statistics (Q1, Median, Q3, IQR)
- **Data Preprocessing & Cleaning**:
  - Missing value imputation (Median/Mode or Mean)
  - Duplicate detection and removal
  - IQR outlier boundary capping ($1.5 \times \text{IQR}$)
  - One-Hot Encoding for categorical features (`cp`, `restecg`, `slope`, `thal`)
  - Feature scaling (Z-Score Standardization or Min-Max Normalization) fitted strictly on the training partition
  - Stratified train/test data splitting
- **From-Scratch Machine Learning Algorithms**:
  - **Logistic Regression**: Analytical batch gradient descent, $L_2$ regularization, cross-entropy loss tracking, and learning rate tuning.
  - **CART Decision Tree**: Recursive binary splitting with configurable impurity criteria (Gini Impurity & Shannon Entropy), max depth, and minimum leaf sample constraints.
- **Evaluation & Diagnostics**:
  - Confusion matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall / Sensitivity, Specificity, and $F_1$-Score
  - Interactive Classification Threshold Slider with real-time metric recalculation
  - ROC Curve with numerical trapezoidal ROC-AUC calculation
  - Precision-Recall Curve
  - 5-Fold Stratified Cross-Validation
  - Systematic Hyperparameter Grid Search
- **Clinical Interpretability**:
  - Odds Ratios ($\text{OR} = e^{w_j}$) derived from logistic coefficients
  - Decision Tree feature importances and human-readable decision path tracing
- **Interactive Patient Simulator**:
  - Pre-configured clinical archetypes (e.g., Healthy Runner, Borderline Asymptomatic, Multi-Vessel Angina)
  - Dynamic risk gauges and patient-specific feature contribution breakdowns
- **Capstone Deliverables**:
  - Full academic capstone report with Print / PDF export
  - 12-slide presentation deck with keyboard navigation and speaker notes
  - JSON model bundle export and import

---

## Tech Stack

- **Framework**: React 18+ & Vite
- **Language**: TypeScript (Strict Mode)
- **Styling**: Tailwind CSS
- **Visualizations**: Recharts & Lucide React
- **ML Engine**: 100% Client-Side Pure TypeScript (No Python, TensorFlow, or Scikit-Learn required)

---

## Quick Start

### Prerequisites
- [Node.js](https://nodejs.org/) (version 18 or higher recommended)
- `npm` or `yarn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   Open your browser to `http://localhost:3000` (or the port specified in terminal output).

4. Build for production:
   ```bash
   npm run build
   ```

---

## Project Structure

```
├── src/
│   ├── components/        # UI tabs (EDA, Cleaning, Training, Evaluation, Interpretability, Predict, Report, Presentation)
│   ├── data/              # UCI Cleveland dataset & synthetic data generator
│   ├── ml/                # Pure TypeScript ML algorithms
│   │   ├── logistic.ts         # Logistic regression with batch gradient descent
│   │   ├── decisionTree.ts     # CART decision tree with Gini/Entropy
│   │   ├── metrics.ts          # Confusion matrix, ROC-AUC, PR curve
│   │   ├── preprocess.ts       # Imputation, outlier handling, one-hot, scaling
│   │   ├── crossValidation.ts  # Stratified K-Fold CV & Grid Search
│   │   └── dataLoader.ts       # CSV parser and dataset loader
│   ├── report/            # Academic capstone report and presentation deck generators
│   ├── types.ts           # Shared TypeScript interfaces & types
│   ├── App.tsx            # Main application coordinator
│   └── main.tsx           # Application entry point
├── package.json
└── vite.config.ts
```

---

## License

MIT License. Designed for academic and educational demonstrations.
