# Financial Analysis of Indian Cities

## Overview
This project performs comprehensive data mining and machine learning analysis on financial data from Indian cities. The analysis includes exploratory data analysis, clustering, classification, model explainability using SHAP, and disparity analysis across different city tiers.

## Project Structure

├── data/
│ └── finance.csv # Raw financial dataset
├── outputs/
│ ├── phase1_processed.csv # Processed data after EDA
│ ├── phase2_clustered.csv # Data with cluster labels and train/test split marker
│ ├── phase3_model.pkl # Trained XGBoost classifier
│ ├── phase3_test_predictions.csv # XGBoost classification predictions on test set
│ ├── model_results.txt # XGBoost evaluation metrics
│ ├── random_forest_model.pkl # Trained Random Forest classifier
│ ├── random_forest_predictions.csv # Random Forest predictions on test set
│ ├── random_forest_results.txt # Random Forest evaluation metrics
│ ├── shap_feature_importance.txt # SHAP feature importance
│ └── shap_prediction_details.txt # SHAP prediction explanations
├── plots/ # Generated visualization plots
├── frontend/ # React frontend for the deployed demo
├── api.py # Flask API for live predictions (demo app)
├── dbscan.py # Standalone exploratory DBSCAN clustering
├── phase1_eda_feature_engineering.py # Exploratory Data Analysis & Feature Engineering
├── phase2_clustering.py # K-Means Clustering Analysis (leakage-safe split)
├── phase2_visualization.py # Clustering Visualization
├── phase3_classification.py # XGBoost Classification Model
├── phase4_shap_analysis.py # SHAP Explainability Analysis
├── phase5_disparity_analysis.py # Disparity Analysis Across City Tiers
├── random_forest_model.py # Random Forest Classification Model
├── requirements.txt # Python dependencies
├── LICENSE # Project license
└── README.md # This file


## Phases

### Phase 1: Exploratory Data Analysis & Feature Engineering
- Loads and examines the financial dataset
- Handles missing values using median imputation
- Performs feature engineering and data preprocessing
- Generates initial visualizations

### Phase 2: Clustering Analysis
- Splits the dataset into train and test partitions **before** any fitting occurs
- Fits `StandardScaler` and K-Means **only on the training partition**
- Applies the fitted scaler and K-Means model to the test partition using `.transform()` / `.predict()` — never re-fit
- Maps cluster indices to vulnerability labels (`Stable`, `At Risk`, `Vulnerable`) using a principled ranking (by mean `Expense_Ratio`), rather than a hardcoded index-to-label mapping
- Saves the train/test assignment as a `Split` column so downstream phases reuse the exact same partition

### Phase 3: Classification
- Trains an XGBoost classifier to predict city vulnerability levels
- Uses the same `Split` column generated in Phase 2, ensuring the classifier's test set was never seen during clustering or scaling
- Evaluates model performance on the held-out test partition
- Saves the trained model for later use

### Phase 4: SHAP Analysis
- Uses SHAP (SHapley Additive exPlanations) to explain model predictions
- Explains predictions on the **test partition only**, keeping the explanation consistent with genuinely held-out data
- Analyzes feature importance and individual prediction explanations
- Generates SHAP summary plots

### Phase 5: Disparity Analysis
- Analyzes financial disparities across different city tiers
- Performs statistical tests (ANOVA) to identify significant differences
- Examines expense ratios across city categories

### Random Forest Model (`random_forest_model.py`)
- An alternative classifier trained and evaluated using the same leakage-safe `Split` column from Phase 2
- Used as a cross-check against the XGBoost results in Phase 3

## Notes on Methodology

**Avoiding data leakage:** An earlier version of this pipeline fit the scaler and K-Means model on the entire dataset before splitting into train/test sets, which meant the test set's cluster-derived labels were influenced by the test set's own feature values. This has been corrected — the dataset is now split first, and the scaler and K-Means model are fit only on training data, then applied (not re-fit) to the test set.

**Why accuracy remains high after the fix:** The classification target (`Vulnerability_Label`) is derived from unsupervised K-Means clustering on the same features used for classification, rather than an independent, externally-verified ground truth. As a result, XGBoost and Random Forest are effectively learning to approximate the K-Means decision boundaries within the same feature space, which naturally yields high accuracy even without any leakage. This is expected behavior for a cluster-then-classify pipeline, not a sign of remaining leakage — a genuine deployment of this approach would ideally validate the cluster-derived labels against a real, independent outcome measure.

**Known limitation — live demo:** The `api.py` / `frontend/` demo app was built against an earlier, richer feature set (raw expense categories like `Rent`, `Groceries`, `Occupation`) that predates the current clustering-based pipeline, and currently expects a model/scaler pair that isn't produced by the scripts in this repo. The core analysis pipeline (Phases 1–5 and the Random Forest comparison) is fully functional and leakage-free; the live demo is a known, separate gap.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap
- joblib
- scipy

## Installation
1. Clone or download the repository
2. Create and activate a virtual environment (recommended):
```bash
   python3 -m venv venv
   source venv/bin/activate
```
3. Install required Python packages:
```bash
   pip install -r requirements.txt
```

## Usage
Run the phases sequentially:

1. **Phase 1**: `python phase1_eda_feature_engineering.py`
2. **Phase 2**: `python phase2_clustering.py` then `python phase2_visualization.py`
3. **Phase 3**: `python phase3_classification.py`
4. **Phase 4**: `python phase4_shap_analysis.py`
5. **Phase 5**: `python phase5_disparity_analysis.py`

Alternatively, run the Random Forest model: `python random_forest_model.py`

## Dataset
The analysis uses `data/finance.csv` containing financial metrics for Indian cities including:
- Income levels
- Expense ratios
- Savings gaps
- Recovery rates
- Demographic data (dependents, age)
- City tier classifications

## Outputs
- Processed datasets in `outputs/`
- Visualization plots in `plots/`
- Model evaluation results and SHAP analyses in text files

## License
See LICENSE file for details.

