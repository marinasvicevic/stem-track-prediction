# Classifying Students' STEM and Non-STEM Educational Tracks Using Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Replication code and materials for the paper:

> **Classifying Students' STEM and Non-STEM Educational Tracks Using Machine Learning: A Multi-Model Comparison with Interpretability and Fairness Assessment**

## Overview

This study applies machine learning to classify students into STEM and non-STEM gymnasium tracks based on survey data from 1,035 students across four grammar schools in Serbia. Five classification algorithms are compared (Logistic Regression, KNN, SVM, Random Forest, XGBoost), with comprehensive evaluation of predictive performance, model interpretability (SHAP, feature importance, LR coefficients), algorithmic fairness, and robustness.

### Key Findings

- All five ML models outperform the majority-class baseline classifier
- SVM achieves the highest accuracy (0.787) and weighted F1 (0.786)
- Random Forest achieves the highest ROC-AUC (0.812)
- No statistically significant differences between models (Nadeau-Bengio corrected t-test with Holm adjustment)
- Physics and mathematics interestingness, subject group preference, and gender are the strongest predictors across all interpretation methods
- Models show performance differences across gender subgroups, with implications for fairness-aware deployment

## Repository Structure

```
stem-track-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── STEM.xlsx                         # Anonymized survey dataset (1,044 responses)
├── src/
│   ├── stem_analysis.py          # Main computation script (results + CSV export)
│   └── stem_plots.py             # Figure generation script (reads results.pkl)
├── notebooks/
│   └── STEM_Analysis_CAEE.ipynb  # Interactive Jupyter/Colab notebook
├── figures/
│   ├── fig2_preprocessing_flowchart.svg  # Data preprocessing flowchart
│   └── fig3_pipeline.svg                 # ML pipeline diagram
└── tables/
    ├── table1_hyperparameters.js          # Table 1: Hyperparameter search spaces
    ├── table2_performance.js              # Table 2: Model performance comparison
    ├── table3_statistical_tests.js        # Table 3: Statistical significance tests
    ├── table4_fairness.js                 # Table 4: Performance by gender
    └── table5_robustness.js               # Table 5: Feature ablation robustness
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Node.js 16+ (only for generating Word tables)

### Installation

```bash
git clone https://github.com/marinasvicevic/stem-track-prediction.git
cd stem-track-prediction
pip install -r requirements.txt
```

### Dataset

The anonymized survey dataset is included in `data/STEM.xlsx` (1,044 responses, 105 survey items). All responses are anonymous — no personally identifiable information was collected. The data was collected under approval from the Ethics Committee of the Faculty of Science, University of Kragujevac (No. 01-06/24).

### Usage

#### Option 1: Python Scripts (Recommended for Reproducibility)

```bash
# Step 1: Run all computations (generates results.pkl + CSV files)
cd src
python stem_analysis.py

# Step 2: Generate all publication-quality figures
python stem_plots.py
```

The analysis script exports:
- `results.pkl` — Complete results (models, predictions, SHAP values, learning curves, ROC data)
- `table1_hyperparameters.csv` — Hyperparameter search spaces
- `table2_model_performance.csv` — Cross-validation and test set performance
- `table3_statistical_tests.csv` — Pairwise corrected t-test results
- `table4_synthetic_ranking.csv` — Top 15 features by synthetic rank
- `table5_lr_coefficients.csv` — Logistic regression coefficients and odds ratios
- `table6_fairness_by_gender.csv` — Model performance by gender
- `table7_robustness.csv` — Feature ablation results
- `table_bias_metrics.csv` — PPR, FPR, FNR by gender and urbanization

#### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/STEM_Analysis_CAEE.ipynb
```

The notebook is also compatible with [Google Colab](https://colab.research.google.com/) — upload `STEM.xlsx` when prompted.

#### Generating Word Tables (AMS Style)

```bash
cd tables
node table1_hyperparameters.js   # → table1_hyperparameters.docx
node table2_performance.js       # → table2_performance.docx
node table3_statistical_tests.js # → table3_statistical_tests.docx
node table4_fairness.js          # → table4_fairness.docx
node table5_robustness.js        # → table5_robustness.docx
```

Requires `docx` npm package: `npm install docx`

## Methodology

### Data Pipeline

1. **Data collection**: Survey of 1,045 students from four grammar schools in Serbia (Oct 2023 - Jul 2024)
2. **Preprocessing**: Invalid record removal, duplicate detection, straight-lining detection via Isolation Forest → 1,035 final observations
3. **Feature engineering**: Ordinal encoding, one-hot encoding (96 → 132 features), standardization within CV folds
4. **Train/test split**: Stratified 80:20 (828 train / 207 test), random_state=42
5. **Model training**: GridSearchCV with 5-fold stratified CV, weighted F1 optimization
6. **Statistical comparison**: 5x5 repeated nested CV with Nadeau-Bengio corrected t-test + Holm adjustment
7. **Interpretability**: LR coefficients, RF/XGBoost feature importance, SHAP (TreeExplainer)
8. **Fairness**: Performance by gender subgroups, PPR/FPR/FNR analysis
9. **Robustness**: Feature ablation (single predictor and feature block removal)

### Models

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | C, penalty (L1/L2) via ElasticNet |
| KNN | n_neighbors, weights, metric |
| SVM | C, kernel (linear/RBF) |
| Random Forest | n_estimators, max_depth, min_samples_split |
| XGBoost | n_estimators, max_depth, learning_rate |

All models include feature count *k* selected via ANOVA F-test as an additional hyperparameter.

## Figures

The paper includes 9 figures:

| Figure | Description | Source |
|--------|------------|--------|
| Fig. 1 | STEM vs Non-STEM distribution | `stem_plots.py` |
| Fig. 2 | Data preprocessing flowchart | `figures/fig2_preprocessing_flowchart.svg` |
| Fig. 3 | ML pipeline diagram | `figures/fig3_pipeline.svg` |
| Fig. 4 | ROC curves (all models) | `stem_plots.py` |
| Fig. 5 | Confusion matrix (SVM) | `stem_plots.py` |
| Fig. 6 | Learning curves | `stem_plots.py` |
| Fig. 7 | SHAP beeswarm plot | `stem_plots.py` |
| Fig. 8 | SHAP bar plot (mean \|SHAP\|) | `stem_plots.py` |
| Fig. 9 | RF vs XGBoost feature importance | `stem_plots.py` |

Figures 2 and 3 are vector diagrams (SVG) designed for editing in Inkscape.

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{svicevic2026stem,
  title={Classifying Students' {STEM} and Non-{STEM} Educational Tracks Using Machine Learning: A Multi-Model Comparison with Interpretability and Fairness Assessment},
  author={Svi\v{c}evi\'{c}, Marina and others},
  year={2026},
  url={https://github.com/marinasvicevic/stem-track-prediction}
}
```

## Related Publications

- Stasevic, F., Milenkovic, A., Maksimovic, A., Svicevic, M., & Krstic, L. (2025). Interest in STEM and Its Relations to the Educational Achievement and Socio-Demographic Characteristics of Grammar School Students in Serbia. *JBSE*, 24(2), 360-376.
- Maksimovic, A., Stasevic, F., & Milenkovic, A. (2025). Differences in Parental Involvement and Expectations during Primary Education: Reflections of STEM and Non-STEM Grammar School Students. *IJEMST*, 13(6), 1595-1614.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was approved by the Ethics Committee of the Faculty of Science, University of Kragujevac (No. 01-06/24). We thank all participating students and schools.
