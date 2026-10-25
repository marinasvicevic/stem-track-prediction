#!/usr/bin/env python3
"""
STEM Track Prediction — Computation Script
Journal paper: Computer Applications in Engineering Education (CAEE)

Usage:
    python stem_analysis.py          # runs all computation, saves results.pkl + CSV files
    python stem_plots.py             # reads results.pkl, generates all figures

Requirements:
    pip install numpy pandas scikit-learn xgboost shap scipy openpyxl
    (matplotlib and seaborn needed only for stem_plots.py)

Expects data/STEM.xlsx in the repository root.
"""

import re, warnings, os, pickle
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import t as student_t

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    GridSearchCV, cross_val_score, learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Output paths (all outputs go to repo root) ──
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

def _out(filename):
    """Return absolute path to a file in the repository root."""
    return os.path.join(REPO_ROOT, filename)

print("All imports successful.\n")

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING AND QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "STEM.xlsx")
if not os.path.isfile(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found.\n"
        "Ensure data/STEM.xlsx exists in the repository root."
    )

df = pd.read_excel(DATA_FILE)

# Shorten long column names for subject grades/perceptions
def rename_subject_columns(columns):
    renames = {}
    patterns = [
        (re.compile(r"^Најчешћа оцена из наведених предмета у основној школи ми је била: (\[.+\])$"), "Најчешћа оцена"),
        (re.compile(r"^Колико ти је поједини предмет био занимљив у основној школи\?\s+(\[.+\])$"), "Занимљив"),
        (re.compile(r"^Колико ти је поједини предмет био тежак у основној школи\?\s+(\[.+\])$"), "Тежак"),
        (re.compile(r"^Колико ти је поједини предмет био користан у основној школи\?\s+(\[.+\])$"), "Користан"),
    ]
    for col in columns:
        if not isinstance(col, str):
            continue
        for pat, prefix in patterns:
            m = pat.match(col)
            if m:
                renames[col] = f"{prefix} {m.group(1)}"
                break
    return renames

df.rename(columns=rename_subject_columns(df.columns), inplace=True)
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")

# 1.1 Grade column fix & anomaly check
bad_grade = df[df['Разред:'] % 1 != 0]
if len(bad_grade) > 0:
    df.loc[bad_grade.index, 'Разред:'] = df.loc[bad_grade.index, 'Разред:'].round()
    print(f"Fixed {len(bad_grade)} non-integer grade value(s).")

out_of_range = df[~df['Разред:'].isin([1, 2, 3, 4])]
print(f"Rows with grade outside 1-4: {len(out_of_range)}")

# 1.2 Target variable definition
stem_tracks = [
    'природно-математички',
    'специјализовано одељење за ученике са посебним способностима за информатику',
    'специјализовано одељење за ученике са посебним способностима за математику',
    'специјализовано одељење за ученике са посебним способностима за биологију и хемију',
    'специјализовано одељење за ученике са посебним способностима за физику',
]
df['СТЕМ'] = df['Смер:'].apply(lambda x: 1 if pd.notna(x) and x in stem_tracks else 0)
n_stem = (df['СТЕМ'] == 1).sum()
n_non  = (df['СТЕМ'] == 0).sum()
print(f"\nSTEM: {n_stem} ({n_stem/len(df)*100:.1f}%)")
print(f"Non-STEM: {n_non} ({n_non/len(df)*100:.1f}%)")

# (target distribution for plotting is saved after cleaning — see below)

# ═══════════════════════════════════════════════════════════════════════
# 2. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

# 2.1 Drop descriptive / free-text columns
cols_to_drop = [
    'Timestamp',
    'Занимање твоје мајке:',
    'Занимање твог оца:',
    'Зашто сте изабрали да упишите Гимназију?',
    'Зашто сте изабрали да упишете овај смер?',
]
X = df.drop(columns=cols_to_drop)
y = df['СТЕМ']

# Convert Likert responses to numeric
def likert_to_number(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, float) and x == int(x): return int(x)
    s = str(x).strip()
    if not s: return np.nan
    return int(s.split('-')[0].strip())

likert_cols = [c for c in X.columns
               if 'Процени' in c or c.startswith('Занимљив ')
               or c.startswith('Тежак ') or c.startswith('Користан ')]
for col in likert_cols:
    X[col] = X[col].apply(likert_to_number)
print(f"\nConverted {len(likert_cols)} Likert columns to numeric.")

# 2.2 Remove leaky / non-predictive columns
cols_remove_model = ['СТЕМ', 'Смер:', 'Похађам средњу школу:', 'Разред:', 'Година рођења:']
X = X.drop(columns=cols_remove_model)
print(f"Features after removal: {X.shape[1]}")

# 2.3 Duplicate & heuristic response-quality filtering (before splitting)
df_no_ts = df.drop(columns=['Timestamp'])
dup_idx = df_no_ts[df_no_ts.duplicated(keep='first')].index
df = df.drop(index=dup_idx)
X = X.drop(index=dup_idx)
y = y.drop(index=dup_idx)
print(f"Removed {len(dup_idx)} duplicate row(s). Rows: {len(df)}")

# Exploratory anomaly check: Isolation Forest + straight-lining
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
iso_labels = pd.Series(iso.fit_predict(X[numeric_cols]), index=X.index)
print(f"Isolation Forest anomalies: {(iso_labels == -1).sum()}")

# Heuristic: <=3 distinct values across heterogeneous numeric items.
# This is NOT a validated straight-lining definition; document sensitivity.
n_unique = X[numeric_cols].nunique(axis=1)
straight_liners = n_unique[n_unique <= 3]
print(f"Straight-liners (≤3 distinct values): {len(straight_liners)}")

df = df.drop(index=straight_liners.index)
X = X.drop(index=straight_liners.index)
y = y.drop(index=straight_liners.index)
print(f"Final dataset size: {len(X)} observations")

# Save target distribution for plotting (after all cleaning steps)
target_counts = y.value_counts().sort_index()
n_total_before_cleaning = len(y)

# 2.4 Encoding categorical variables
children_col = 'Колико има деце у твојој породици, рачунајући и тебе (упиши број):'
X[children_col] = pd.to_numeric(X[children_col], errors='coerce')

# Missing-value imputation is fitted only on training folds inside the pipeline.

urbanization_map = {'селу': 1, 'варошици': 2, 'мањем граду': 3, 'већем граду': 4}
education_map = {'завршена основна школа': 1, 'завршена средња школа': 2,
                 'завршена виша школа': 3, 'завршен факултет': 4}
student_map = {'добар/а': 3, 'врло добар/а': 4, 'одличан/а': 5}

X['Одрастао/ла сам у:'] = X['Одрастао/ла сам у:'].map(urbanization_map)
X['Основну школу сам завршио/ла у:'] = X['Основну школу сам завршио/ла у:'].map(urbanization_map)
X['Стручна спрема твоје мајке:'] = X['Стручна спрема твоје мајке:'].map(education_map)
X['Стручна спрема твог оца:'] = X['Стручна спрема твог оца:'].map(education_map)
X['Какав си био/ла ђак у основној школи?'] = X['Какав си био/ла ђак у основној школи?'].map(student_map)

nominal_cols = [
    'Пол:',
    'Најомиљенији предмет:',
    'Други омиљени предмет (омиљени предмет, мање омиљени од претходног):',
    'Трећи омиљени предмет (омиљени предмет, мање омиљени од претходна два):',
]
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
print(f"Total features after encoding: {X.shape[1]}")
print(f"Remaining object columns: {X.select_dtypes(include='object').columns.tolist()}")
print(f"Missing values: {X.isna().sum().sum()}")

# 2.5 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"STEM proportion — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════
# 3. ML PIPELINE AND MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════

# 3.1 Pipeline construction
binary_cols = X_train.select_dtypes(include='bool').columns.tolist()
# SimpleImputer does not accept bool dtype — cast to int8
X_train[binary_cols] = X_train[binary_cols].astype('int8')
X_test[binary_cols] = X_test[binary_cols].astype('int8')
numeric_cols_scale = [c for c in X_train.columns if c not in binary_cols]
print(f"\nContinuous features (scaled): {len(numeric_cols_scale)}")
print(f"Binary features (passthrough): {len(binary_cols)}")

def make_pipeline(model, num_cols=None, bin_cols=None):
    num_cols = numeric_cols_scale if num_cols is None else num_cols
    bin_cols = binary_cols if bin_cols is None else bin_cols
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), num_cols),
        ('binary', Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), bin_cols),
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_classif)),
        ('model', model),
    ])

# 3.2 Cross-validation setup
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"\nStratified {N_SPLITS}-Fold Cross-Validation:")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    stem_prop = y_train.iloc[val_idx].mean()
    print(f"  Fold {fold}: {len(tr_idx)} train / {len(val_idx)} val, STEM = {stem_prop:.3f}")

# 3.3 Baseline model
baseline = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
y_proba_base = baseline.predict_proba(X_test)[:, 1]
print(f"\nBaseline accuracy: {accuracy_score(y_test, y_pred_base):.3f}")
print(f"Baseline F1 (weighted): {f1_score(y_test, y_pred_base, average='weighted'):.3f}")

# 3.4 Hyperparameter tuning — all five models

# Table 1: Hyperparameter search spaces (not best parameters)
table1_data = {
    'Model': ['Logistic Regression', 'Logistic Regression', 'Logistic Regression',
              'KNN', 'KNN', 'KNN', 'KNN',
              'SVM', 'SVM', 'SVM',
              'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest',
              'XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
    'Hyperparameter': ['C', 'l1_ratio (penalty)', 'k (features)',
                       'n_neighbors', 'weights', 'metric', 'k (features)',
                       'C', 'kernel', 'k (features)',
                       'n_estimators', 'max_depth', 'min_samples_split', 'k (features)',
                       'n_estimators', 'max_depth', 'learning_rate', 'k (features)'],
    'Search Range': ['{0.01, 0.1, 1, 10, 100}', '{0 (L2), 1 (L1)}', '{10, 30, 50, 70, 90, all}',
                     '{3, 5, 7, 9, 11, 15}', '{uniform, distance}', '{euclidean, manhattan}', '{10, 30, 50, 70, 90, all}',
                     '{0.1, 1, 10}', '{linear, rbf}', '{10, 30, 50, 70, 90, all}',
                     '{100, 200}', '{None, 10, 20}', '{2, 5}', '{10, 30, 50, 70, 90, all}',
                     '{100, 200}', '{3, 5}', '{0.05, 0.1, 0.2}', '{10, 30, 50, 70, 90, all}'],
}
table1 = pd.DataFrame(table1_data)
print("\n" + "="*60)
print("Table 1. Hyperparameter Search Spaces")
print("="*60)
print(table1.to_string(index=False))
table1.to_csv(_out('table1_hyperparameters.csv'), index=False)
print("  → saved table1_hyperparameters.csv")

# ── Train all five models ──
models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_STATE),
        'params': {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__l1_ratio': [1, 0],
            'selector__k': [10, 30, 50, 70, 90, 'all'],
        },
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9, 11, 15],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan'],
            'selector__k': [10, 30, 50, 70, 90, 'all'],
        },
    },
    'SVM': {
        'model': SVC(probability=True, random_state=RANDOM_STATE),
        'params': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf'],
            'selector__k': [10, 30, 50, 70, 90, 'all'],
        },
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'selector__k': [10, 30, 50, 70, 90, 'all'],
        },
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'selector__k': [10, 30, 50, 70, 90, 'all'],
        },
    },
}

model_order = ['Logistic Regression', 'KNN', 'SVM', 'Random Forest', 'XGBoost']
best_estimators = {}
cv_results = {}
test_results = {}
predictions = {}

for name, cfg in models_config.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    gs = GridSearchCV(
        estimator=make_pipeline(cfg['model']),
        param_grid=cfg['params'],
        scoring='f1_weighted',
        cv=skf,
        n_jobs=1,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    bi = gs.best_index_
    cv_mean = gs.cv_results_['mean_test_score'][bi]
    cv_std = gs.cv_results_['std_test_score'][bi]

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    best_estimators[name] = best
    cv_results[name] = (cv_mean, cv_std)
    predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}
    test_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 (weighted)': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }

    print(f"Best params: {gs.best_params_}")
    print(f"CV F1 (weighted): {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"Test Accuracy: {test_results[name]['Accuracy']:.3f}")
    print(f"Test ROC-AUC: {test_results[name]['ROC-AUC']:.3f}")

# Baseline results
test_results['Baseline (Dummy)'] = {
    'Accuracy': accuracy_score(y_test, y_pred_base),
    'Precision': precision_score(y_test, y_pred_base, zero_division=0),
    'Recall': recall_score(y_test, y_pred_base, zero_division=0),
    'F1 (weighted)': f1_score(y_test, y_pred_base, average='weighted', zero_division=0),
    'F1 (macro)': f1_score(y_test, y_pred_base, average='macro', zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_proba_base),
}
predictions['Baseline (Dummy)'] = {'y_pred': y_pred_base, 'y_proba': y_proba_base}
print("\n✓ All models trained successfully.")

# ═══════════════════════════════════════════════════════════════════════
# 4. RESULTS AND COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

# Table 2: Comparative model performance
comparison_df = pd.DataFrame(test_results).T.round(3)
comparison_df.insert(0, 'CV F1 (w) ± SD', [
    f"{cv_results[m][0]:.3f} ± {cv_results[m][1]:.3f}" if m in cv_results else '—'
    for m in comparison_df.index
])
comparison_df = comparison_df.sort_values('F1 (weighted)', ascending=False)
print("\n" + "="*60)
print("Table 2. Comparative Model Performance")
print("="*60)
print(comparison_df.to_string())
comparison_df.to_csv(_out('table2_model_performance.csv'))
print("  → saved table2_model_performance.csv")

best_model_name = max(model_order, key=lambda n: test_results[n]['F1 (weighted)'])
print(f"\nHighest observed holdout F1 (descriptive, not a selection protocol): {best_model_name}")

# ═══════════════════════════════════════════════════════════════════════
# 5. REPEATED NESTED CV AND APPROXIMATE STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# The nested-CV grids below were intentionally narrower than the main
# GridSearchCV grids in the original experiment. They are retained unchanged
# for reproducibility; comparisons concern these nested-CV procedures and
# should not be described as tests of the exact main-search configurations.
inner_grids = {
    'Logistic Regression': {
        'model__C': [0.1, 1, 10],
        'model__l1_ratio': [0, 1],
        'selector__k': [30, 70, 'all'],
    },
    'KNN': {
        'model__n_neighbors': [5, 9, 15],
        'model__weights': ['uniform', 'distance'],
        'selector__k': [30, 70, 'all'],
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'selector__k': [30, 70, 'all'],
    },
    'Random Forest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10],
        'selector__k': [30, 70, 'all'],
    },
    'XGBoost': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.05, 0.1],
        'selector__k': [30, 70, 'all'],
    },
}

print("\nRunning 5×5 repeated stratified nested CV (this may take several minutes)...")
model_scores_rcv = {}
for name in model_order:
    cfg = models_config[name]
    nested_est = GridSearchCV(
        estimator=make_pipeline(cfg['model']),
        param_grid=inner_grids[name],
        cv=inner_cv,
        scoring='f1_weighted',
        n_jobs=1,
    )
    scores = cross_val_score(
        nested_est, X_train, y_train, cv=rskf,
        scoring='f1_weighted', n_jobs=1,
    )
    model_scores_rcv[name] = scores
    print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")
print("✓ Repeated CV complete.")

# Approximate corrected resampled paired t-tests (Nadeau & Bengio correction).
outer_splits = list(RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=RANDOM_STATE
).split(X_train, y_train))
mean_test_train_ratio = float(np.mean([
    len(test_idx) / len(train_idx) for train_idx, test_idx in outer_splits
]))
n_outer = len(outer_splits)
correction = 1.0 / n_outer + mean_test_train_ratio

pairwise_results = []
for m1, m2 in combinations(model_order, 2):
    diffs = np.asarray(model_scores_rcv[m1]) - np.asarray(model_scores_rcv[m2])
    if len(diffs) != n_outer:
        raise ValueError('Unexpected outer-CV score count; cannot pair folds.')
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    corrected_se = float(np.sqrt(correction) * sd_diff)
    if corrected_se == 0:
        t_stat = 0.0 if mean_diff == 0 else np.nan
        p_raw = 1.0 if mean_diff == 0 else np.nan
    else:
        t_stat = mean_diff / corrected_se
        p_raw = float(2 * student_t.sf(abs(t_stat), df=n_outer - 1))
    ci_half = float(student_t.ppf(0.975, df=n_outer - 1) * corrected_se)
    pairwise_results.append({
        'Model A': m1,
        'Model B': m2,
        'Mean F1 difference (A-B)': mean_diff,
        'SD of paired outer-fold differences': sd_diff,
        'Corrected SE': corrected_se,
        'Corrected t statistic': t_stat,
        'df': n_outer - 1,
        'Raw p (approximate)': p_raw,
        'Approx. 95% CI lower': mean_diff - ci_half,
        'Approx. 95% CI upper': mean_diff + ci_half,
        'A higher in folds (of 25)': int((diffs > 0).sum()),
    })

# Holm step-down adjustment
raw_ps = np.array([row['Raw p (approximate)'] for row in pairwise_results])
valid = np.flatnonzero(np.isfinite(raw_ps))
order = valid[np.argsort(raw_ps[valid])]
adjusted = np.full(len(raw_ps), np.nan)
running_max = 0.0
m = len(raw_ps)
for rank, idx in enumerate(order):
    running_max = max(running_max, (m - rank) * raw_ps[idx])
    adjusted[idx] = min(1.0, running_max)
for row, p_holm in zip(pairwise_results, adjusted):
    row['Holm-adjusted p (approximate)'] = p_holm
    row['Significant at 0.05 after Holm (approx.)'] = (
        bool(p_holm < 0.05) if np.isfinite(p_holm) else False
    )

table3 = pd.DataFrame(pairwise_results)
print('\nTable 3. Approximate corrected resampled paired t-tests (Holm-adjusted)')
print(table3.round(4).to_string(index=False))
table3.to_csv(_out('table3_statistical_tests.csv'), index=False)

with open('table3_statistical_method_note.txt', 'w', encoding='utf-8') as f:
    f.write(
        'Paired comparisons used the same 5x5 repeated stratified outer folds '
        'and nested 3-fold hyperparameter tuning within each outer training fold.\n'
        'Approximate Nadeau-Bengio corrected resampled paired t-test: '
        'SE = SD(differences) * sqrt(1/N + mean(n_test/n_train)); '
        'N = 25; degrees of freedom = N - 1.\n'
        'Ten two-sided pairwise p-values were adjusted using Holm correction.\n'
        'Because repeated folds share observations, these p-values and confidence '
        'intervals are approximate and should be interpreted cautiously. '
        'The analysis uses training data only, not the held-out test set.\n'
        'Nested-CV hyperparameter grids are narrower than the main GridSearchCV '
        'grids; the tests compare the nested-CV procedures, not the exact main '
        'search configurations.\n'
        'No causal or definitive superiority claims should be based on these tests.\n'
    )
print('✓ Approximate paired tests and method note saved.')

# ═══════════════════════════════════════════════════════════════════════
# 6. LEARNING CURVES  (save data — plotting done by stem_plots.py)
# ═══════════════════════════════════════════════════════════════════════

train_sizes_frac = np.linspace(0.1, 1.0, 10)
learning_curve_data = {}

for name in model_order:
    est = make_pipeline(models_config[name]['model'])
    est.set_params(selector__k='all')
    train_sizes_abs, train_scores, val_scores = learning_curve(
        est, X_train, y_train,
        train_sizes=train_sizes_frac,
        cv=skf,
        scoring='f1_weighted',
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    learning_curve_data[name] = {
        'train_sizes': train_sizes_abs,
        'val_mean': val_scores.mean(axis=1),
        'val_std': val_scores.std(axis=1),
    }

print("\n✓ Learning curve data computed.")

# ═══════════════════════════════════════════════════════════════════════
# 7. MODEL INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════

# ── English feature-name mapping ──
FEAT_EN = {
    # Subject interestingness
    'Занимљив [Физика]': 'Physics interestingness',
    'Занимљив [Математика]': 'Math interestingness',
    'Занимљив [Хемија]': 'Chemistry interestingness',
    'Занимљив [Биологија]': 'Biology interestingness',
    'Занимљив [Информатика]': 'Informatics interestingness',
    'Занимљив [Српски језик и књижевност]': 'Serbian lang. interestingness',
    'Занимљив [Историја]': 'History interestingness',
    'Занимљив [Географија]': 'Geography interestingness',
    'Занимљив [Енглески језик]': 'English lang. interestingness',
    'Занимљив [2. страни језик]': '2nd foreign lang. interestingness',
    'Занимљив [Музичка култура]': 'Music interestingness',
    'Занимљив [Ликовна култура]': 'Art interestingness',
    'Занимљив [Физичко васпитање]': 'PE interestingness',
    'Занимљив [Техника и технологија]': 'Tech. & technology interestingness',
    # Subject difficulty
    'Тежак [Физика]': 'Physics difficulty',
    'Тежак [Математика]': 'Math difficulty',
    'Тежак [Хемија]': 'Chemistry difficulty',
    'Тежак [Биологија]': 'Biology difficulty',
    'Тежак [Информатика]': 'Informatics difficulty',
    'Тежак [Српски језик и књижевност]': 'Serbian lang. difficulty',
    'Тежак [Историја]': 'History difficulty',
    'Тежак [Географија]': 'Geography difficulty',
    'Тежак [Енглески језик]': 'English lang. difficulty',
    'Тежак [2. страни језик]': '2nd foreign lang. difficulty',
    'Тежак [Музичка култура]': 'Music difficulty',
    'Тежак [Ликовна култура]': 'Art difficulty',
    'Тежак [Физичко васпитање]': 'PE difficulty',
    'Тежак [Техника и технологија]': 'Tech. & technology difficulty',
    # Subject usefulness
    'Користан [Физика]': 'Physics usefulness',
    'Користан [Математика]': 'Math usefulness',
    'Користан [Хемија]': 'Chemistry usefulness',
    'Користан [Биологија]': 'Biology usefulness',
    'Користан [Информатика]': 'Informatics usefulness',
    'Користан [Српски језик и књижевност]': 'Serbian lang. usefulness',
    'Користан [Историја]': 'History usefulness',
    'Користан [Географија]': 'Geography usefulness',
    'Користан [Енглески језик]': 'English lang. usefulness',
    'Користан [2. страни језик]': '2nd foreign lang. usefulness',
    'Користан [Музичка култура]': 'Music usefulness',
    'Користан [Ликовна култура]': 'Art usefulness',
    'Користан [Физичко васпитање]': 'PE usefulness',
    'Користан [Техника и технологија]': 'Tech. & technology usefulness',
    # Subject grades
    'Најчешћа оцена [Математика]': 'Math grade',
    'Најчешћа оцена [Физика]': 'Physics grade',
    'Најчешћа оцена [Информатика]': 'Informatics grade',
    'Најчешћа оцена [Хемија]': 'Chemistry grade',
    'Најчешћа оцена [Биологија]': 'Biology grade',
    'Најчешћа оцена [Српски језик и књижевност]': 'Serbian lang. grade',
    'Најчешћа оцена [Историја]': 'History grade',
    'Најчешћа оцена [Географија]': 'Geography grade',
    'Најчешћа оцена [Енглески језик]': 'English lang. grade',
    'Најчешћа оцена [2. страни језик]': '2nd foreign lang. grade',
    'Најчешћа оцена [Музичка култура]': 'Music grade',
    'Најчешћа оцена [Ликовна култура]': 'Art grade',
    'Најчешћа оцена [Физичко васпитање]': 'PE grade',
    'Најчешћа оцена [Техника и технологија]': 'Tech. & technology grade',
    # Demographics & background
    'Пол:_мушки': 'Gender (male)',
    'Одрастао/ла сам у:': 'Grew up in (urbanization)',
    'Основну школу сам завршио/ла у:': 'Primary school location',
    'Стручна спрема твоје мајке:': "Mother's education",
    'Стручна спрема твог оца:': "Father's education",
    'Какав си био/ла ђак у основној школи?': 'Academic achievement',
    'Колико има деце у твојој породици, рачунајући и тебе (упиши број):': 'Number of children',
}

# Subject name translation (for one-hot encoded favourite subjects)
SUBJECT_EN = {
    'Математика': 'Mathematics', 'Физика': 'Physics', 'Хемија': 'Chemistry',
    'Биологија': 'Biology', 'Информатика': 'Informatics', 'Историја': 'History',
    'Географија': 'Geography', 'Енглески језик': 'English lang.',
    '2. страни језик': '2nd foreign lang.',
    'Српски језик и књижевност': 'Serbian lang.',
    'Музичка култура': 'Music', 'Ликовна култура': 'Art',
    'Физичко васпитање': 'PE', 'Техника и технологија': 'Tech. & technology',
}

# Translate enrollment-question bracket options.
ENROLL_OPTION_EN = {
    'Породица': 'Family',
    'Вршњаци': 'Peers',
    'Наставник': 'Teacher',
    'Медији (телевизија, друштвене мреже, интернет, блогови итд)': 'Media',
    'Склоност ка одређеној групи предмета': 'Affinity for a subject group',
    'Добра проходност на факултет који желим да студирам по завршетку средње школе': 'Access to desired university program',
    'Интересовање за занимање којим желим да се бавим у будућности': 'Interest in future occupation',
}

ENROLL_FACTOR_KW = [
    ('жеља за стицањем знања', 'Enrollment: desire for knowledge'),
    ('лакше запошљавање', 'Enrollment: easier employment'),
    ('утицај родитеља', 'Enrollment: parental influence'),
    ('жеље и интересовања', 'Enrollment: personal wishes & interests'),
    ('могућност студирања', 'Enrollment: university access'),
    ('занимање у будућности', 'Enrollment: future occupation'),
    ('утицај вршњака', 'Enrollment: peer influence'),
    ('утицај наставника', 'Enrollment: teacher influence'),
    ('утицај медија', 'Enrollment: media influence'),
    ('друштвене мреже', 'Enrollment: social media influence'),
    ('склоност ка предметима', 'Enrollment: subject affinity'),
    ('утицај породице', 'Enrollment: family influence'),
]
OTHER_LIKERT_EN = {
    'Процени подршку у учењу омиљеног предмета': 'Primary-school teacher support',
    'Процени навике у учењу': 'Primary-school study habits',
    'Процени укљученост родитеља': 'Parental/peer involvement',
    'Процени очекивања својих родитеља': 'Parental expectations',
    'Процени колико је ко од запослених': 'School career guidance',
}

LIKERT_BRACKET_EN = {
    'Самостално сам организовао/ла своје време и активности које су се односиле на учење и домаће задатке.':
        'Self-organised study time',
    'Волео/ла сам да учим': 'Enjoyed studying',
    'Учио/ла сам редовно': 'Studied regularly',
    'Био/ла сам истрајан/на у учењу': 'Persistent in studying',
    'Имао/ла сам јасне циљеве у вези учења': 'Had clear learning goals',
    'Родитељи су ми помагали у изради домаћих задатака.':
        'Parents helped with homework',
    'Родитељи су разговарали са мном о школи.':
        'Parents discussed school',
    'Родитељи су ме подстицали да учим.':
        'Parents encouraged studying',
    'Друг/другарица ми је помагао/ла у учењу.':
        'Friend helped with studying',
    'Друг/другарица и ја смо заједно учили.':
        'Studied with a friend',
    'Родитељи су ме подстицали да будем бољи.':
        'Parents encouraged improvement',
    'Родитељи су имали висока очекивања у вези мојег школског успеха.':
        'Parents had high expectations',
    'Родитељи су ме подстицали да постигнем одличан успех.':
        'Parents pushed for top grades',
    'Школски психолог/педагог': 'School psychologist/pedagogue',
    'Одељењски старешина': 'Homeroom teacher',
    'Наставник/ца неког предмета': 'Subject teacher',
    'Родитељи': 'Parents',
    'Вршњаци': 'Peers',
    'Наставник/ца ми је поклањао/ла више пажње': 'Teacher gave extra attention',
    'Наставник/ца ме је подстицао/ла да више учим': 'Teacher encouraged more studying',
    'Наставник/ца ми је давао/ла додатне задатке': 'Teacher gave extra tasks',
}

def translate_feature(name):
    """Translate Serbian feature name to English — complete translation."""
    if name in FEAT_EN:
        return FEAT_EN[name]
    for prefix_sr, prefix_en in [
        ('Најомиљенији предмет:_', 'Fav. subject: '),
        ('Други омиљени предмет', '2nd fav. subject: '),
        ('Трећи омиљени предмет', '3rd fav. subject: '),
    ]:
        if name.startswith(prefix_sr):
            suffix = name[len(prefix_sr):].strip('_ ')
            suffix_en = SUBJECT_EN.get(suffix, suffix)
            return prefix_en + suffix_en
    if name.startswith('Процени који од фактора је имао колико утицаја'):
        match = re.search(r'\[([^\]]+)\]\s*$', name)
        if match:
            bracket = match.group(1)
            if bracket in ENROLL_OPTION_EN:
                return 'Enrollment: ' + ENROLL_OPTION_EN[bracket]
            for kw, en in ENROLL_FACTOR_KW:
                if kw in bracket.lower():
                    return en
            return 'Enrollment: ' + bracket[:60]
        name_lower = name.lower()
        for kw, en in ENROLL_FACTOR_KW:
            if kw in name_lower:
                return en
        return 'Enrollment: ' + name[-65:]
    for prefix, english in OTHER_LIKERT_EN.items():
        if name.startswith(prefix):
            match = re.search(r'\[([^\]]+)\]\s*$', name)
            if match:
                bracket = match.group(1).rstrip('.')
                bracket_en = LIKERT_BRACKET_EN.get(
                    match.group(1),
                    LIKERT_BRACKET_EN.get(bracket, bracket))
                return english + ': ' + bracket_en
            return english + ': ' + name[-55:]
    return name

def get_selected_features(pipe):
    names = pipe.named_steps['preprocessor'].get_feature_names_out()
    mask = pipe.named_steps['selector'].get_support()
    return [n.split('__', 1)[1] for n, keep in zip(names, mask) if keep]

# 7.1 Logistic Regression coefficients and odds ratios (Table 5)
lr_best = best_estimators['Logistic Regression']
coef_df = pd.DataFrame({
    'Feature (Serbian)': get_selected_features(lr_best),
    'Coefficient': lr_best.named_steps['model'].coef_[0],
})
coef_df['Feature'] = coef_df['Feature (Serbian)'].apply(translate_feature)
coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
coef_df['|Coefficient|'] = coef_df['Coefficient'].abs()

n_zero = (coef_df['Coefficient'] == 0).sum()
print(f"\nSelected features (k): {len(coef_df)}")
print(f"Coefficients zeroed out by L1: {n_zero} / {len(coef_df)}")

top15_lr = coef_df.sort_values('|Coefficient|', ascending=False).head(15)
print(f"\n{'='*60}")
print("Table 5. Top 15 Logistic Regression Coefficients and Odds Ratios")
print("="*60)
print(top15_lr[['Feature', 'Coefficient', 'Odds Ratio']].to_string(index=False))
top15_lr[['Feature', 'Coefficient', 'Odds Ratio']].to_csv(_out('table5_lr_coefficients.csv'), index=False)
print("  → saved table5_lr_coefficients.csv")

# 7.2 Feature importance (Random Forest & XGBoost)
rf_best_est = best_estimators['Random Forest']
xgb_best_est = best_estimators['XGBoost']

rf_imp = pd.DataFrame({
    'Feature (Serbian)': get_selected_features(rf_best_est),
    'Importance_RF': rf_best_est.named_steps['model'].feature_importances_,
})
xgb_imp = pd.DataFrame({
    'Feature (Serbian)': get_selected_features(xgb_best_est),
    'Importance_XGB': xgb_best_est.named_steps['model'].feature_importances_,
})

combined_imp = rf_imp.merge(xgb_imp, on='Feature (Serbian)')
combined_imp['Feature'] = combined_imp['Feature (Serbian)'].apply(translate_feature)
combined_imp['Avg Rank'] = (
    combined_imp['Importance_RF'].rank(ascending=False) +
    combined_imp['Importance_XGB'].rank(ascending=False)
) / 2
top15_fi = combined_imp.sort_values('Avg Rank').head(15)

# 7.3 SHAP analysis
feat_rf = get_selected_features(rf_best_est)
feat_rf_en = [translate_feature(f) for f in feat_rf]

X_test_trans = rf_best_est[:-1].transform(X_test)
# Ensure dense array for SHAP colour mapping
if hasattr(X_test_trans, 'toarray'):
    X_test_trans = X_test_trans.toarray()
elif hasattr(X_test_trans, 'values'):
    X_test_trans = X_test_trans.values
X_test_trans = np.asarray(X_test_trans, dtype=np.float64)
explainer = shap.TreeExplainer(rf_best_est.named_steps['model'])
shap_values = explainer.shap_values(X_test_trans)

if isinstance(shap_values, list):
    shap_values_stem = shap_values[1]
elif shap_values.ndim == 3:
    shap_values_stem = shap_values[:, :, 1]
else:
    shap_values_stem = shap_values

mean_abs_shap = np.abs(shap_values_stem).mean(axis=0)
shap_imp_df = pd.DataFrame({
    'Feature': feat_rf_en,
    'Mean |SHAP|': mean_abs_shap,
}).sort_values('Mean |SHAP|', ascending=False)
top15_shap = shap_imp_df.head(15)

# Table 4: Descriptive synthesis across three importance measures
rank_lr = coef_df[['Feature (Serbian)', '|Coefficient|']].copy()
rank_lr['Rank_LR'] = rank_lr['|Coefficient|'].rank(ascending=False)

rank_fi = combined_imp[['Feature (Serbian)', 'Avg Rank']].rename(columns={'Avg Rank': 'Rank_FI'})

shap_imp_sr = pd.DataFrame({
    'Feature (Serbian)': feat_rf,
    'Mean |SHAP|': mean_abs_shap,
})
shap_imp_sr['Rank_SHAP'] = shap_imp_sr['Mean |SHAP|'].rank(ascending=False)

synthesis = (
    rank_lr[['Feature (Serbian)', 'Rank_LR']]
    .merge(rank_fi, on='Feature (Serbian)')
    .merge(shap_imp_sr[['Feature (Serbian)', 'Rank_SHAP']], on='Feature (Serbian)')
)
synthesis['Avg Rank'] = synthesis[['Rank_LR', 'Rank_FI', 'Rank_SHAP']].mean(axis=1)
synthesis['Feature'] = synthesis['Feature (Serbian)'].apply(translate_feature)
synthesis = synthesis.sort_values('Avg Rank')

print(f"\n{'='*60}")
print("Table 4. Top 15 Predictors — Synthetic Ranking Across Three Methods")
print("="*60)
table4_df = synthesis[['Feature', 'Rank_LR', 'Rank_FI', 'Rank_SHAP', 'Avg Rank']].head(15)
print(table4_df.to_string(index=False))
table4_df.to_csv(_out('table4_synthetic_ranking.csv'), index=False)
print("  → saved table4_synthetic_ranking.csv")

# ═══════════════════════════════════════════════════════════════════════
# 8. FAIRNESS AND ROBUSTNESS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

# Table 6: Performance by gender
male_mask = X_test['Пол:_мушки'].values.astype(bool)
female_mask = ~male_mask

def baseline_accuracy(y):
    p = y.mean()
    return max(p, 1 - p)

def group_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    return {
        'Accuracy': acc,
        'Above Baseline': acc - baseline_accuracy(y_true),
        'F1 (weighted)': f1_score(y_true, y_pred, average='weighted'),
        'ROC-AUC': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
    }

fairness_results = {}
for name in model_order:
    model = best_estimators[name]
    for gender, mask in [('Male', male_mask), ('Female', female_mask)]:
        y_sub = y_test[mask]
        y_pred = model.predict(X_test[mask])
        y_proba = model.predict_proba(X_test[mask])[:, 1]
        fairness_results[(name, gender)] = group_metrics(y_sub, y_pred, y_proba)

table6 = pd.DataFrame(fairness_results).T.round(3)
table6.index.names = ['Model', 'Gender']
print(f"\n{'='*60}")
print("Table 6. Model Performance by Gender")
print("="*60)
print(table6)
table6.to_csv(_out('table6_fairness_by_gender.csv'))
print("  → saved table6_fairness_by_gender.csv")

for g, m in [('Male', male_mask), ('Female', female_mask)]:
    print(f"\n{g}: n={m.sum()}, STEM proportion={y_test[m].mean():.3f}")

# Bias analysis (gender & urbanization)
location_codes = X_test['Одрастао/ла сам у:']
rural_mask = location_codes.eq(1).values
urban_mask = location_codes.isin([2, 3, 4]).values
print(f"Urbanization groups: village={rural_mask.sum()}, "
      f"non-village={urban_mask.sum()}, "
      f"missing/unknown={(~(rural_mask | urban_mask)).sum()}")

def bias_metrics(model, X_te, y_true, mask):
    y_sub = y_true[mask].values
    y_pred = model.predict(X_te[mask])
    pos_rate = y_pred.mean()
    true_neg = y_sub == 0
    true_pos = y_sub == 1
    fpr = (y_pred[true_neg] == 1).mean() if true_neg.sum() > 0 else np.nan
    fnr = (y_pred[true_pos] == 0).mean() if true_pos.sum() > 0 else np.nan
    return {'Positive Pred. Rate': pos_rate, 'FPR': fpr, 'FNR': fnr, 'n': int(mask.sum())}

rf_model = best_estimators['Random Forest']
groups = {
    'Male': male_mask, 'Female': female_mask,
    'Rural': rural_mask, 'Urban': urban_mask,
}
bias_res = {name: bias_metrics(rf_model, X_test, y_test, mask) for name, mask in groups.items()}
bias_table = pd.DataFrame(bias_res).T.round(3)
print("\nBias Metrics (Random Forest) by Group:")
print(bias_table)
bias_table.to_csv(_out('table_bias_metrics.csv'))
print("  → saved table_bias_metrics.csv")

# Table 7: Robustness analysis
grid_rf_robust = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'selector__k': [10, 30, 50, 70, 90, 'all'],
}

def rf_without_columns(cols_to_drop):
    Xtr = X_train.drop(columns=cols_to_drop, errors='ignore')
    Xte = X_test.drop(columns=cols_to_drop, errors='ignore')
    num = [c for c in numeric_cols_scale if c in Xtr.columns]
    bincols = [c for c in binary_cols if c in Xtr.columns]
    gs = GridSearchCV(
        make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE), num_cols=num, bin_cols=bincols),
        grid_rf_robust, scoring='f1_weighted', cv=skf, n_jobs=1,
    )
    gs.fit(Xtr, y_train)
    est = gs.best_estimator_
    yp = est.predict(Xte)
    ypr = est.predict_proba(Xte)[:, 1]
    return {
        'Accuracy': accuracy_score(y_test, yp),
        'F1 (weighted)': f1_score(y_test, yp, average='weighted'),
        'ROC-AUC': roc_auc_score(y_test, ypr),
    }

rf_full = {
    'Accuracy': test_results['Random Forest']['Accuracy'],
    'F1 (weighted)': test_results['Random Forest']['F1 (weighted)'],
    'ROC-AUC': test_results['Random Forest']['ROC-AUC'],
}

top_predictor = 'Занимљив [Физика]'
rf_no_top = rf_without_columns([top_predictor])

enrollment_factors = [c for c in X_train.columns
                      if c.startswith('Процени који од фактора је имао колико утицаја')]
print(f"\nEnrollment decision factors block: {len(enrollment_factors)} columns")
rf_no_block = rf_without_columns(enrollment_factors)

table7 = pd.DataFrame({
    'All features': rf_full,
    'Without top predictor (Physics interest.)': rf_no_top,
    f'Without enrollment factors ({len(enrollment_factors)} cols)': rf_no_block,
}).T.round(3)
print(f"\n{'='*60}")
print("Table 7. Robustness Analysis — Random Forest")
print("="*60)
print(table7)
table7.to_csv(_out('table7_robustness.csv'))
print("  → saved table7_robustness.csv")

# ═══════════════════════════════════════════════════════════════════════
# 9. SAVE ALL RESULTS FOR PLOTTING  (stem_plots.py reads this file)
# ═══════════════════════════════════════════════════════════════════════

# Compute ROC curve data for each model (needed for plotting)
roc_data = {}
for name in model_order:
    fpr, tpr, _ = roc_curve(y_test, predictions[name]['y_proba'])
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': test_results[name]['ROC-AUC']}
fpr_b, tpr_b, _ = roc_curve(y_test, predictions['Baseline (Dummy)']['y_proba'])
roc_data['Baseline (Dummy)'] = {'fpr': fpr_b, 'tpr': tpr_b, 'auc': 0.5}

# Confusion matrix for best model
cm = confusion_matrix(y_test, predictions[best_model_name]['y_pred'])

results = {
    # Figure 1 data
    'target_counts': target_counts,
    'n_total': n_total_before_cleaning,
    # Figure 4 data (ROC curves)
    'roc_data': roc_data,
    'model_order': model_order,
    # Figure 5 data (confusion matrix)
    'confusion_matrix': cm,
    'best_model_name': best_model_name,
    # Figure 6 data (learning curves)
    'learning_curve_data': learning_curve_data,
    # Figure 7 data (SHAP beeswarm)
    'shap_values_stem': shap_values_stem,
    'X_test_trans': X_test_trans,
    'feat_rf_en': feat_rf_en,
    # Figure 8 data (SHAP bar)
    'top15_shap': top15_shap,
    # Figure 9 data (feature importance)
    'top15_fi': top15_fi,
}

with open(_out('results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print("\n✓ results.pkl saved.")

# ═══════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nDataset: {len(X)} observations, {X.shape[1]} features")
print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
print(f"\nBest model by test F1: {best_model_name}")
print(f"  Accuracy: {test_results[best_model_name]['Accuracy']:.3f}")
print(f"  ROC-AUC: {test_results[best_model_name]['ROC-AUC']:.3f}")
print(f"\nCSV tables saved:")
print("  table1_hyperparameters.csv   — Best hyperparameters per model")
print("  table2_model_performance.csv — Test set performance metrics")
print("  table3_statistical_tests.csv — Approximate corrected paired t-tests (Holm)")
print("  table3_statistical_method_note.txt — Statistical method and caveats")
print("  table4_synthetic_ranking.csv — Descriptive feature ranking across methods")
print("  table5_lr_coefficients.csv   — Top LR coefficients & odds ratios")
print("  table6_fairness_by_gender.csv— Performance by gender")
print("  table_bias_metrics.csv       — Group positive rate, FPR, FNR and sample size")
print("  table7_robustness.csv        — Robustness analysis (Random Forest)")
print("\nPlot data saved to results.pkl — run stem_plots.py to generate figures.")
print("(Figures 2 & 3 are schematic diagrams — created separately.)")
