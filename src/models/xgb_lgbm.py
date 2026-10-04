from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Correct column names from your raw dataset
F = [
    'branch',
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]
T = 'placement_status'

def load():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    df = df[F + [T]].dropna()

    # Convert 'Tier 1', 'Tier 2', etc. into numeric numbers
    if not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = (
            df['college_tier']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(df[F], columns=['branch'], dtype=float)

    # Encode binary target if needed
    y = df[T]
    if not pd.api.types.is_numeric_dtype(y):
        lab = sorted(y.astype(str).unique())
        if len(lab) != 2:
            raise ValueError('placement_status must be binary')
        y = y.astype(str).map({lab[0]: 0, lab[1]: 1})

    return X, y

def shap_report(model, X, name):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # Handle multi-class / binary output differences across SHAP versions
    if isinstance(sv, list):
        vals = sv[1] if len(sv) > 1 else sv[0]
    elif len(getattr(sv, 'shape', ())) == 3:
        vals = sv[:, :, 1]
    else:
        vals = sv

    # Feature Importance Table
    imp = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Absolute_SHAP': np.abs(vals).mean(axis=0)
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    imp.to_csv(OUT / f'{name}_shap_feature_importance.csv', index=False)

    # Summary Plot
    plt.figure()
    shap.summary_plot(vals, X, show=False)
    plt.tight_layout()
    plt.savefig(OUT / f'{name}_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Dependence Plot for top feature
    top = imp.iloc[0].Feature
    plt.figure()
    shap.dependence_plot(top, vals, X, show=False)
    plt.tight_layout()
    plt.savefig(OUT / f'{name}_shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nTop features ({name}):\n', imp.head(10).to_string(index=False))

def run():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'xgboost': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
    }

    rows = []
    for name, m in models.items():
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        sec = round(time.perf_counter() - t, 4)
        pred = m.predict(Xte)
        acc = round(accuracy_score(yte, pred), 4)

        print(f"\n{'='*40}")
        print(f"{name.upper()} | Accuracy: {acc} | Training Time: {sec}s")
        print(f"{'='*40}")
        print(classification_report(yte, pred))

        shap_report(m, Xte, name)
        rows.append([name, acc, sec])

    pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Training_Time_Seconds']).to_csv(
        OUT / 'xgb_lightgbm_comparison.csv', index=False
    )

if __name__ == '__main__':
    run()