from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

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

    # Convert 'Tier 1', 'Tier 2', etc. into numeric values (1, 2, ...)
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
        y = y.astype(str).map({lab[0]: 0, lab[1]: 1})

    return X, y

def score(m, X, y):
    p = m.predict(X)
    return [
        round(accuracy_score(y, p), 4),
        round(precision_score(y, p, zero_division=0), 4),
        round(recall_score(y, p, zero_division=0), 4),
        round(f1_score(y, p, zero_division=0), 4)
    ]

def run():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1. Baseline Decision Tree vs Random Forest
    dt = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
    print('Decision Tree [accuracy, precision, recall, F1]:', score(dt, Xte, yte))

    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print('Random Forest [accuracy, precision, recall, F1]:', score(rf, Xte, yte))
    print('OOB score:', round(rf.oob_score_, 4), '| OOB error:', round(1 - rf.oob_score_, 4))

    # 2. Effect of Number of Trees
    counts = [10, 25, 50, 100, 200]
    oob = []
    acc = []
    for n in counts:
        m = RandomForestClassifier(n_estimators=n, max_features='sqrt', oob_score=True, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        oob.append(1 - m.oob_score_)
        acc.append(accuracy_score(yte, m.predict(Xte)))

    plt.figure(figsize=(9, 6))
    plt.plot(counts, oob, marker='o', label='OOB Error')
    plt.plot(counts, acc, marker='o', label='Test Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Value')
    plt.title('Effect of Number of Trees - Placement Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'random_forest_number_of_trees.png', dpi=150)
    plt.close()

    # 3. Effect of Feature Subsampling (max_features)
    opts = ['sqrt', 'log2', None]
    labs = ['sqrt', 'log2', 'all features']
    vals = []
    for o in opts:
        m = RandomForestClassifier(n_estimators=100, max_features=o, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        vals.append(accuracy_score(yte, m.predict(Xte)))

    print('Feature subsampling:', {k: round(v, 4) for k, v in zip(labs, vals)})

    plt.figure(figsize=(8, 6))
    plt.bar(labs, vals, color=['#2b5c8f', '#4682b4', '#70a1d7'])
    plt.ylabel('Test Accuracy')
    plt.title('Feature Subsampling - Placement Prediction')
    plt.ylim(0, 1)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / 'random_forest_feature_subsampling.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    run()