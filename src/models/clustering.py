from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Exact matching columns from raw_placement_data.csv
F = [
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]

def load():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()

    # Extract digits from 'Tier 1', 'Tier 2', etc. before numeric conversion
    if 'college_tier' in df.columns and not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = df['college_tier'].astype(str).str.extract(r'(\d+)')

    df_subset = df[F].apply(pd.to_numeric, errors='coerce').dropna()
    return StandardScaler().fit_transform(df_subset)

def scores(X, lab):
    mask = lab != -1
    xx = X[mask]
    yy = lab[mask]
    n_clusters = len(set(yy))
    
    if n_clusters < 2:
        return np.nan, np.nan, n_clusters
        
    sample_sz = min(len(xx), 2000)
    return (
        round(silhouette_score(xx, yy, sample_size=sample_sz, random_state=42), 4),
        round(davies_bouldin_score(xx, yy), 4),
        n_clusters
    )

def run():
    print("Loading data...")
    X = load()
    n_samples = len(X)
    print(f"Dataset loaded with {n_samples} rows. Running K-Means evaluation...")

    sample_sz = min(n_samples, 2000)
    ks = range(2, 11)
    inert = []
    sil = []

    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42)
        l = m.fit_predict(X)
        inert.append(m.inertia_)
        # Use sample_size to make silhouette score fast
        sil.append(silhouette_score(X, l, sample_size=sample_sz, random_state=42))
        print(f"  Finished K={k}")

    best = list(ks)[int(np.argmax(sil))]
    print('\nOptimal K selected by highest silhouette:', best)

    # 1. Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inert, marker='o', color='#2b5c8f')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve - Placement Dataset')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'clustering_kmeans_elbow.png', dpi=150)
    plt.close()

    # 2. Silhouette analysis curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), sil, marker='o', color='#d95f02')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis - Placement Dataset')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'clustering_kmeans_silhouette.png', dpi=150)
    plt.close()

    print("\nRunning model comparisons...")
    # K-Means clustering with optimal K
    km = KMeans(n_clusters=best, n_init=10, random_state=42).fit_predict(X)
    ks1, db1, c1 = scores(X, km)

    # Agglomerative clustering with optimal K
    ag = AgglomerativeClustering(n_clusters=best, linkage='ward').fit_predict(X)
    ks2, db2, c2 = scores(X, ag)

    # Hierarchical Dendrogram (subsampled if dataset is large to prevent OOM/freeze)
    print("Generating dendrogram...")
    dendro_sample = X[np.random.RandomState(42).choice(n_samples, min(n_samples, 1000), replace=False)]
    Z = linkage(dendro_sample, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Dendrogram - Placement Dataset')
    plt.xlabel('Cluster / Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(OUT / 'hierarchical_dendrogram.png', dpi=150)
    plt.close()

    # DBSCAN clustering
    print("Running DBSCAN...")
    db = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    ks3, db3, c3 = scores(X, db)
    noise = int(np.sum(db == -1))

    # Summary table
    res = pd.DataFrame([
        ['K-Means', c1, ks1, db1, 0],
        ['Agglomerative', c2, ks2, db2, 0],
        ['DBSCAN', c3, ks3, db3, noise]
    ], columns=['Algorithm', 'Clusters', 'Silhouette', 'Davies_Bouldin', 'Noise_Points'])

    print('\nClustering Comparison:')
    print(res.to_string(index=False))
    res.to_csv(OUT / 'clustering_comparison.csv', index=False)
    print(f"\nAll outputs and plots saved to: {OUT}")

if __name__ == '__main__':
    run()