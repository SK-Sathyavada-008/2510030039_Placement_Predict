from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / 'src' / 'data' / 'input_image.jpg'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

def run():
    # Load and preprocess image
    img = Image.open(IMAGE).convert('RGB').resize((300, 300))
    arr = np.array(img)
    X = arr.reshape(-1, 3).astype(float)
    
    ks = [2, 4, 6, 8]
    inertias = []
    segs = []

    # Run K-Means for each K
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42)
        lab = m.fit_predict(X)
        
        # Reconstruct segmented image from cluster centroids
        seg = m.cluster_centers_[lab].reshape(arr.shape).astype('uint8')
        inertias.append(m.inertia_)
        segs.append(seg)
        
        # Save individual segmented output
        Image.fromarray(seg).save(OUT / f'segmented_k{k}.png')

    # Side-by-side segmentation comparison plot
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(arr)
    ax[0].set_title('Original')
    ax[0].axis('off')

    for a, k, s in zip(ax[1:], ks, segs):
        a.imshow(s)
        a.set_title(f'K={k}')
        a.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / 'image_segmentation_comparison.png', dpi=150)
    plt.close()

    # Inertia / Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o', color='#2b5c8f', linewidth=2)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve - Image Segmentation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT / 'image_segmentation_elbow.png', dpi=150)
    plt.close()

    print("Segmentation completed. Figures saved to:", OUT)

if __name__ == '__main__':
    run()