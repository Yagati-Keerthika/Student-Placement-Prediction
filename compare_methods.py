"""
Simple comparison of PCA, t-SNE, and UMAP
for Student Placement Prediction dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

# Load data
df = pd.read_csv('student_placement_prediction_dataset_2026.csv')

# Select features
features = ['cgpa', 'aptitude_score', 'coding_skill_score']
X = df[features].copy().fillna(df[features].median()).fillna(0)

# Standardize
X_scaled = StandardScaler().fit_transform(X)

print("=" * 60)
print("DIMENSIONALITY REDUCTION COMPARISON")
print("=" * 60)

# PCA
print("\n1. PCA (Principal Component Analysis)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"   Variance Retained: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

# t-SNE
print("\n2. t-SNE (t-Distributed Stochastic Neighbor Embedding)")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
print("   Status: Completed (non-linear transformation)")

# UMAP
print("\n3. UMAP (Uniform Manifold Approximation and Projection)")
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
print("   Status: Completed (non-linear transformation)")

# Clustering comparison
print("\n" + "=" * 60)
print("CLUSTERING QUALITY METRICS (3 clusters)")
print("=" * 60)

comparison_data = []

for method_name, X_reduced in [('PCA', X_pca), ('t-SNE', X_tsne), ('UMAP', X_umap)]:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_reduced, labels)

    comparison_data.append({
        'Method': method_name,
        'Inertia': f"{inertia:.4f}",
        'Silhouette Score': f"{silhouette:.4f}",
        'Cluster Sizes': dict(zip(*np.unique(labels, return_counts=True)))
    })

    print(f"\n{method_name}:")
    print(f"  Inertia: {inertia:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Cluster Sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

kmeans_all = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_all = kmeans_all.fit_predict(X_pca)

# PCA
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_all, cmap='viridis',
                           edgecolors='black', alpha=0.7)
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].set_title('PCA with K-Means Clusters')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# t-SNE
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_all, cmap='viridis',
                           edgecolors='black', alpha=0.7)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('t-SNE with K-Means Clusters')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

# UMAP
scatter3 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=labels_all, cmap='viridis',
                           edgecolors='black', alpha=0.7)
axes[2].set_xlabel('UMAP Dimension 1')
axes[2].set_ylabel('UMAP Dimension 2')
axes[2].set_title('UMAP with K-Means Clusters')
axes[2].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[2], label='Cluster')

plt.tight_layout()
plt.savefig('dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
print("\n\nComparison plot saved as: dimensionality_reduction_comparison.png")
# plt.show()  # Commented out for headless execution

# Key insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
PCA:
  ✓ Fastest computation
  ✓ Linear transformation preserving global variance
  ✓ Good for interpretability
  ✗ May miss non-linear patterns

t-SNE:
  ✓ Excellent cluster visualization
  ✓ Preserves local neighborhood structure
  ✗ Slower computation
  ✗ Global structure may be distorted

UMAP:
  ✓ Faster than t-SNE
  ✓ Preserves both local and global structure
  ✓ Better for large datasets
  ✓ Good manifold learning
""")
