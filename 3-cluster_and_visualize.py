import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

num_clusters = 10
visualization = "UMAP"

data_reduced = torch.load(f"pca_data_reduced.pt", weights_only=False)
low_data_reduced = torch.load(f"pca_low_data_reduced.pt", weights_only=False)

# Cluster using k-means (for example, assuming k=3 clusters)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(data_reduced)
torch.save(clusters, "final_clusters.pt")

#if visualization == "PCA":
# Visualize the clusters in the space of the first two principal components
plt.figure(figsize=(8, 5))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=clusters, cmap='viridis', s=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters in Reduced PCA Space')
plt.show()
#else:
# Use UMAP for final 2D visualization
umap_reducer = umap.UMAP(random_state=42)
embedding = umap_reducer.fit_transform(low_data_reduced)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=0.5)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('Clusters Visualized with UMAP')
plt.show()

