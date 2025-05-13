import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



def standardize_float32(data):
    # Ensure data is float32
    data = data.astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    return ((data - mean) / std).astype(np.float32)



def cluster_activations(data_tensor, num_clusters=10, visualization="PCA", variance_cutoff=0.95, plot_variance=True):
    # Assume 'data_tensor' is  of shape (n_samples, n_features)
    data_scaled = standardize_float32(data_tensor.numpy())

    # Step 1: Perform PCA
    pca = PCA()
    pca.fit_transform(data_scaled)

    if plot_variance:
        # Step 2: Plot the cumulative explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.show()

    # Step 3: Choose the number of components (e.g., those that explain 95% of the variance)
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_cutoff) + 1
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_scaled)
    torch.save(data_reduced, f"pca_data_reduced.pt")

    low_n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_cutoff*0.8) + 1
    low_pca = PCA(n_components=low_n_components)
    low_data_reduced = low_pca.fit_transform(data_scaled)
    torch.save(low_data_reduced, f"pca_low_data_reduced.pt")

    # Step 4: Cluster using k-means (for example, assuming k=3 clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_reduced)



    if visualization == "PCA":
        # Visualize the clusters in the space of the first two principal components
        plt.figure(figsize=(8, 5))
        plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=clusters, cmap='viridis', s=0.5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Clusters in Reduced PCA Space')
        plt.show()
    else:
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

    return clusters


filenames_tensor_list = []
activations_tensor_list = []
for i in range(13):
    temp_filenames_tensor = torch.load(f"activation_vectors_1281167_training/final_labels_torch_vector_{i}.pt")
    filenames_tensor_list += temp_filenames_tensor

    temp_activations_tensor = torch.load(f"activation_vectors_1281167_training/final_activations_torch_vector_{i}.pt")
    activations_tensor_list.append(temp_activations_tensor)

activations_tensor = torch.cat(activations_tensor_list, dim=0)
activations_tensor = activations_tensor.to(dtype=torch.float32)



clusters = cluster_activations(activations_tensor,
                               num_clusters=10,
                               visualization="UMAP",
                               variance_cutoff=0.95,
                               plot_variance=True)

torch.save(clusters,f"final_clusters.pt")











