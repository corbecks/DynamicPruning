import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

filenames_tensor_list = []
activations_tensor_list = []
cluster_list = []

for i in range(13):
    temp_filenames_tensor = torch.load(f"activation_vectors_1281167_training/final_labels_torch_vector_{i}.pt")
    filenames_tensor_list += temp_filenames_tensor

    temp_activations_tensor = torch.load(f"activation_vectors_1281167_training/final_activations_torch_vector_{i}.pt")
    activations_tensor_list.append(temp_activations_tensor)

activations_tensor = torch.cat(activations_tensor_list, dim=0)

cluster_list = torch.load(f"activation_vectors_1281167_training/Analysis/final_clusters.pt", weights_only=False)

means_by_cluster = []
stds_by_cluster = []
for i in range(10):
    temp_cluster_activations = []
    print("Finding activations for cluster: ", i)

    for j in range(len(cluster_list)):
        cluster = cluster_list[j]
        if(cluster == i):
            temp_cluster_activations.append(activations_tensor[j].tolist())

    means_by_cluster.append(np.mean(temp_cluster_activations, axis=0))
    stds_by_cluster.append(np.std(temp_cluster_activations, axis=0))

torch.save(means_by_cluster,f"feature_means_by_clusters.pt")
torch.save(stds_by_cluster,f"feature_stds_by_clusters.pt")


#means_by_cluster = torch.load(f"feature_means_by_clusters.pt", weights_only=False)
#stds_by_cluster = torch.load(f"feature_stds_by_clusters.pt", weights_only=False)


fig, ax = plt.subplots()

# We'll keep track of one scatter object to attach the colorbar.
scatter_obj = None

for i in range(10):
    # x-values: means for set i
    # y-values: stds for set i
    x = means_by_cluster[i]
    y = stds_by_cluster[i]

    # Color all points in this set by the same index i.
    # We also set vmin=0, vmax=9 so the color scale spans 10 distinct indices.
    scatter_obj = ax.scatter(
        x,
        y,
        c=np.full(len(x), i, dtype=float),
        cmap='viridis',
        vmin=0,
        vmax=9,
        label=f"Set {i}",
        s=1,
        marker='o'
    )

# Optionally add a colorbar to show which color corresponds to which set index
cbar = plt.colorbar(scatter_obj, ax=ax)
cbar.set_label("Set Index")

ax.set_xlabel("Mean")
ax.set_ylabel("Standard Deviation")
ax.set_title("Mean vs Standard Deviation by Set")
ax.legend()
plt.show()





fig = plt.figure(figsize=(15, 15))
# Create a grid: 4 rows, 3 columns.
# The bottom row will have a shorter height ratio so the extra subplot is smaller.
gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 0.7])

# Plot the first 9 subplots in a 3x3 grid
for i in range(10):
    row = i // 3  # integer division gives row index (0, 1, or 2)
    col = i % 3  # remainder gives column index (0, 1, or 2)
    ax = fig.add_subplot(gs[row, col])

    x = means_by_cluster[i]
    y = stds_by_cluster[i]

    scatter_obj = ax.scatter(
        x,
        y,
        c=np.full(len(x), i, dtype=float),
        cmap='viridis',
        vmin=0,
        vmax=9,
        s=1,
        marker='x',
        alpha=0.5
    )

    ax.set_title(f"Set {i}")
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std")


fig.suptitle("Mean vs Standard Deviation by Set", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()