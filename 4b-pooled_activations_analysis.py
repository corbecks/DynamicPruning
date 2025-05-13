import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cluster_list = torch.load(f"activation_vectors_1281167_training/Analysis/final_clusters.pt", weights_only=False)

cluster_counts = []
total_count = 0

for i in range(10):
    count = (cluster_list == i).sum()
    print(count)
    cluster_counts.append(count)
    total_count += count

means_by_cluster = torch.load(f"activation_vectors_1281167_training/Analysis/feature_means_by_clusters.pt", weights_only=False)
stds_by_cluster = torch.load(f"activation_vectors_1281167_training/Analysis/feature_stds_by_clusters.pt", weights_only=False)

running_means = []
running_stds = []
for i in range(10):
    cluster_means = means_by_cluster[i]
    cluster_stds = stds_by_cluster[i]
    cluster_num = cluster_counts[i]

    if (i == 0):
        running_means = cluster_means * cluster_num
        running_stds = cluster_stds ** 2 * (cluster_num - 1)
    else:
        running_means += cluster_means * cluster_num
        running_stds += cluster_stds ** 2 * (cluster_num - 1)



total_means = running_means * (1 / total_count )
total_stds = np.sqrt(running_stds * (1 / (total_count - 10)))

torch.save(total_means,f"feature_means_all_activations_pooled.pt")
torch.save(total_stds,f"feature_stds_all_activations_pooled.pt")

x = total_means
y = total_stds

# Color all points in this set by the same index i.
# We also set vmin=0, vmax=9 so the color scale spans 10 distinct indices.
plt.scatter(x, y, cmap='viridis', s=1, marker='o')

# Optionally add a colorbar to show which color corresponds to which set index

plt.xlabel("Mean")
plt.ylabel("Standard Deviation")
plt.title("Mean vs Standard Deviation For All Features")
plt.show()
