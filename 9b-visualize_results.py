import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



conventional_accuracy = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_conventionally_pruned_accuracy_vs_cuttoffs_N_5000.pt", weights_only=False)
conventional_filters_removed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_conventionally_pruned_filters_removed_vs_cuttoffs_N_5000.pt", weights_only=False)
conventional_products_zeroed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_conventionally_pruned_zero_products_vs_cuttoffs_N_5000.pt", weights_only=False)

femtonet_accuracy = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_femtonet_accuracy_vs_cuttoffs_N_5000.pt", weights_only=False)
femtonet_filters_removed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_femtonet_filters_removed_vs_cuttoffs_N_5000.pt", weights_only=False)
femtonet_products_zeroed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_femtonet_zero_products_vs_cuttoffs_N_5000.pt", weights_only=False)

nanonet_accuracy = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_nanonet_accuracy_vs_cuttoffs_N_5000.pt", weights_only=False)
nanonet_filters_removed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_nanonet_filters_removed_vs_cuttoffs_N_5000.pt", weights_only=False)
nanonet_products_zeroed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_nanonet_zero_products_vs_cuttoffs_N_5000.pt", weights_only=False)

random_accuracy = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_random_accuracy_vs_cuttoffs_N_5000.pt", weights_only=False)
random_filters_removed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_random_filters_removed_vs_cuttoffs_N_5000.pt", weights_only=False)
random_products_zeroed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_random_zero_products_vs_cuttoffs_N_5000.pt", weights_only=False)

ideal_accuracy = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_ideal_accuracy_vs_cuttoffs_N_5000.pt", weights_only=False)
ideal_filters_removed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_ideal_filters_removed_vs_cuttoffs_N_5000.pt", weights_only=False)
ideal_products_zeroed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_ideal_zero_products_vs_cuttoffs_N_5000.pt", weights_only=False)

combined_accuracy = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_combined_femto_conventional_accuracy_vs_cuttoffs_N_2400.pt", weights_only=False)
combined_filters_removed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_combined_femto_conventional_filters_removed_vs_cuttoffs_N_2400.pt", weights_only=False)
combined_products_zeroed = torch.load("activation_vectors_1281167_training/analysis/performance_characteristics_run2/run2_combined_femto_conventional_zero_products_vs_cuttoffs_N_2400.pt", weights_only=False)

conventional_pairs_products = sorted(zip(conventional_accuracy.flatten(), conventional_products_zeroed.flatten()), key=lambda pair: pair[0])
femtonet_pairs_products = sorted(zip(femtonet_accuracy.flatten(), femtonet_products_zeroed.flatten()), key=lambda pair: pair[0])
nanonet_pairs_products = sorted(zip(nanonet_accuracy.flatten(), nanonet_products_zeroed.flatten()), key=lambda pair: pair[0])
random_pairs_products = sorted(zip(random_accuracy.flatten(), random_products_zeroed.flatten()), key=lambda pair: pair[0])
ideal_pairs_products = sorted(zip(ideal_accuracy.flatten(), ideal_products_zeroed.flatten()), key=lambda pair: pair[0])
combined_pairs_products = sorted(zip(combined_accuracy.flatten(), combined_products_zeroed.flatten()), key=lambda pair: pair[0])

conv_accur, conv_prods = zip(*conventional_pairs_products)
femto_accur, femto_prods = zip(*femtonet_pairs_products)
nano_accur, nano_prods = zip(*nanonet_pairs_products)
rand_accur, rand_prods = zip(*random_pairs_products)
ideal_accur, ideal_prods = zip(*ideal_pairs_products)
comb_accur, comb_prods = zip(*combined_pairs_products)


plt.plot(conv_accur,  conv_prods, label="conventional")
plt.plot(femto_accur, np.array(femto_prods) - 3379324, label="femtonet")
plt.plot(nano_accur,  np.array(nano_prods) - 20255572, label="nanonet")
plt.plot(rand_accur,  np.array(rand_prods), label="random")
plt.plot(ideal_accur,  np.array(ideal_prods), label="ideal")
plt.plot(comb_accur,  np.array(comb_prods) - 3379324, label="combined")



plt.xlabel('Accuracy %')
plt.ylabel('Zero-Multiplications (#)')
plt.title("Zero-Products vs Accuracy with Different Pruning Techniques\n (Note: corrected for additional multiplications from cluster classifiers)")

plt.legend()

plt.show()

# Create an interpolation function
conv_interp = interp1d(conv_accur, conv_prods, kind='linear')  # You can also use 'quadratic', 'cubic', etc.
max_accuracy = np.max(conv_accur)



plt.plot(femto_accur, np.array(femto_prods) - 3379324 - conv_interp(np.clip(np.array(femto_accur), None, max_accuracy)), label="femtonet")
plt.plot(nano_accur,  np.array(nano_prods) - 20255572- conv_interp(np.clip(np.array(nano_accur), None, max_accuracy)), label="nanonet")
plt.plot(rand_accur,  np.array(rand_prods)- conv_interp(np.clip(np.array(rand_accur), None, max_accuracy)), label="random")
plt.plot(ideal_accur,  np.array(ideal_prods)- conv_interp(np.clip(np.array(ideal_accur), None, max_accuracy)), label="ideal")
plt.plot(comb_accur,  np.array(comb_prods) - 3379324 - conv_interp(np.clip(np.array(ideal_accur), None, max_accuracy)), label="combined")


plt.xlabel('Accuracy %')
plt.ylabel('Zero-Multiplications Relative to Conventional(#)')
plt.title("Zero-Products vs Accuracy with Different Pruning Techniques\n (Note: additional zero multiplications beyond conventional pruning)")

plt.legend()

plt.show()





conventional_pairs_filters = sorted(zip(conventional_accuracy.flatten(), conventional_filters_removed.flatten()), key=lambda pair: pair[0])
femtonet_pairs_filters = sorted(zip(femtonet_accuracy.flatten(), femtonet_filters_removed.flatten()), key=lambda pair: pair[0])
nanonet_pairs_filters = sorted(zip(nanonet_accuracy.flatten(), nanonet_filters_removed.flatten()), key=lambda pair: pair[0])
random_pairs_filters = sorted(zip(random_accuracy.flatten(), random_filters_removed.flatten()), key=lambda pair: pair[0])
ideal_pairs_filters = sorted(zip(ideal_accuracy.flatten(), ideal_filters_removed.flatten()), key=lambda pair: pair[0])
combined_pairs_filters = sorted(zip(combined_accuracy.flatten(), combined_filters_removed.flatten()), key=lambda pair: pair[0])

conv_accur, conv_filts = zip(*conventional_pairs_filters)
femto_accur, femto_filts = zip(*femtonet_pairs_filters)
nano_accur, nano_filts = zip(*nanonet_pairs_filters)
rand_accur, rand_filts = zip(*random_pairs_filters)
ideal_accur, ideal_filts = zip(*ideal_pairs_filters)
comb_accur, comb_filts = zip(*combined_pairs_filters)


plt.plot(conv_accur,  conv_filts, label="conventional")
plt.plot(femto_accur, femto_filts, label="femtonet")
plt.plot(nano_accur,  nano_filts, label="nanonet")
plt.plot(rand_accur,  rand_filts, label="random")
plt.plot(ideal_accur,  ideal_filts, label="ideal")
plt.plot(comb_accur,  comb_filts, label="combined")


plt.xlabel('Accuracy %')
plt.ylabel('Filters Removed (#)')
plt.title("Filters Pruned vs Accuracy with Different Pruning Techniques")

plt.legend()

plt.show()



means_by_cluster = torch.load(f"activation_vectors_1281167_training/analysis/feature_means_by_clusters.pt", weights_only=False)
stds_by_cluster = torch.load(f"activation_vectors_1281167_training/analysis/feature_stds_by_clusters.pt", weights_only=False)

means_all = torch.load(f"activation_vectors_1281167_training/analysis/feature_means_all_activations_pooled.pt", weights_only=False)
stds_all = torch.load(f"activation_vectors_1281167_training/analysis/feature_stds_all_activations_pooled.pt", weights_only=False)


# Convert the data to numpy arrays for easier handling
means_array = [np.array(cluster) for cluster in means_by_cluster]
stds_array = [np.array(cluster) for cluster in stds_by_cluster]

means_array.append(means_all)
stds_array.append(stds_all)

# Customize the y-axis labels
y_ticks = np.arange(len(stds_array))  # Get the current y-axis tick positions
y_tick_labels = [str(i) for i in y_ticks]  # Convert to string labels (default is just the number)
y_tick_labels[10] = "all clusters"  # Change the label for the 10th tick (index 9) to "all"


# Let's assume each cluster has the same number of features (columns)
num_features = len(means_array[0])

# Create the heatmap for means
plt.figure(figsize=(12, 8))
sns.heatmap(means_array, cmap="YlGnBu", cbar_kws={'label': 'Mean Value'}, yticklabels=np.arange(len(means_array)))

# Set the new y-axis labels
plt.yticks(ticks=y_ticks, labels=y_tick_labels)

plt.title("Heatmap of Mean Activations by Cluster")
plt.xlabel("Feature Index")
plt.ylabel("Cluster")
plt.show()

# Optionally, you can do a similar plot for standard deviations:
plt.figure(figsize=(12, 8))
sns.heatmap(stds_array, cmap="YlGnBu", cbar_kws={'label': 'Standard Deviation'}, yticklabels=np.arange(len(stds_array)))

# Set the new y-axis labels
plt.yticks(ticks=y_ticks, labels=y_tick_labels)

plt.title("Heatmap of Standard Deviations of Activations by Cluster")
plt.xlabel("Feature Index")
plt.ylabel("Cluster")
plt.show()


mean_squared = [means_cluster*means_cluster for means_cluster in means_array]
stds_squared = [stds_cluster*stds_cluster for stds_cluster in stds_array]
norm_squared = [np.sqrt(mean_sqr + std_sqr) for mean_sqr, std_sqr in zip(mean_squared, stds_squared)]

# Optionally, you can do a similar plot for standard deviations:
plt.figure(figsize=(12, 8))
sns.heatmap(norm_squared, cmap="YlOrRd", cbar_kws={'label': 'Activation'}, yticklabels=np.arange(len(stds_array)))

# Set the new y-axis labels
plt.yticks(ticks=y_ticks, labels=y_tick_labels)

plt.title("Heatmap of Activation Norm(STD, Mean) by Cluster")
plt.xlabel("Feature Index")
plt.ylabel("Cluster")
plt.show()


norm_squared_clipped = [np.clip(norm, 0, 1) for norm in norm_squared]

# Optionally, you can do a similar plot for standard deviations:
plt.figure(figsize=(12, 8))
sns.heatmap(norm_squared_clipped, cmap="YlOrRd", cbar_kws={'label': 'Activation'}, yticklabels=np.arange(len(stds_array)))

# Set the new y-axis labels
plt.yticks(ticks=y_ticks, labels=y_tick_labels)

plt.title("Heatmap of Activation Norm(STD, Mean) by Cluster (Clipped at 1)")
plt.xlabel("Feature Index")
plt.ylabel("Cluster")
plt.show()