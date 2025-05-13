import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import numpy as np


classifier_names = ["femtonet", "piconet", "nanonet", "micronet", "tinynet"]

cmap_red = plt.get_cmap("Reds")
cmap_blue = plt.get_cmap("Blues")

colors_red = cmap_red(np.linspace(0.25, 1, len(classifier_names)))
colors_blue = cmap_blue(np.linspace(0.25, 1, len(classifier_names)))

parameter_counts = []
final_accuracies = []

for i, classifier_name in enumerate(classifier_names):
    epoch_losses = torch.load(f"cluster_classifiers/{classifier_name}/{classifier_name}_epoch_losses.pt", weights_only=False)
    epoch_accuracies = torch.load(f"cluster_classifiers/{classifier_name}/{classifier_name}_epoch_accuracies.pt", weights_only=False)
    epoch_accuracies = [accuracy/100 for accuracy in epoch_accuracies]
    plt.plot(epoch_losses[0:6], label=f"{classifier_name}_loss", color=colors_red[i])
    plt.plot(epoch_accuracies[0:6], label=f"{classifier_name}_accuracy", color=colors_blue[i])

    parameters = torch.load(f"cluster_classifiers/{classifier_name}/{classifier_name}_final_weights.pt", weights_only=False)
    total_params = sum(t.numel() for t in parameters.values())
    parameter_counts.append(total_params)
    final_accuracies.append(epoch_accuracies[5])


plt.xlabel("Epochs")
plt.ylabel("Loss (red) / Accuracy (blue)")
plt.title("Small Cluster Classifier Performance After Training")
plt.legend()
plt.show()

plt.scatter(parameter_counts, final_accuracies)
plt.xscale('log')
plt.xlabel("Number of Parameters")
plt.ylabel("Accuracy")
plt.title("Small Cluster Classifier Performance Versus Parameter Count")
plt.show()