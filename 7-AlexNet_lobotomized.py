import torch
from torchvision import transforms
import torchvision.datasets as datasets

import copy
import numpy as np

from threadpoolctl import threadpool_limits

import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# GPU SETTINGS
CUDA_DEVICE = 0  # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1  # set to 1 if want to run on gpu.
BATCH_SIZE = 1 #000

standard_deviation_cuttoffs = np.linspace(0,0.5,5)
mean_activation_cuttoffs = np.linspace(0,0.5,5)


# HELPER FUNCTIONS
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import torch
import torch.nn as nn

import torch
import torch.nn as nn

filenames_tensor_list = []
cluster_list = []

for i in range(13):
    temp_filenames_tensor = torch.load(f"activation_vectors_1281167_training/final_labels_torch_vector_{i}.pt")
    filenames_tensor_list += temp_filenames_tensor

cluster_list = torch.load(f"activation_vectors_1281167_training/Analysis/final_clusters.pt", weights_only=False)
cluster_labels_by_filename = dict(zip(filenames_tensor_list, cluster_list))



means_by_cluster = torch.load(f"activation_vectors_1281167_training/analysis/feature_means_by_clusters.pt", weights_only=False)
stds_by_cluster = torch.load(f"activation_vectors_1281167_training/analysis/feature_stds_by_clusters.pt", weights_only=False)


def filters_to_zero(activation_means, activation_deviations, cluster, activation_cutoff=0.1, std_cuttoff=0.25):
    filter_index_to_zero = []

    activations = activation_means[cluster]
    deviations = activation_deviations[cluster]

    for i, activation in enumerate(activations):
        if ((activation < activation_cutoff) & (deviations[i]<std_cuttoff)):
            filter_index_to_zero.append(i)

    return filter_index_to_zero



def zero_out_filters_by_activation_indices(model: nn.Module, activation_indices: list):
    """
    Zero out the filters in the network corresponding to the given list of activation indices.

    Parameters:
      model (nn.Module): The CNN model (e.g., AlexNet).
      activation_indices (list of int): A list of indices in the concatenated activation vector,
                                        each corresponding to a specific convolution filter.

    Raises:
      ValueError: If any activation index exceeds the total number of filters.
    """
    # Convert list to a set for faster lookup
    indices_to_zero = set(activation_indices)

    # Gather all leaf conv modules and sort them by name to match sorted(activations.keys())
    conv_modules = []
    for name, module in model.named_modules():
        # A leaf module is one with no children.
        if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
            conv_modules.append((name, module))
    conv_modules = sorted(conv_modules, key=lambda x: x[0])


    cumulative_filters = 0
    num_filters_zeroed = 0

    # Iterate over all leaf modules that are convolutional layers.
    for name, module in conv_modules:
        num_filters = module.out_channels
        # Determine which indices fall in the current layer.
        current_layer_indices = [idx - cumulative_filters for idx in indices_to_zero
                                 if cumulative_filters <= idx < cumulative_filters + num_filters]
        for filter_index in current_layer_indices:
            # Zero out the weights and bias (if present) of the corresponding filter.
            module.weight.data[filter_index].zero_()
            num_filters_zeroed += 1
            if module.bias is not None:
                module.bias.data[filter_index].zero_()
            #print(f"Zeroed out filter {filter_index} in layer '{name}'.")
        cumulative_filters += num_filters

        # Check for indices that exceed the total number of filters.
    remaining = [idx for idx in indices_to_zero if idx >= cumulative_filters]
    print(f"{num_filters_zeroed} of {cumulative_filters} filters pruned")
    #if remaining:
    #    print(f"Total filters counted: {cumulative_filters}")
    #    raise ValueError(f"The following activation indices are out of range: {remaining}")

    return num_filters_zeroed

class ImageNetWithClusters(datasets.ImageNet):
    def __getitem__(self, index):
        # Get the original image and label using the parent class method
        image, label = super().__getitem__(index)

        # The file path is stored in the dataset's samples attribute.
        # (ImageNet is similar to ImageFolder, which stores a list of (file_path, class_index) pairs)
        path = self.samples[index][0]

        cluster = cluster_labels_by_filename[path]

        # Return the image, label, and the file path.
        return image, label, cluster



class NanoNet(nn.Module):
    def __init__(self, num_classes=10):
        super(NanoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2304, 100),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(100, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        return out

cluster_classifier_model = NanoNet()

cluster_classifier_state_dict = torch.load(f"cluster_classifiers/nanonet/nanonet_final_weights.pt", weights_only=False)
cluster_classifier_model.load_state_dict(cluster_classifier_state_dict)



model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights='AlexNet_Weights.DEFAULT')

# Save a deep copy of the original state dict.
original_state_dict = copy.deepcopy(model.state_dict())

imagenet_data =ImageNetWithClusters("C:\\Users\\cmbec\\OneDrive\\Cloud_Documents\\Harvard\\NEUROBIO240\\AlexNet\\ClassificationSubsetDataset\\ILSVRC2012",
                                              "train",
                                              transform=preprocess)

data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)

## If you want to use the GPU, set GPU_MODE TO 1 in config file
device = torch.device(f"cuda:{CUDA_DEVICE}" if GPU_MODE and torch.cuda.is_available() else "cpu")
model.to(device)
cluster_classifier_model.to(device)


# Initialize counters for accuracy
#correct = 0
#total = 0

# Make sure to turn off gradients for evaluation
model.eval()


number_of_images_seen = 0
images_seen_100K_batch = 0
old_images_seen_100K_batch = 0


cuttoff_increment = 0
images_seen_increment = 0


grid_correct = np.empty((5, 5))
grid_total = np.empty((5, 5))
grid_accuracy = np.empty((5, 5))

grid_filters_removed = np.empty((5, 5))
grid_ave_filters_removed = np.empty((5, 5))




with torch.no_grad():
    for inputs, labels, clusters in data_loader:

        if(cuttoff_increment==24):
            cuttoff_increment = 0
            images_seen_increment += 1


            if(images_seen_increment%100 == 0):
                # Display the data as an image with a chosen colormap
                plt.imshow(grid_accuracy,
                               cmap='viridis',
                               interpolation='none',
                               extent=(standard_deviation_cuttoffs[0],
                                       standard_deviation_cuttoffs[-1],
                                       mean_activation_cuttoffs[0],
                                       mean_activation_cuttoffs[-1]),
                               origin='lower')
                plt.ylabel("Standard Deviation Cuttoff")
                plt.xlabel("Mean Activation Cuttoff")
                plt.colorbar()
                plt.title(f"Accuracy vs Cuttoffs w Nanonet({images_seen_increment} images per square)")
                plt.show()

                # Display the data as an image with a chosen colormap
                plt.imshow(grid_ave_filters_removed,
                           cmap='viridis',
                           interpolation='none',
                           extent=(standard_deviation_cuttoffs[0],
                                   standard_deviation_cuttoffs[-1],
                                   mean_activation_cuttoffs[0],
                                   mean_activation_cuttoffs[-1]),
                           origin='lower')
                plt.ylabel("Standard Deviation Cuttoff")
                plt.xlabel("Mean Activation Cuttoff")
                plt.colorbar()
                plt.title(f"Filters Removed vs Cuttoffs w Nanonet ({images_seen_increment} images per square)")
                plt.show()

        else:
            cuttoff_increment += 1

        std_dev_increment = 4 - int(cuttoff_increment/5)
        mean_act_increment = cuttoff_increment % 5


        # Send inputs and labels to the appropriate device
        inputs = inputs.to(device)
        labels = labels.to(device)
        #predicted_clusters = clusters.to(device)
        predicted_clusters = cluster_classifier_model(inputs)

        _, clusters = torch.max(predicted_clusters, 1)

        filters_to_set_zero = filters_to_zero(means_by_cluster,
                                              stds_by_cluster,
                                              clusters,
                                              activation_cutoff=mean_activation_cuttoffs[mean_act_increment],
                                              std_cuttoff=standard_deviation_cuttoffs[std_dev_increment])

        num_filters_zeroed = zero_out_filters_by_activation_indices(model, filters_to_set_zero)

        # Run the model forward
        outputs = model(inputs)

        # To reset the model back to the original weights:
        model.load_state_dict(original_state_dict)

        # Get the top 5 predictions for each input
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)

        # Check if the true label is in the top 5 predictions
        grid_correct[std_dev_increment][mean_act_increment] += torch.sum(top5_pred == labels.unsqueeze(1)).item()
        grid_total[std_dev_increment][mean_act_increment] += labels.size(0)
        grid_filters_removed[std_dev_increment][mean_act_increment] += num_filters_zeroed

        #print(f"Total so far: ", total)
        grid_accuracy[std_dev_increment][mean_act_increment] = 100 * grid_correct[std_dev_increment][mean_act_increment] / grid_total[std_dev_increment][mean_act_increment]
        grid_ave_filters_removed[std_dev_increment][mean_act_increment] = grid_filters_removed[std_dev_increment][mean_act_increment] / grid_total[std_dev_increment][mean_act_increment]
        #print(f"Accuracy on the validation set: {grid_accuracy[std_dev_increment][mean_act_increment]:.2f}%")



