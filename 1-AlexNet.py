import torch
import torchvision
from torchvision import transforms


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

import torchvision.datasets as datasets

import os


# GPU SETTINGS
CUDA_DEVICE = 0  # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1  # set to 1 if want to run on gpu.
BATCH_SIZE = 1000

# HELPER FUNCTIONS
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Define a dictionary to store the activations
activations = {}


def get_activation(name):
    # This function returns a hook that saves the output in the activations dict
    def hook(model, input, output):
        # Detach the output and store it using the given name
        activations[name] = output.detach()
    return hook


class ImageNetWithPaths(datasets.ImageNet):
    def __getitem__(self, index):
        # Get the original image and label using the parent class method
        image, label = super().__getitem__(index)

        # The file path is stored in the dataset's samples attribute.
        # (ImageNet is similar to ImageFolder, which stores a list of (file_path, class_index) pairs)
        path = self.samples[index][0]

        # Return the image, label, and the file path.
        return image, label, path


def cluster(data_tensor, num_clusters = 3):
    with threadpool_limits(limits=5):
        data_np = data_tensor.numpy()  # Convert to NumPy for scikit-learn compatibility

        # Step 1: Reduce dimensionality with PCA for visualization (2 components)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_np)

        # Step 2: Run K-Means clustering on the original or PCA-transformed data.
        # Here, we'll cluster using the original data.
        #kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=10, batch_size = 5000)
        #cluster_labels = kmeans.fit_predict(data_np)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_np)  # Alternatively, use data_pca

        # Step 3: Visualize the PCA projection and color-code by cluster labels

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='viridis', s=100)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Projection of 10 Samples with Cluster Assignments')
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()


model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights='AlexNet_Weights.DEFAULT')

# Register forward hooks on all leaf modules (modules without children)
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Only register on layers that aren't containers
        module.register_forward_hook(get_activation(name))

imagenet_data =ImageNetWithPaths("C:\\Users\\cmbec\\OneDrive\\Cloud_Documents\\Harvard\\NEUROBIO240\\AlexNet\\ClassificationSubsetDataset\\ILSVRC2012",
                                              "train",
                                              transform=preprocess)

#imagenet_data = torchvision.datasets.ImageNet("C:\\Users\\cmbec\\OneDrive\\Cloud_Documents\\Harvard\\NEUROBIO240\\AlexNet\\ClassificationSubsetDataset\\ILSVRC2012",
#                                              "val",
#                                              transform=preprocess)

data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)

## If you want to use the GPU, set GPU_MODE TO 1 in config file
device = torch.device(f"cuda:{CUDA_DEVICE}" if GPU_MODE and torch.cuda.is_available() else "cpu")
model.to(device)


# Initialize counters for accuracy
correct = 0
total = 0

# Make sure to turn off gradients for evaluation
model.eval()

final_activations_vector = []
final_labels_vector = []

number_of_images_seen = 0
images_seen_100K_batch = 0
old_images_seen_100K_batch = 0

with torch.no_grad():
    for inputs, labels, file_names in data_loader:
        # Send inputs and labels to the appropriate device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Run the model forward
        outputs = model(inputs)

        # Get the top 5 predictions for each input
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)

        # Check if the true label is in the top 5 predictions
        correct += torch.sum(top5_pred == labels.unsqueeze(1)).item()
        total += labels.size(0)

        # Get the predicted class (top-1)
        #_, predicted = torch.max(outputs, 1)
        # Update counts
        #correct += (predicted == labels).sum().item()

        print(f"Total so far: ", total)
        accuracy = 100 * correct / total
        print(f"Accuracy on the validation set: {accuracy:.2f}%")

        activation_vectors = []
        for key in sorted(activations.keys()):
            act = activations[key]  # shape: [batch_size, ...]
            #print(key, act.shape)

            if act.dim() == 4:
                # Compute the average over the last two dimensions.
                x_avg = act.mean(dim=(2, 3), keepdim=True)
                x_avg = x_avg.squeeze(-1)  # Reduces extra dimensions
                x_avg = x_avg.squeeze(-1)  # Reduces extra dimensions
                activation_vectors.append(x_avg)

        # Concatenate along the feature dimension to get a big vector per input
        batch_activations_vector = torch.cat(activation_vectors, dim=1)
        for vec in batch_activations_vector:
            final_activations_vector.append(vec.detach().cpu())
            number_of_images_seen += 1

        for filename in file_names:
            final_labels_vector.append(filename)



        images_seen_100K_batch = int(number_of_images_seen / 100000)

        if (images_seen_100K_batch > old_images_seen_100K_batch):
            final_activations_torch_vector = torch.stack(final_activations_vector)

            # Save the list to a file
            torch.save(final_labels_vector,
                       f"final_labels_torch_vector_{old_images_seen_100K_batch}.pt")
            torch.save(final_activations_torch_vector,
                       f"final_activations_torch_vector_{old_images_seen_100K_batch}.pt")

            final_activations_vector = []
            final_labels_vector = []

        old_images_seen_100K_batch = images_seen_100K_batch

        #cluster(final_activations_torch_vector, num_clusters=10)

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy on the validation set: {accuracy:.2f}%")
