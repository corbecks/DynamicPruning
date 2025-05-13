import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler

filenames_tensor_list = []
cluster_list = []

for i in range(13):
    temp_filenames_tensor = torch.load(f"activation_vectors_1281167_training/final_labels_torch_vector_{i}.pt")
    filenames_tensor_list += temp_filenames_tensor

cluster_list = torch.load(f"activation_vectors_1281167_training/Analysis/final_clusters.pt", weights_only=False)

cluster_labels_by_filename = dict(zip(filenames_tensor_list, cluster_list))


# GPU SETTINGS
CUDA_DEVICE = 0  # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1  # set to 1 if want to run on gpu.
BATCH_SIZE = 128

# HELPER FUNCTIONS
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageNetWithPaths(datasets.ImageNet):
    def __getitem__(self, index):
        # Get the original image and label using the parent class method
        image, label = super().__getitem__(index)

        # The file path is stored in the dataset's samples attribute.
        # (ImageNet is similar to ImageFolder, which stores a list of (file_path, class_index) pairs)
        path = self.samples[index][0]

        label = cluster_labels_by_filename[path]

        # Return the image, label, and the file path.
        return image, label, path

class FemtoNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FemtoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2028, 100),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(100, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        return out

model = FemtoNet()


if __name__ == '__main__':
    imagenet_data =ImageNetWithPaths("C:\\Users\\cmbec\\OneDrive\\Cloud_Documents\\Harvard\\NEUROBIO240\\AlexNet\\ClassificationSubsetDataset\\ILSVRC2012",
                                                  "train",
                                                  transform=preprocess)

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=4)

    ## If you want to use the GPU, set GPU_MODE TO 1 in config file
    device = torch.device(f"cuda:{CUDA_DEVICE}" if GPU_MODE and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize counters for accuracy
    correct = 0
    total = 0

    # Make sure to turn off gradients for evaluation
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10  # Define how many epochs you want to train for

    counter = 0

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Set model to training mode
        model.train()

        for inputs, labels, file_names in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if(counter % 1000 == 0):
                print(counter * BATCH_SIZE)
            counter += 1

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients from previous step
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Statistics for monitoring training progress (optional)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * correct / total

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        plt.plot(epoch_losses)
        plt.plot(epoch_accuracies)
        plt.show()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        torch.save(model.state_dict(), f"femtonet_epoch_{epoch+1}_weights.pt")

        torch.save(epoch_losses, "femtonet_epoch_losses.pt")
        torch.save(epoch_accuracies, "femtonet_epoch_accuracies.pt")
        torch.save(model.state_dict(), "femtonet_final_weights.pt")

    plt.plot(epoch_losses)
    plt.plot(epoch_accuracies)
    plt.show()
