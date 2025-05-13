import torch
import torch.nn as nn

def calculate_multiplications(model, input_size):
    # Function to calculate number of multiplications in a given CNN model
    total_multiplications = 0
    input_channels = input_size[0]  # Assuming the input is a tensor (C, H, W)
    input_height = input_size[1]
    input_width = input_size[2]

    # Helper function to recursively go through all layers
    def count_layer_multiplications(layer, input_channels, input_height, input_width):
        nonlocal total_multiplications

        if isinstance(layer, nn.Conv2d):
            # Get convolution layer parameters
            kernel_size = layer.kernel_size[0]  # Assuming square kernels
            stride = layer.stride[0]
            padding = layer.padding[0]
            output_channels = layer.out_channels

            # Calculate the output size of the convolution layer
            output_height = (input_height - kernel_size + 2 * padding) // stride + 1
            output_width = (input_width - kernel_size + 2 * padding) // stride + 1

            # Calculate number of multiplications for this convolution layer
            multiplications = output_height * output_width * output_channels * kernel_size * kernel_size * input_channels
            total_multiplications += multiplications

            # Update input size for the next layer (output of this convolution layer)
            return output_channels, output_height, output_width

        elif isinstance(layer, nn.Linear):
            # For fully connected layers
            in_features = layer.in_features
            out_features = layer.out_features

            # Multiplications for the fully connected layer
            multiplications = in_features * out_features
            total_multiplications += multiplications

            return out_features, 1, 1  # Fully connected layer doesn't affect spatial dimensions

        else:
            # For other layers (like ReLU, BatchNorm, etc.), we just return the input size as is
            return input_channels, input_height, input_width

    # Loop through the model layers recursively
    for child in model.children():
        if isinstance(child, nn.Sequential):
            for sub_layer in child:
                input_channels, input_height, input_width = count_layer_multiplications(sub_layer, input_channels, input_height, input_width)
        else:
            input_channels, input_height, input_width = count_layer_multiplications(child, input_channels, input_height, input_width)

    return total_multiplications

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


# Example usage with your FemtoNet model:
model = NanoNet()
input_size = (3, 224, 224)  # Input size for RGB image (C, H, W)

total_multiplications = calculate_multiplications(model, input_size)
print(f"Total number of multiplications: {total_multiplications}")
