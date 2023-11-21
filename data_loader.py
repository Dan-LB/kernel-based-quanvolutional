import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Optional imports for data visualization and manipulation
import matplotlib.pyplot as plt
import numpy as np

def load_mnist(batch_size, download=True, train=True):
    """Load MNIST data using PyTorch datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the MNIST dataset
    ])

    dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def main():
    batch_size = 64  # You can modify this as needed

    # Load the datasets
    train_loader = load_mnist(batch_size=batch_size, train=True)
    test_loader = load_mnist(batch_size=batch_size, train=False)

    # Example: Visualizing the first batch of the training dataset
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # Optional: Use Matplotlib to visualize some examples
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
