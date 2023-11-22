import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import os
import numpy as np



import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from PIL import Image

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info import Statevector

from cnn import CNN
from QuanvNN import QuanvNN
from CustomDataset import CustomDataset, load_custom_dataset

from darqk.core import Ansatz

from util import test, train

import matplotlib.pyplot as plt

import random
import math
#random.seed(12345)
#np.random.seed(12345)

import constants

def get_data(n = 200, size = 10):
    # Define the transform to preprocess the MNIST images
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Download the MNIST dataset and apply the transform
    mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Set the number of training examples you want in the batch

    # Create a DataLoader with a batch size of n_train
    train_loader = DataLoader(mnist_train, batch_size=n, shuffle=True)

    # Iterate through the DataLoader to get the first batch
    for batch_idx, (data, labels) in enumerate(train_loader):
        # data is a tensor of shape (n_train, 1, 28, 28), labels is a tensor of shape (n_train,)
        #print(f"Batch {batch_idx + 1}:")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        return data, labels  # Stop after the first batch


def create_and_process(n, size, model, folder_name):
    X, y = get_data(n, size)
    q_X = model.preprocess_dataset(X)

    print(f"\nProcessing dataset of {n} images...\n\n")

    path = constants.SAVE_PATH + "\\" + folder_name 

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path + "\images.npy", X)
    np.save(path + "\q_images.npy", q_X)
    np.save(path + "\labels.npy", y)

    with open(os.path.join(path, "model_info.txt"), "w") as info_file:

        model.info["Images processed: "] = n

        for key, value in model.info.items():
            info_file.write(f"{key}: {value}\n")

def plot_some():
    first_image = q_images[0]
# Plot each channel as a black and white image
    for channel in range(channels):
        plt.subplot(1, channels, channel + 1)
        plt.imshow(first_image[channel], cmap='gray')
        plt.axis('off')  # Turn off axis labels and ticks
        plt.title(f'Channel {channel + 1}')
        print(first_image[channel])
    plt.show()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

quanvPQC_model = QuanvNN(kernel_size=3, out_channels=10, quanv_model=constants.RANDOM_PQC, 
                         PQC_qubits=4, PQC_L=15,
                         verbose=True)

quanvVQC_model = QuanvNN(kernel_size=3, out_channels=10, quanv_model=constants.RANDOM_VQC, 
                         VQC_n_shots=1000, VQC_encoding=constants.ROTATIONAL,
                         verbose=True)

folder_name = "quanvPQC"


create_and_process(n=30, size = 10, model = quanvPQC_model, folder_name=folder_name)

load_path = constants.SAVE_PATH + "\\" + folder_name

q_images = np.load(load_path + "\q_images.npy")
q_labels = np.load(load_path + "\labels.npy")
images = np.load(load_path + "\images.npy")


train_loader, test_loader = load_custom_dataset(batch_size=64, npy_file=q_images, labels_file=q_labels)

print("Printing new dataset shape:")
n_train, channels, w, h = q_images.shape
print(q_images.shape)
print("Printing old dataset shape:")
print(images.shape)


quanvPQC_model.on_preprocessed = True
optimizer = optim.Adam(quanvPQC_model.parameters(), lr=0.001)
device = torch.device("cpu")

for epoch in range(1, 35):  # 100 epochs
    train(quanvPQC_model, device, train_loader, optimizer, epoch)
    loss, correct = test(quanvPQC_model, device, test_loader)



