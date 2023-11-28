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

from util import test, train

from PIL import Image

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info import Statevector

from cnn import CNN
from QuanvNN import QuanvNN
from CustomDataset import CustomDataset, load_custom_dataset

from darqk.core import Ansatz



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
        transforms.ToTensor(),
        transforms.Normalize(0.0, 1.0)  # Convert images to PyTorch tensors
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
    print(f"\nProcessing dataset of {n} images...\n\n")

    q_X = model.preprocess_dataset(X)

    

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
    first_image = q_images[0]*255
# Plot each channel as a black and white image
    for channel in range(10):
        plt.subplot(1, 10, channel + 1)
        plt.imshow(first_image[channel], cmap='gray')
        plt.axis('off')  # Turn off axis labels and ticks
        plt.title(f'Channel {channel + 1}')
        print(first_image[channel])
    plt.show()

quanvPQC_model = QuanvNN(kernel_size=3, out_channels=4, quanv_model=constants.RANDOM_PQC, 
                         PQC_qubits=4, PQC_L=15,
                         verbose=True)
"""
quanvVQC_model = QuanvNN(kernel_size=3, out_channels=10, quanv_model=constants.RANDOM_VQC, 
                         VQC_n_shots=1000, VQC_encoding=constants.ROTATIONAL,
                         verbose=True)

classicalCNN_model = QuanvNN(kernel_size=3, out_channels=50, quanv_model=constants.CLASSICAL_CNN,
                             verbose=True)
"""


folder_name = "opt_test"

X, y = get_data(n=100, size=10)

X_and_y = (X.numpy(), y.numpy())

copy_of_model = QuanvNN(kernel_size=3, out_channels=4, quanv_model=constants.RANDOM_PQC, 
                                        PQC_qubits=4, PQC_L=15,
                                        verbose=True)

quanvPQC_model.quanv.optimize_layer(X_and_y, quanvPQC_model.quanv, copy_of_model) #<--- palesemente idea stupida

#create_and_process(n=100, size = 10, model = quanvPQC_model, folder_name=folder_name)

#X, _ = get_data(n=10, size=10)


#raise Exception("Sto testando altre robe")



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



