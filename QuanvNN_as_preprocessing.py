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

from Quanv2D import Quanv2D

from darqk.core import Ansatz

import matplotlib.pyplot as plt

import random
import math
#random.seed(12345)
#np.random.seed(12345)

THRESHOLD = "THRESHOLD"
ROTATIONAL = "ROTATIONAL"


class QuanvNN(nn.Module):
    def __init__(self, verbose = True):
        super(QuanvNN, self).__init__()
        self.quanv = Quanv2D(1, 10, kernel_size=3, verbose = verbose)
        #self.conv1 = nn.Conv2d(10, 50, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(10, 64, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.quanv(x))) #28x28 -> 13x13
        #x = self.pool(F.relu(self.conv1(x))) #28x28 -> 24x24 -> 12x12 [50]
        x = self.pool(F.relu(self.conv2(x))) #5x5 [64]?
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
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

def load_mnist(batch_size, train_ratio=0.8):
    n = 28
    transform = transforms.Compose([
        transforms.Resize((n, n)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Load the full dataset
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Reduce dataset to 1/20th of its original size
    reduced_dataset_size = len(full_dataset) // 200
    reduced_dataset, _ = random_split(full_dataset, [reduced_dataset_size, len(full_dataset) - reduced_dataset_size])

    # Split the reduced dataset into training and testing sets
    train_size = int(train_ratio * reduced_dataset_size)
    test_size = reduced_dataset_size - train_size
    train_dataset, test_dataset = random_split(reduced_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def main():
    device = torch.device("cpu")
    model = QuanvNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = load_mnist(batch_size=64)

    for epoch in range(1, 11):  # 10 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == "__main__":
    main()