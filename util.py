import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import os
import numpy as np



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
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss, 100. * correct / len(test_loader.dataset)

def extract_patches(images, n):
    patch_size = (3, 3)  # Size of the patches to extract

    # Create an empty list to store the extracted patches
    extracted_patches = []

    size, channels, h, w = images.size()

    # Loop to extract n patches
    for _ in range(n):
        # Randomly select an image from the dataset
        random_image_index = np.random.randint(0, size)
        image = images[random_image_index]
        # Randomly select the top-left corner of the patch within the image
        top_left_x = np.random.randint(0, w - patch_size[1] + 1)
        top_left_y = np.random.randint(0, h - patch_size[0] + 1)
        # Extract the patch
        patch = image[:, top_left_y:top_left_y+patch_size[0], top_left_x:top_left_x+patch_size[1]]
        # Append the patch to the list of extracted patches
        extracted_patches.append(patch[0])
    # The extracted_patches list now contains n 3x3 patches as NumPy arrays 
    return extracted_patches
