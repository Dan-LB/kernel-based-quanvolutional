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

from Quanvs import PQCQuanv, VQCQuanv

from darqk.core import Ansatz

import matplotlib.pyplot as plt

import random
import math

import constants

#random.seed(12345)
#np.random.seed(12345)


class QuanvNN(nn.Module):
    #aggiungi anche input size.......... in modo da calcolare altre robe in automatico?
    def __init__(self, kernel_size = 3, out_channels = 10, quanv_model = None, verbose = False,
                 VQC_n_shots = None,  VQC_encoding = None,
                 PQC_qubits = None, PQC_L = None
                 ):
        super(QuanvNN, self).__init__()

        self.quanv_model = quanv_model
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.info = {
                "kernel size": self.kernel_size,
                "out channels": self.out_channels,
                "quanvolutional model": self.quanv_model
        }

        self.verbose = verbose

        if self.quanv_model == constants.RANDOM_VQC:
            self.quanv = VQCQuanv(1, out_channels = self.out_channels, kernel_size=self.kernel_size,
                                  n_shots=VQC_n_shots, encoding=VQC_encoding, verbose = verbose)
            
            self.info["VQC_encoding"] = VQC_encoding
            self.info["VQC_n_shots"] = VQC_n_shots


        elif self.quanv_model == constants.RANDOM_PQC:
            print("TO DO: check the decoding mod!!!")
            self.quanv = PQCQuanv(1, out_channels = self.out_channels, kernel_size=self.kernel_size, verbose = verbose, 
                                  n_qubits = PQC_qubits, L=PQC_L)
            self.info["PQC_qubits"] = PQC_qubits
            self.info["PQC_L"] = PQC_L

        elif self.quanv_model == constants.CLASSICAL_CNN:
            print("Class cnn preferred")
            self.quanv = nn.Conv2d(1, out_channels = self.out_channels, kernel_size=self.kernel_size, padding=0)

        elif self.quanv_model == constants.OTHER_MODEL:
            raise Exception("We want to implement some trainable variational algorithm but we don't have it yet")
        
        else:
            raise Exception("Unknown model name")

        self.conv2 = nn.Conv2d(self.out_channels, 64, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self.on_preprocessed = False

        if self.verbose:
            self.print_network_structure()


    def forward(self, x):
        if self.on_preprocessed:
            x = x.permute(0, 2, 1, 3)
            x = self.pool(F.relu(x))
        else:
            x = self.pool(F.relu(self.quanv(x))) #10x10 -> 4x4 [10]
   
        x = self.pool(F.relu(self.conv2(x))) #1x1 [64]?
        x = x.view(-1, 64 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def preprocess_dataset(self, images):
            q_images = self.quanv(images)
            return q_images


    def print_network_structure(self):
        print(self)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

