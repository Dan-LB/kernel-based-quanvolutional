import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import time

import os
import numpy as np

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from PIL import Image

from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info import Statevector

from darqk.core import Ansatz, Kernel, KernelFactory, KernelType
from darqk.evaluator.custom_evaluator import CustomEvaluator
from darqk.evaluator.layer_evaluator import LayerEvaluator
from darqk.optimizer import BayesianOptimizer

import matplotlib.pyplot as plt

import random
import math

import constants

from util import extract_patches

from sklearn.metrics.pairwise import cosine_similarity

from itertools import product


class PQCQuanv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1, verbose = False, n_qubits = 10, L=10):
        super(PQCQuanv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.simulator = Aer.get_backend('qasm_simulator')

        self.n_qubits = n_qubits
        self.L = L

        self.verbose = verbose

        self.values = None

        self.copy_of_model = None

        # Initialize weights and bias

        start_time = time.time()
        print(f"Initializing Quanv2D module as {self.out_channels} random PQCs, kernel size = {self.kernel_size}, {self.n_qubits} qubits and length {self.L}.")
        self.circuits = self.generate_random_PQC_layer()
        end_time = time.time()
        print(f"Time required to generate circuit layer: {end_time - start_time:.2f} seconds")


    def forward(self, x):

        # Calculate output dimensions
        batch_size, _, height, width = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1


        # Create output tensor
        output = torch.zeros((batch_size, self.out_channels, out_height, out_width))

        # Perform convolution
        start_time = time.time()
        for i in range(batch_size):
            image_start_time = time.time()

            for j in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        patch = x[i, :, h_start:h_end, w_start:w_end].numpy()
                        patch = patch[0]*math.pi/2
                        output[i, j, h, w] = self.to_quanvolute_patch(self.circuits[j], patch)

            image_end_time = time.time()
            image_time = image_end_time - image_start_time
            elapsed_time = image_end_time - start_time
            average_time_per_image = elapsed_time / (i + 1)
            estimated_remaining_time = average_time_per_image * (batch_size - i - 1)
            
            if self.verbose and i%10==9:
                print(f"Time for image {i+1}: {image_time:.2f} seconds")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes\n")

        return output
    

    def generate_random_PQC(self):

        """

        """
        new_ansatz = Ansatz(self.kernel_size**2, self.n_qubits, self.L)
        new_ansatz.initialize_to_random_circuit()
        #new_ansatz = transpile(new_ansatz, self.simulator)
        kernel = KernelFactory.create_kernel(new_ansatz, "Z"*self.n_qubits, KernelType.OBSERVABLE)

        return kernel

    def generate_random_PQC_layer(self):
        layer = []
        for i in range(self.out_channels):
            random_PQC = self.generate_random_PQC()
            layer.append(random_PQC)
        return layer

    def to_quanvolute_patch(self, circuit, patch):
        output = circuit.phi(patch.ravel())
        return output
    
    def generate_patches(self, images, n_patches):
        self.patches = extract_patches(images, n=n_patches)

    def compute_similarity_array(self):
        """
        In realtà ci sono due tipi di cose che posso provare a calcolare:
        1) fidelity (ma che è? come si calcola? è dataset dependant?)
        2) la "patches output similarity" che mi sembra facile e naturale
        

        qualche calcoletto preliminare
            10 PQC (3x3, 15 op, 4 qubit) richiedono su una pic 10x10 (ovvero 8x8 = 64 applicazioni) 4 secondi circa
            supponendo di avere in fase iniziale 10 PQC da ottimizzare, con 100 patch (calcolabili in meno di 4 secondi)
            supponendo di usare un BO con 5 applicazioni per iterazione, 10 iterazioni
            dovremmo calcolare per ogni PQC, 50 possibili PQC per ogni patch, quindi 5000 applicazioni, meno di 40 secondi
            un ciclo completo di ottimizzazione richiede quindi 400 secondi
            ottimizzando iterativamente 10 volte, abbiamo 4000 secondi, circa 1h

            
        lato similarità, alcune idee:   
            MSE (banale)
            Average KATZ Index
            Distanza dallo span lineare 
        """


        patches = self.patches
        n_patches = len(patches)

        n_circuits = self.out_channels
        values = np.zeros((n_circuits, n_patches))
        circuits = self.circuits
        for i in range(n_patches):
            current_patch = patches[i].numpy()
            for j in range(n_circuits):
                #print(circuits[j])
                values[j, i] =  self.to_quanvolute_patch(circuits[j], current_patch)

        self.values = values
        return values

    def compute_similarity_from_values(self, i):
        similarities = cosine_similarity(self.values)
        average_similarity = np.mean(similarities[i])
        return average_similarity

    def optimize_PQC_i(self, i):
        X, Y = None, None

        kernel = self.circuits[i]
        ke = CustomEvaluator(i, self.values, self.patches)
        bayesian_opt = BayesianOptimizer(kernel, X, Y, ke)
        kernel = bayesian_opt.optimize(5, 5, 1)
        self.circuits[i] = kernel
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.compute_similarity_array() #forse questo serve :)

    def optimize_layer(self, X_and_y, L, copy_of_model):

        K = self.obtain_big_K_from_layer()

        ke = LayerEvaluator(X_and_y, L, copy_of_model)

        X, Y = None, None
        bayesian_opt = BayesianOptimizer(K, X, Y, ke)
        K = bayesian_opt.optimize(5, 5, 1)

        self.circuits = self.obtain_big_K_from_layer(K)
        return 0
    
    def obtain_big_K_from_layer(self):
        op_list = []
        for circuit in self.circuits:
            print(circuit.ansatz.operation_list)
            for op in circuit.ansatz.operation_list:
                op_list.append(op)

        new_ansatz = Ansatz(self.kernel_size**2, self.n_qubits, self.L*self.out_channels)
        new_ansatz.operation_list = op_list
        #new_ansatz = transpile(new_ansatz, self.simulator)
        K = KernelFactory.create_kernel(new_ansatz, "Z"*self.n_qubits, KernelType.OBSERVABLE)
        
        return K    
    
    def obtain_layer_from_big_K(self, K):
        new_layer = []
        for i in range(0, len(K.ansatz.operation_list), self.L):
            # Extract a subset of operations for a smaller kernel
            subset_operations = K.ansatz.operation_list[i:i + self.L]

            # Create a new ansatz for the smaller kernel with appropriate parameters
            new_ansatz = Ansatz(self.kernel_size**2, self.n_qubits, self.L) # Add other necessary parameters

            # Assign the subset of operations to the new ansatz
            new_ansatz.operation_list = subset_operations

            # Create a new kernel with the new ansatz
            new_kernel = KernelFactory.create_kernel(new_ansatz, "Z"*self.n_qubits, KernelType.OBSERVABLE)
            
            # Add the new kernel to the layer
            new_layer.append(new_kernel)

            # Break the loop if the desired number of kernels (self.out_channels) is reached
            if len(new_layer) == self.out_channels:
                return new_layer
        
        


