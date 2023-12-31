import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from copy import deepcopy

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
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1, padding=0, verbose = False, n_qubits = 10, L=10):
        super(PQCQuanv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.simulator = Aer.get_backend('qasm_simulator')

        self.n_qubits = n_qubits
        self.L = L

        self.verbose = verbose

        self.values = None

        #self.copy_of_model = None

        self.parent = None

        # Initialize weights and bias

        start_time = time.time()
        print(f"Initializing Quanv2D module as {self.out_channels} random PQCs, kernel size = {self.kernel_size}, {self.n_qubits} qubits and length {self.L}.")
        self.circuits = self.generate_random_PQC_layer()
        end_time = time.time()
        print(f"Time required to generate circuit layer: {end_time - start_time:.2f} seconds")


    def forward(self, x):

        # Calculate output dimensions
        batch_size, _, height, width = x.shape
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1

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
                        
                        # Apply padding to input tensor
                        padded_x = torch.nn.functional.pad(x[i], (self.padding, self.padding, self.padding, self.padding))
                        patch = padded_x[:, h_start:h_end, w_start:w_end].numpy()
                        patch = patch[0] * math.pi * 2
                        output[i, j, h, w] = self.to_quanvolute_patch(self.circuits[j], patch)

            image_end_time = time.time()
            image_time = image_end_time - image_start_time
            elapsed_time = image_end_time - start_time
            average_time_per_image = elapsed_time / (i + 1)
            estimated_remaining_time = average_time_per_image * (batch_size - i - 1)
            
            if self.verbose == True and i%10==9:
                print(f"Time for image {i+1}: {image_time:.2f} seconds")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes\n")

        return output
    

    def generate_random_PQC(self):

        """

        """

        print("Ancora di nuovo devo ottimizzare il circuito!!!!!!!!!")

        new_ansatz = Ansatz(self.kernel_size**2, self.n_qubits, self.L)
        new_ansatz.initialize_to_random_circuit()

        print(new_ansatz)

        qc = to_qiskit_circuit(new_ansatz)

        print(qc)

        kernel = KernelFactory.create_kernel(new_ansatz, "Z"*self.n_qubits, KernelType.OBSERVABLE)

        to_transpile_k = kernel.to_qiskit_circuit()
        print(to_transpile_k)
 
        new_ansatz = transpile(to_transpile_k, self.simulator)
        print(new_ansatz)
        print("\n\n")
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
    

    def optimize_layer(self, X_and_y, parent):

        print("Initializing optimization phase.")
        print("\nALERT: some stuff should be changed (e.g. the X_and_y)... :)\n")

        K = self.obtain_big_K_from_layer()

        ke = LayerEvaluator(X_and_y, self, deepcopy(parent))

        X, Y = None, None
        bayesian_opt = BayesianOptimizer(K, X, Y, ke)
        K = bayesian_opt.optimize(5, 5, 5) ###!!!!!!!!!!

        self.circuits = self.obtain_layer_from_big_K(K)
        return 0
    
    def obtain_big_K_from_layer(self):
        op_list = []
        for circuit in self.circuits:
            #print(circuit.ansatz.operation_list)
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
        
        


def to_qiskit_circuit(self):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    import math

    qc = QuantumCircuit(self.n_qubits)
    params = ParameterVector('x', self.n_features)
    print(params)
    qc.rx(0 * math.prod(params), 0)


    for i in range(self.n_operations):
        op = self.operation_list[i]
        feature = params[op.feature] if op.feature in range(self.n_features) else 1.0
        operator = SparsePauliOp(op.generator)
        operator._coeffs = operator._coeffs.astype(float)
        gate = PauliEvolutionGate(operator, time=op.bandwidth * feature)
        print(gate)
        print(op.wires)
        if op.generator != "II":
            qc.append(gate, op.wires)

    return qc