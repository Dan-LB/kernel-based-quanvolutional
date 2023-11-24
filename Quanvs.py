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
from darqk.optimizer import BayesianOptimizer

import matplotlib.pyplot as plt

import random
import math

import constants

from util import extract_patches

from sklearn.metrics.pairwise import cosine_similarity


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
                        patch = patch[0]
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

    def optimize_PQC_i(self, i, values):
        X, Y = None, None

        kernel = self.circuits[i]
        ke = CustomEvaluator(i, self.values, self.patches)
        bayesian_opt = BayesianOptimizer(kernel, X, Y, ke)
        kernel = bayesian_opt.optimize(5, 5, 1)
        self.circuits[i] = kernel
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.compute_similarity_array()






class VQCQuanv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_shots, stride=1, encoding = None, verbose = False):
        super(VQCQuanv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.encoding = encoding
        self.simulator = Aer.get_backend('qasm_simulator')
        self.n_shots = n_shots
        self.verbose = verbose

        # Initialize weights and bias
        if self.verbose:
                    start_time = time.time()
                    print(f"Initializing Quanv2D module with {self.out_channels} circuits, kernel size = {self.kernel_size}")
                    self.circuits = self.generate_random_quantum_circuit_layer(out_channels, filter_size=kernel_size)
                    end_time = time.time()
                    print(f"Time required to generate circuit layer: {end_time - start_time:.2f} seconds")
        else:
            self.circuits = self.generate_random_quantum_circuit_layer(out_channels, filter_size=kernel_size)

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
                        patch = patch[0]
                        output[i, j, h, w] = self.to_quanvolute_patch(self.circuits[j], patch, encoding=self.encoding)

            image_end_time = time.time()
            image_time = image_end_time - image_start_time
            elapsed_time = image_end_time - start_time
            average_time_per_image = elapsed_time / (i + 1)
            estimated_remaining_time = average_time_per_image * (batch_size - i - 1)
            
            if self.verbose and i%10==9:
                print(f"Time for image {i+1}: {image_time:.2f} seconds")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes\n")

        return output
    

    def generate_random_quantum_circuit(self, filter_size, connection_prob = 0.15):

        """
        Generate... according to [quanvolutional]
        3x3 on 9 qubits
        """
        
        n_qubits = filter_size*filter_size
        one_qb_list = ["X", "Y", "Z",  "P", "T", "H"]
        two_qb_list = ["Cnot", "Swap", "SqrtSwap"] #check on SqrtSwap
        gate_list = []

        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j and random.random() < connection_prob:
                    g_index = random.randint(0, len(two_qb_list)-1)
                    gate_list.append({"gate": two_qb_list[g_index], "first_q": i, "second_q": j})

        n_one_qg =   random.randint(0, 2*filter_size*filter_size)   
        for i in range(n_one_qg):
            q = random.randint(0, n_qubits-1) 
            g_index = random.randint(0, len(one_qb_list)-1)  
            gate_list.append({"gate": one_qb_list[g_index], "first_q": q})

        random.shuffle(gate_list)

        circuit = QuantumCircuit(n_qubits)
        for gate in gate_list:
            theta = random.random()*math.pi
            if gate["gate"] == "Cnot":
                circuit.cx(gate["first_q"], gate["second_q"])
            elif gate["gate"] == "Swap":
                circuit.swap(gate["first_q"], gate["second_q"])
            elif gate["gate"] == "SqrtSwap":
                # Implementing SQRTSWAP using Qiskit's standard gates
                circuit.rxx(math.pi / 2, gate["first_q"], gate["second_q"])
                circuit.ryy(math.pi / 2, gate["first_q"], gate["second_q"])
            elif gate["gate"] == "RX":
                circuit.rx(theta, gate["first_q"])
            elif gate["gate"] == "RY":
                circuit.ry(theta, gate["first_q"])
            elif gate["gate"] == "RZ":
                circuit.rz(theta, gate["first_q"])
            elif gate["gate"] == "P":
                circuit.p(math.pi / 2, gate["first_q"])  # Phase gate
            elif gate["gate"] == "T":
                circuit.t(gate["first_q"])
            elif gate["gate"] == "H":
                circuit.h(gate["first_q"])


        circuit = transpile(circuit, self.simulator)
        #circuit_image = circuit_drawer(circuit, output='mpl', style="iqp")
        #circuit_image.savefig('quantum_circuit.png')
        return circuit

    def generate_random_quantum_circuit_layer(self, n, filter_size = 3, connection_prob = 0.15):
        layer = []
        for i in range(n):
            layer.append(self.generate_random_quantum_circuit(filter_size=filter_size, connection_prob=connection_prob))
        return layer

    def to_quanvolute_patch(self, circuit, patch, encoding):
        n = len(patch)

        if encoding == constants.THRESHOLD: #IN 0,1
            emb = QuantumCircuit(n*n)
            for i in range(n*n):
                row = i // n
                col = i % n
                if patch[row][col] >= 0.5:
                    emb.x(i)

        if encoding == constants.ROTATIONAL:
            emb = QuantumCircuit(n*n)
            for i in range(n*n):
                row = i // n
                col = i % n
                theta = patch[row][col]*3.14
                emb.rx(theta, i)
        

        combined_circuit = combined_circuit = emb.compose(circuit)
        combined_circuit.measure_all()

        #circuit_image = circuit_drawer(combined_circuit, output='mpl', style="iqp")
        #circuit_image.savefig('quantum_circuit.png')

        
        result = execute(combined_circuit, self.simulator, shots=self.n_shots).result()
        counts = result.get_counts(combined_circuit)
        #print(counts)
        #plot_histogram(counts)

        #plot_histogram(counts).savefig('histogram.png')
        sum_1s = sum(key.count('1') * count for key, count in counts.items()) / self.n_shots / (n*n)

            #print(sum_1s)

        return sum_1s