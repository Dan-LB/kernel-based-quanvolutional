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

from itertools import product


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

        self.look_up = {}

        self.counter1 = 0
        self.counter2 = 0

        self.discretizer = 2

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

                        if self.encoding == constants.THRESHOLD:
                            patch = discretize_patch(patch, 3)
                            value =  self.look_up.get((j, tuple(patch.reshape(-1))))

                            if value == None:
                                value =  self.to_quanvolute_patch(self.circuits[j], patch, encoding=self.encoding) 
                                self.look_up[(j, tuple(patch.reshape(-1)))] = value
                                #print(self.look_up)
                                self.counter1 += 1
                                print("New patch, total: "+str(self.counter1))

                            else:
                                self.counter2 += 1
                                #print("Patch already in lookup. Duplicated patches found: "+str(self.counter2))

                            output[i, j, h, w] = value #questo funziona!!!
                        else:
                            raise Exception("Errore: self.encoding != THRESHOLD")


            image_end_time = time.time()
            image_time = image_end_time - image_start_time
            elapsed_time = image_end_time - start_time
            average_time_per_image = elapsed_time / (i + 1)
            estimated_remaining_time = average_time_per_image * (batch_size - i - 1)

            if i%10==0: #boh qualche prob con self.verbose 
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

        #if self.look_up == None:
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
                    theta = patch[row][col]*3.1415*2
                    print("!!! controlla rotational enc!!!")
                    emb.rx(theta, i)
            

            combined_circuit = combined_circuit = emb.compose(circuit)
            combined_circuit.measure_all()
            
            result = execute(combined_circuit, self.simulator, shots=self.n_shots).result()
            counts = result.get_counts(combined_circuit)

            sum_1s = sum(key.count('1') * count for key, count in counts.items()) / self.n_shots / (n*n)

                #print(sum_1s)
            return sum_1s
    
    def generate_look_up_table(self):

        raise Exception("Deprecated! Adaptive Look-up table implemented.")

        print("Currently working only for 3x3.")
        print("TO DO: adaptive look-up generation instead")
        self.look_up = {}
        combinations = list(product([0, 1], repeat=9))

        # Reshape each combination into a 3x3 matrix
        combinations = [np.array(combination).reshape(3, 3) for combination in combinations]

        start_time = time.time()

        for j in range(len(self.circuits)):
            print("Generating look-up table for circuit "+str(j+1)+"...")
            for i, combination in enumerate(combinations):
                # Replace this with your actual computation logic
                value =  self.to_quanvolute_patch(self.circuits[j], combination, encoding=self.encoding) 
                self.look_up[(j, tuple(combination.reshape(-1)))] = value

        end_time = time.time()
        print(self.look_up)
        print("Look up table generated.")
        print("Time required: "+str(end_time-start_time))
        

def discretize(value, levels):
    #return round(value * (levels)) / (levels)
    return math.floor(value*levels*0.9999) / (levels-1)

def discretize_patch(patch, levels):
    for ii in range(3*3):
        row = ii // 3
        col = ii % 3
        patch[row][col] = discretize(patch[row][col], levels)
    return patch