import os
import numpy as np

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from PIL import Image

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info import Statevector

from darqk.core import Ansatz

import matplotlib.pyplot as plt

import random
import math
#random.seed(12345)
#np.random.seed(12345)

THRESHOLD = "THRESHOLD"
ROTATIONAL = "ROTATIONAL"

n_features = 4
n_operations = 18 
n_qubits = 4

#very far from working...
def generate_PQC(n_features, n_operations, n_qubits, measure_type = "X"):
    gens = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
    assert n_qubits >= 2

    ansatz = Ansatz(n_features=n_features, n_qubits=n_qubits, n_operations=n_operations)
    ansatz.initialize_to_identity()
    for i in range(n_operations):
        w1 = random.randint(0, n_qubits-1)
        w2 = w1
        while(w2 == w1):
            w2 = random.randint(0, n_qubits-1)
        gen_index = random.randint(0, len(gens)-1)
        bw = random.random()
        bw = 1
        feat = random.randint(0, n_features)

        ansatz.change_operation(i, new_feature=feat, new_wires=[w1, w2], new_generator=gens[gen_index], new_bandwidth=bw)
    
    #not yet working
    if measure_type == "X":
        measure = "X" * n_qubits
    if measure_type == "random":
        letters = ['X', 'Y', 'Z']   # replace with the desired set of letters
        measure = ''.join(random.choice(letters) for _ in range(n_qubits))
    if measure_type == "XZ":
        measure = ""
        for i in range(n_qubits):
            if i % 2 == 0:
                measure += 'X'
            else:
                measure += 'Z'

    print(ansatz)

    return ansatz

def generate_random_quantum_circuit(filter_size = 3):
    """
    Generate... according to [quanvolutional]
    3x3 on 9 qubits
    """
    
    n_qubits = filter_size*filter_size
    connection_prob = 0.15
    one_qb_list = ["X", "Y", "Z",  "P", "T", "H"]
    two_qb_list = ["Cnot", "Swap", "SqrtSwap"] #check on SqrtSwap
    gate_list = []

    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j and random.random() < connection_prob:
                g_index = random.randint(0, len(two_qb_list)-1)
                gate_list.append({"gate": two_qb_list[g_index], "first_q": i, "second_q": j})

    n_one_qg =   random.randint(0, 2*3*3)   
    for i in range(n_one_qg):
        q = random.randint(0, n_qubits-1) 
        g_index = random.randint(0, len(one_qb_list)-1)  
        gate_list.append({"gate": one_qb_list[g_index], "first_q": q})

    random.shuffle(gate_list)

    circuit = QuantumCircuit(n_qubits)
    for gate in gate_list:
        theta = random.random()*3.14
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



    circuit_image = circuit_drawer(circuit, output='mpl', style="iqp")
    circuit_image.savefig('quantum_circuit.png')
    return circuit

def to_quanvolute_patch(circuit, patch, encoding, n_shots):
    #print(circuit.num_qubits)
    #print(len(patch))
    n = len(patch)
    
    if encoding == THRESHOLD: #IN 0,1
        emb = QuantumCircuit(n*n)
        for i in range(n*n):
            row = i // n
            col = i % n
            if patch[row][col] >= 0.5:
                emb.x(i)

    if encoding == ROTATIONAL:
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

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(combined_circuit, simulator, shots=n_shots).result()
    counts = result.get_counts(combined_circuit)
    #print(counts)
    #plot_histogram(counts)

    #plot_histogram(counts).savefig('histogram.png')
    sum_1s = sum(key.count('1') * count for key, count in counts.items()) / n_shots / (n*n)

        #print(sum_1s)

    return sum_1s

def to_quanvolute(circuit, image, encoding = THRESHOLD, n_shots = 100):
    filter_size = int(math.sqrt(circuit.num_qubits))
    image_size = len(image)
    output_size = image_size - filter_size + 1
    result = [[0] * output_size for _ in range(output_size)]

    for i in range(output_size):
        for j in range(output_size):
            # Extract the patch from the image
            patch = [row[j:j+filter_size] for row in image[i:i+filter_size]]

            # Apply convolution to the patch and store the result in the output feature map
            result[i][j] = to_quanvolute_patch(circuit, patch, encoding=encoding, n_shots=n_shots)

    return result


def load_mnist(batch_size, download=True, train=True):

    """Load MNIST data using PyTorch datasets."""
    n = 28
    transform = transforms.Compose([
        transforms.Resize((n, n)),
        transforms.ToTensor() # Convert images to PyTorch tensors
    ])

    dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    # Get the first image from the dataset
    first_image, _ = next(iter(dataloader))
    first_image = first_image[0]  # Take the first image from the batch
    
    # Convert the tensor to a NumPy array and reshape
    first_image_numpy = first_image.numpy()
    
    # You can specify the desired n x n dimensions here
    #  # Assuming MNIST images are 28x28
    first_image_numpy = first_image_numpy.reshape(n, n)
    return dataloader, first_image_numpy

def save_image(filename, image):
    # Create an image object from the 2D matrix

    img = Image.new('L', (len(image[0]), len(image)))
    # Convert the 2D matrix to a flattened list of pixel values
    pixel_values = [pixel*255 for row in image for pixel in row]

    # Set the pixel data for the image
    img.putdata(pixel_values)

    # Save the image to a file
    img.save(filename)



#generate_PQC(n_features, n_operations, n_qubits)

random_matrix = np.random.rand(10, 10)
print(random_matrix)

_, random_matrix = load_mnist(batch_size=100)


#to_quanvolute_patch(random_circuit, random_matrix, encoding = THRESHOLD, n_shots = 100)

save_image("basic_image.png", random_matrix)
for i in range(5):
    random_circuit  = generate_random_quantum_circuit(3)
    new_image = to_quanvolute(random_circuit, random_matrix, encoding = ROTATIONAL, n_shots = 1000)
    save_image("convoluted_image"+str(i)+".png", new_image)



