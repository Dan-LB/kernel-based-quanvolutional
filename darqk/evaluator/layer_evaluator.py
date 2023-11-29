import numpy as np
from ..core import Kernel
from . import KernelEvaluator

from darqk.core import Ansatz, Kernel, KernelFactory, KernelType


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from CustomDataset import CustomDataset, load_custom_dataset

import constants



class LayerEvaluator(KernelEvaluator):
    """
    INGREDIENTI:
        TRAINING DATASET X
        UN LAYER (CONVERTITO IN KERNEL)

        UN MODELLO (O ARCHITETTURA?)

        FUNZIONE PER COSTRUIRE UN KERNEL GROSSO DA N PICCOLI
        FUNZIONE PER SPEZZARE UN KERNEL GROSSO IN N PICCOLI

        FUNZIONE PER SPEZZARE X IN X_TRAIN E X_TEST
        FUNZIONE PER TRAINARE UN MODELLO SU X_TRAIN E TESTARLO SU X_TEST (SENZA PRE PROCESSING)

    PROCEDURA DI OTTIMIZZAZIONE LAYER:
        PRENDO LAYER L
        COSTRUISCO KERNEL K DA L
        VALUTO K
        OTTIMIZZO K
        RESTITUISCO L

    PROCEDURA DI VALUTAZIONE K:
        SPEZZO K IN L
        COSTRUISCO MODELLO M DA L
        TRAINING E TESTING DI M
        USO ACCURACY
    """
    
    def __init__(self, X_and_y, L, copy_of_model):

        #parto da X e Y come coppia dei file np con data e label

        self.layer = L 
        #self.model_structure = model_structure

        #L is a list of kernels

        self.in_channels = L.in_channels
        assert self.in_channels == 1, "So far we really want to work with just a single layer :)"

        self.circuits_amount = L.out_channels #this is the numer of circuits we want to divide the big circuit to
        self.kernel_size = L.kernel_size #e.g. 3 for a 3x3
        self.stride = L.stride #this should be 1, however not really relevant
        self.n_qubits = L.n_qubits

        self.single_length = L.L

        #this is the big circuit, we can initialize it to random without any problem (I guess?)
        #actually, it is a kernel!

        self.model = copy_of_model
        
        print("Sanity check: (from layer_evaluator, init)")
        print(self.model)


    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray): #not sure if this stuff is required
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """

        model = self.model
        circuits_list  = self.obtain_layer_from_big_K(kernel)
        model.quanv.circuits = circuits_list

        print("Getting data...")
        X, y = get_data(n=100, size=10) #questa cosa non ha senso, meglio salvare questi dati da qualche parte
        model.verbose = False #redundacy
        model.quanv.verbose = False
        print("Preprocessing dataset...")
        q_X = model.preprocess_dataset(X)

        model.on_preprocessed = True

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        train_loader, test_loader = load_custom_dataset(batch_size=64, npy_file=q_X.numpy(), labels_file=y.numpy())

        #q_X = model.preprocess_dataset(X)
        #y = y


        for epoch in range(1, 100):  # 100 epochs
            train(model, device, train_loader, optimizer, epoch)
            loss, correct = test(model, device, test_loader)

        accuracy = correct
        print("Accuracy of last evaluation:" +str(accuracy))
        return -accuracy

    def split_to_test_and_train(self, size = 50): #questo non si usa
        """
        """
        train_loader, test_loader = load_custom_dataset(batch_size=64, npy_file=self.X[:size], labels_file=self.y[:size])
        return train_loader, test_loader

    def obtain_layer_from_big_K(self, K):
        new_layer = []
        for i in range(0, len(K.ansatz.operation_list), self.single_length):
            # Extract a subset of operations for a smaller kernel
            subset_operations = K.ansatz.operation_list[i:i + self.single_length]

            # Create a new ansatz for the smaller kernel with appropriate parameters
            new_ansatz = Ansatz(self.kernel_size**2, self.n_qubits, self.single_length) # Add other necessary parameters

            # Assign the subset of operations to the new ansatz
            new_ansatz.operation_list = subset_operations

            # Create a new kernel with the new ansatz
            new_kernel = KernelFactory.create_kernel(new_ansatz, "Z"*self.n_qubits, KernelType.OBSERVABLE)
            
            # Add the new kernel to the layer
            new_layer.append(new_kernel)

            # Break the loop if the desired number of kernels (self.out_channels) is reached
            if len(new_layer) == self.circuits_amount:
                return new_layer



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % 100 == 0:
        #    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
    #print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss, 100. * correct / len(test_loader.dataset)


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