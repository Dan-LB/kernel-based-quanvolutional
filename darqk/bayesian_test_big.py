#import copy
import os
#import cirq
import numpy as np

np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(suppress=True)
from darqk.core import Ansatz, Kernel, KernelFactory, KernelType
from darqk.optimizer import ReinforcementLearningOptimizer, GreedyOptimizer, MetaheuristicOptimizer, BayesianOptimizer
from darqk.evaluator import RidgeGeneralizationEvaluator, CenteredKernelAlignmentEvaluator, KernelAlignmentEvaluator

from graph_generator import generate_artificial_data

import random
random.seed(12345)
np.random.seed(12345)
#import itertools

#1 1 2 4

gens = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']


def is_not_constant(vector):
    first_element = vector[0]
    for element in vector:
        if abs(element - first_element) > 0.00001:
            return True
    print("The vector is constant.")
    return False

def quantum_process(x, kernel):
    return kernel.phi(x)

def generate_data(n_features, n_operations, n_qubits, n_samples, time_out = 10, measure_type = "X", do_not_check = False):
    assert n_qubits >= 2
    done = False
    tries = 0
    while (not done):
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
        
        #!!!!
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
        
            
        real_kernel = KernelFactory.create_kernel(ansatz, measure, KernelType.OBSERVABLE)
        print(real_kernel.ansatz)

        X = np.random.uniform(-np.pi, np.pi, size=(n_samples, n_features))
        y = np.array([quantum_process(x, real_kernel) for x in X])

        if is_not_constant(y) or do_not_check:
            done = True
        else:
            tries += 1
            if(tries>time_out):
                print("Sono stanco...")
                return None
    print(real_kernel)
    return X, y, real_kernel


ke = KernelAlignmentEvaluator()

#ke2 = KernelAlignmentEvaluator()

n_features = 3
n_operations = 4
n_qubits = 3

X, y = generate_artificial_data(N=n_qubits, E=n_features)

#X, y, real_kernel = generate_data(n_features=n_features, n_operations=n_operations, n_qubits=n_qubits,
#                                  n_samples=10, time_out = 50000,
#                                  measure_type="XZ")


_, _, new_kernel = generate_data(n_features=n_features, n_operations=n_operations, n_qubits=n_qubits,
                                  n_samples=10, time_out = 50000, measure_type="X",
                                  do_not_check=True)

print(new_kernel)




def testing1(init, measure, n_features, n_operations, n_qubits, n_epochs, n_points):
    random.seed(12345)
    np.random.seed(12345)

    output_text = f"initialization = {init}, measure = {measure}, n_features = {n_features}, n_operations = {n_operations}, n_qubits = {n_qubits}, n_epochs = {n_epochs}, n_points = {n_points}"



    X, y = generate_artificial_data(N=n_qubits, E=n_features)
    ansatz = Ansatz(n_features=n_features, n_qubits=n_qubits, n_operations=n_operations)

    _, _, new_kernel = generate_data(n_features=n_features, n_operations=n_operations, n_qubits=n_qubits,
                                n_samples=10, time_out = 50000, measure_type="X",
                                do_not_check=True)
    if init == "IDENTITY":
        new_kernel.ansatz.initialize_to_identity()

    if measure == "RANDOM":
        letters = ['X', 'Y', 'Z']   # replace with the desired set of letters
        measure = ''.join(random.choice(letters) for _ in range(n_qubits))
        new_kernel.measurement = measure


    opt_baye = BayesianOptimizer(new_kernel, X, y, ke)
    RL_kernel = opt_baye.optimize(n_epochs=n_epochs, n_points=n_points, n_jobs=1)

    output_text +=  f", final_ansatz = {str(RL_kernel.ansatz)}, final_measurement = {str(RL_kernel.measurement)}, score = {ke.evaluate(RL_kernel, None, X, y)}\n"
    #print(RL_kernel)
    #print("", ke.evaluate(RL_kernel, None, X, y))

    


    return output_text





if not os.path.exists("results"):
    os.mkdir("results")

output = "I am going to print some output.\n Speriamo bene!\n\n"

q_list = [3, 4, 5, 8, 10]
for q in q_list:
    if q == 3:
        n_features = 3
    else:
        n_features = 4

    for N in [10, 20]:

        for init in ["RANDOM", "IDENTITY"]:
            for measure in ["RANDOM", "X"]:
                output += testing1(init=init, measure=measure, n_features=n_features, n_operations=4, n_qubits=q, n_epochs=N, n_points=N)

# Create a new file called output.txt in the results folder
with open("results/bayesian_big.txt", "w") as f:
    f.write(output)

