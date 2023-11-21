#import copy
import os
#import cirq
import numpy as np

np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(suppress=True)
from darqk.core import Ansatz, Kernel, KernelFactory, KernelType
from darqk.optimizer import ReinforcementLearningOptimizer, GreedyOptimizer, MetaheuristicOptimizer, BayesianOptimizer
from darqk.evaluator import RidgeGeneralizationEvaluator, CenteredKernelAlignmentEvaluator, KernelAlignmentEvaluator

from darqk.evaluator.one_classSVM_evaluator import OneClassSVMEvaluator

from darqk.evaluator.latent_ad_qml.scripts.kernel_machines.train_callable import train_and_evaluate

from darqk.evaluator.latent_ad_qml.scripts.kernel_machines.my_util import save_kernel, load_kernel, create_model_from_kernel, train_model, test_model, load_trained_model

#from darqk.evaluator.one_classSVM_evaluator import OneClassSVMEvaluator

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


n_samples = 15
#X = data_array[:n_samples]
#Y = data_array[n_samples:n_samples*2]






#sto testando: 8 12 8 con epoch=5, points=5

#ke = KernelAlignmentEvaluator()



#10-120 <- config sul train_callable

#8 6 8 - 5 5 -> 0.83333 

#8 12 8 - 5 5 -> 0.88333 

#8 12 8 - 5 5 - x4 ->  0.76

#8 12 8 - 10 10 -> 0.875

#8 12 4 - 5 5 -> 0.86666

#8 6 4 - 5 5 -> 0.875

#8 6 4 - 5 5 - x2 -> 0.841666

#8 6 4 - 5 5 - x3 -> 0.791666



#60-720 <- config sul train_callable

#8 12 8 - 5 5 -> 0.9208333333333333 #dopo mille ore




p1 = "latentrep_AtoHZ_to_ZZZ_35.h5"
p2 = "latentrep_RSGraviton_WW_BR_15.h5"
p3 = "latentrep_RSGraviton_WW_NA_35.h5"



X, Y = None, None

n_features = 8
n_operations = 18 #12
n_qubits = 4


ansatz = Ansatz(n_features=n_features, n_qubits=n_qubits, n_operations=n_operations)
ansatz.initialize_to_identity()
#ansatz.initialize_to_random_circuit()
kernel = KernelFactory.create_kernel(ansatz, "Z"*n_qubits, KernelType.OBSERVABLE)




ke = OneClassSVMEvaluator(35, 35, 0)
bayesian_opt = BayesianOptimizer(kernel, X, Y, ke)

#optimized_kernel = bayesian_opt.optimize(n_epochs=5, n_points=5, n_jobs=1)

#print(mh_opt_kernel)


#save_kernel(optimized_kernel, "i_am_testing")

#new_kernel = load_kernel("i_am_testing")

#model, args = create_model_from_kernel(new_kernel)

model = load_trained_model("model_L4_T200_P0")

test_model(model, 1500, 0)

#train_model(model, args, 200, 200, 0) #<- meglio 100 e 100



#print(optimized_kernel.to_qiskit_circuit())
#print(ke.evaluate(real_kernel, None, X, Y, save = True))


