import numpy as np
from ..core import Kernel
from . import KernelEvaluator

from darqk.core import Ansatz, Kernel, KernelFactory, KernelType

from sklearn.metrics.pairwise import cosine_similarity

class CustomEvaluator(KernelEvaluator):
    """
    Kernel compatibility measure based on the kernel-target alignment
    See: Cristianini, Nello, et al. "On kernel-target alignment." Advances in neural information processing systems 14 (2001).
    """
    def __init__(self, i, values, patches):

        self.i = i
        self.values = values 

        self.patches = patches


    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
        n_patches = len(self.patches)
        for i in range(n_patches):
            current_patch = self.patches[i].numpy()
            self.values[self.i, i] =  self.to_quanvolute_patch(kernel, current_patch)


        similarities = cosine_similarity(self.values)
        average_similarity = np.mean(similarities[self.i])
        
        print(similarities)

        #print(average_similarity)
        # assert not np.isnan(the_cost), f"{kernel=} {K=} {y=}"
        return abs(average_similarity-0.5) 
    
    def to_quanvolute_patch(self, circuit, patch):
        output = circuit.phi(patch.ravel())
        return output


