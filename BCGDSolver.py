from GradientSolver import GradientSolver
from DataProperties import DataProperties
import numpy as np
import random

class BCGDSolver(GradientSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        available_strategies = [
            # 'cyclic_order',
            'random permutation',
            'random sampling',
            'cyclic'
        ]

        block_strategy = kwargs['bcgd_strategy']
        assert block_strategy in available_strategies, \
            print(f'Please choose available block strategy: {available_strategies}')
        
        self.block_strategy = block_strategy
        # strategies - random permutation & random sampling

    def pick_block_indices(self, unlabeled_indices):
        
        if self.block_strategy == 'random permutation':
            # we just permute the dimensions
            return np.random.permutation(unlabeled_indices)
        if self.block_strategy == 'random sampling':
            return [random.choice(unlabeled_indices)]
        if self.block_strategy == 'cyclic':
            return unlabeled_indices


    def solve(self, X, Y, iter_limit, delta_loss_limit):

        Y_res = np.ndarray.copy(Y)
        labeled_indices = np.where(Y_res != DataProperties.unlabeled)[0]
        unlabeled_indices = np.where(Y_res == DataProperties.unlabeled)[0]
        assert len(labeled_indices) + len(unlabeled_indices) == len(Y_res)

        # Step 1. Choose initial point
        Y_res[unlabeled_indices] = 0.5

        loss_prev = 0
        self.losses = []
        self.n_iterations = 0
        for i in range(iter_limit):

            loss = self.compute_loss(X, Y_res, labeled_indices, unlabeled_indices)
            self.losses.append(loss)
            delta_loss = abs(loss - loss_prev)
            loss_prev = loss
            print(f'LOSS: {loss}, Delta: {delta_loss}')

            # Specific condition
            if (i > 0) and delta_loss <= delta_loss_limit:
                break
            else:
                y = np.ndarray.copy(Y_res)  # y_0
                del Y_res

                # depending on the strategy,
                # S contains one random block, permutation
                # of available coordinates, or same sequence of
                # coordinates (for cyclic approach)
                S = self.pick_block_indices(unlabeled_indices)  # Pick random permutation of unlabeled indices
                
                # print(f'Picked: {S}')
                
                # l = len(S)
                # And now we move across S and update y variable
                for index in S:
                    assert(index in unlabeled_indices)
                    learning_rate = self.get_learning_rate()
                    
                    grad_component = self.compute_grad_component(
                        X,
                        y,
                        labeled_indices,
                        unlabeled_indices,
                        idx = index
                    )
                    grad_vector = np.zeros(y.shape[0])
                    grad_vector[index] = grad_component
                    y = y - learning_rate * grad_vector
                
                Y_res = y
                del y
                self.n_iterations += 1
        return self.threshold_proc(Y_res)