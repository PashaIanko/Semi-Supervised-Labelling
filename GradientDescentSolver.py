from GradientSolver import GradientSolver
from DataProperties import DataProperties
import numpy as np

class GradientDescentSolver(GradientSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def solve(self, X, Y, iter_limit, delta_loss_limit, stop_loss):

        labeled_idxs = np.where(Y != DataProperties.unlabeled)[0]
        unlabeled_idxs = np.where(Y == DataProperties.unlabeled)[0]

        Y_res = np.ndarray.copy(Y)
        
        # fix initial approximation
        Y_res[Y_res == DataProperties.unlabeled] = 0.5

        loss_prev = 0.0
        self.losses = []
        self.n_iterations = 0

        assert(self.lr_strategy == 'lr_constant')  # optimization, now work with lr==const
        learning_rate = self.get_learning_rate()
        
        for i in range(iter_limit):
            loss = self.compute_loss(X, Y_res, labeled_idxs, unlabeled_idxs)
            self.losses.append(loss)
            delta_loss = abs(loss - loss_prev)
            print(f'Loss: {loss}, delta loss: {delta_loss}')

            if not ((delta_loss < delta_loss_limit) or (loss < stop_loss)):
                updates = -learning_rate * self.compute_grad(X, Y_res, labeled_idxs, unlabeled_idxs)
                for i in range(len(updates)): Y_res[unlabeled_idxs[i]] += updates[i] 
                
                loss_prev = loss
                self.n_iterations += 1

        # return Y_res
        print('\n')
        return self.threshold_proc(Y_res)