from GradientSolver import GradientSolver
from DataProperties import DataProperties
import numpy as np
from timeit import default_timer

class GradientDescentSolver(GradientSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def solve(self, X, Y, iter_limit, delta_loss_limit, stop_loss, weight_matrix = None):

        labeled_idxs = np.where(Y != DataProperties.unlabeled)[0]
        unlabeled_idxs = np.where(Y == DataProperties.unlabeled)[0]

        Y_res = np.ndarray.copy(Y)

        
        # fix initial approximation
        Y_res[Y_res == DataProperties.unlabeled] = 0.5

        loss_prev = 0.0
        self.losses = []
        self.cpu_times = []
        self.n_iterations = 0

        if weight_matrix is None:
            self.calc_weight_matrix(X)
        else:
            self.weight_matrix = weight_matrix  # to keep time safe, not recalculating in every solver

        assert(self.lr_strategy == 'lr_constant')  # optimization, now work with lr==const
        learning_rate = self.get_learning_rate()
        
        algo_start = default_timer()
        
        for i in range(iter_limit):
            loss = self.compute_loss(X, Y_res, labeled_idxs, unlabeled_idxs)
            self.losses.append(loss)
            self.cpu_times.append(default_timer() - algo_start)

            delta_loss = abs(loss - loss_prev)
            print(f'Iteration: {i}, Loss: {loss}, delta loss: {delta_loss}')

            if loss < stop_loss:
                break

            if not (delta_loss < delta_loss_limit):
                updates = -learning_rate * self.compute_grad(X, Y_res, labeled_idxs, unlabeled_idxs)
                for i in range(len(updates)): Y_res[unlabeled_idxs[i]] += updates[i] 
                
                loss_prev = loss
                self.n_iterations += 1

        # return Y_res
        print('\n')
        return self.threshold_proc(Y_res)