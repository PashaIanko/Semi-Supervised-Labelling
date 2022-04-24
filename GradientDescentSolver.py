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
        for i in range(iter_limit):
            loss = self.compute_loss(X, Y_res, labeled_idxs, unlabeled_idxs)
            self.losses.append(loss)
            delta_loss = abs(loss - loss_prev)
            print(f'Loss: {loss}, delta loss: {delta_loss}')


            if ((i > 0 and delta_loss < delta_loss_limit) or (loss < stop_loss)):
                break
            else:
                grad = self.compute_grad(X, Y_res, labeled_idxs, unlabeled_idxs)

                
                
                learning_rate = self.get_learning_rate()
                updates = -learning_rate * grad


                assert len(unlabeled_idxs) == len(grad)
                for i in range(len(grad)):
                    Y_res[unlabeled_idxs[i]] += updates[i]
                
                loss_prev = loss
                self.n_iterations += 1

        # return Y_res
        print('\n')
        return self.threshold_proc(Y_res)