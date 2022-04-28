from numpy.linalg import norm
from numpy import sum
from numpy import zeros
from numpy import array

class GradientSolver:
    def __init__(
        self, 
        lr_strategy, 
        learning_rate, 
        similarity_func, 
        bcgd_strategy = None
    ):

        available_lr_strategies = [
            'lr_constant'
        ]
        assert lr_strategy in available_lr_strategies, print(f'Pick available lr_strategy: {available_lr_strategies}')

        self.lr_strategy = lr_strategy
        self.learning_rate = learning_rate
        self.similarity_func = similarity_func

        self.losses = []
        self.cpu_times = []
        self.weights_matrix = None
        self.n_iterations = 0

    
    def calc_weight_matrix(self, X):
        self.weight_matrix = zeros((X.shape[0], X.shape[0]))
    
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.weight_matrix[i][j] = 1 / (norm(X[i] - X[j]) + 0.001)
                


    def get_learning_rate(self):
        if self.lr_strategy == 'lr_constant':
            return self.learning_rate   

    def compute_grad_component(self, X, Y, labeled_idxs, unlabeled_idxs, idx):
        return sum([2 * self.calc_weight(idx, labeled_idx) * (Y[idx] - Y[labeled_idx]) for labeled_idx in labeled_idxs]) + sum([2 * self.calc_weight(idx, another_unlab_idx) * (Y[idx] - Y[another_unlab_idx]) for another_unlab_idx in unlabeled_idxs])
        # grad_component = 0.0
        # comp_1 = np.sum([2 * self.calc_weight(X[idx], X[labeled_idx]) * (Y[idx] - Y[labeled_idx]) for labeled_idx in labeled_idxs])    
        # comp_2 = np.sum([2 * self.calc_weight(X[idx], X[another_unlab_idx]) * (Y[idx] - Y[another_unlab_idx]) for another_unlab_idx in unlabeled_idxs])
        # return comp_1 + comp_2  # Optimization code
        

    def calc_weight(self, i, j):
        return self.weight_matrix[i][j] # 1 / (norm(Xi - Xj) + 0.001)

    def compute_grad(self, X, Y, labeled_idxs, unlabeled_idxs):
        return array([self.compute_grad_component(X, Y, labeled_idxs, unlabeled_idxs, idx=unlabeled_idx) for unlabeled_idx in unlabeled_idxs])

        # grad = []
        # for unlabeled_idx in unlabeled_idxs:
        #     # compute component of gradient for this 
        #     # current unlabeled
        #     grad_component = self.compute_grad_component(X, Y, labeled_idxs, unlabeled_idxs, idx=unlabeled_idx)
        #     grad.append(grad_component)

        # return np.array(grad)

    def compute_loss(self, X, Y, labeled_idxs, unlabeled_idxs):
        # Optimized code:
        return sum([self.calc_weight(labeled_idx, unlab_idx) * ((Y[labeled_idx] - Y[unlab_idx]) ** 2) for labeled_idx in labeled_idxs for unlab_idx in unlabeled_idxs]) + 0.5 * sum([self.calc_weight(unlab_idx, another_unlab_idx) * ((Y[unlab_idx] - Y[another_unlab_idx]) ** 2) for unlab_idx in unlabeled_idxs for another_unlab_idx in unlabeled_idxs])
        
        # comp_1 = sum([self.calc_weight(X[labeled_idx], X[unlab_idx]) * ((Y[labeled_idx] - Y[unlab_idx]) ** 2) for labeled_idx in labeled_idxs for unlab_idx in unlabeled_idxs]) 
        # comp_2 = sum([0.5 * self.calc_weight(X[unlab_idx], X[another_unlab_idx]) * ((Y[unlab_idx] - Y[another_unlab_idx]) ** 2) for unlab_idx in unlabeled_idxs for another_unlab_idx in unlabeled_idxs])
        # return comp_1 + comp_2

        # res = 0
        
        # for labeled_idx in labeled_idxs:
        #     for unlab_idx in unlabeled_idxs:
        #         res += self.calc_weight(X[labeled_idx], X[unlab_idx]) * ((Y[labeled_idx] - Y[unlab_idx]) ** 2)

        # for unlab_idx in unlabeled_idxs:
        #     for another_unlab_idx in unlabeled_idxs:
        #         res += 0.5 * self.calc_weight(X[unlab_idx], X[another_unlab_idx]) * ((Y[unlab_idx] - Y[another_unlab_idx]) ** 2)
        
        # return res
    
    def threshold_proc(self, Y_preds):
        Y_preds[Y_preds > 0.5] = 1.0
        Y_preds[Y_preds <= 0.5] = 0.0
        return Y_preds