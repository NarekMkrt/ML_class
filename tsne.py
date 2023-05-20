from sklearn.metrics import pairwise_distances
import numpy as np


class TSne:

    def __init__(self, n_components=2, sigma=30, learning_rate=200,
                 n_iter=1000):

        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.sigma = sigma

    def fit_transform(self, X):
        # Setting P matrix of higher dimension
        dissimilarity_matrix = pairwise_distances(X, metric='euclidean')
        n = dissimilarity_matrix.shape[0]
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                P[i][j] = P[j][i] = np.exp(-dissimilarity_matrix[i][j] ** 2 / (2 * 0.5 ** 2))
        P = (P / np.sum(P, axis=1)).T

        # Initializing solution
        solution = np.random.normal(loc=0, scale=np.sqrt(1e-10), size=(n, self.n_components))

        # Computing low dimensional simmmilarity matrix
        for _ in range(self.n_iter):
            dissimilarity_matrix_low = pairwise_distances(solution, metric='euclidean')
            q = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    q[i][j] = q[j][i] = 1 / (1 + dissimilarity_matrix_low[i][j] ** 2)
            q = (q / np.sum(q, axis=1)).T

            # Calculating gradient
            gradient = np.zeros((n, self.n_components))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        gradient[i] += 4 * (P[i][j] - q[i][j]) * (solution[i] - solution[j]) * (
                                    (1 + dissimilarity_matrix_low[i][j] ** 2) ** -1)

            # Updating solution
            solution += self.learning_rate * gradient

        return solution
