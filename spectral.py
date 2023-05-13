import numpy as np
from sklearn.cluster import KMeans

class SpectralClustering():

    def __init__(self, affinity='rbf', n_clusters=8, sigma=1.0):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.affinity = affinity

    def fit(self, X):
        n = X.shape[0]
        W = np.zeros((n, n))
        if self.affinity == 'rbf':
            for i in range(n):
                for j in range(i + 1, n):
                    W[i][j] = np.exp(-(np.linalg.norm(X[i] - X[j])) ** 2 / (self.sigma ** 2))
                    W[j][i] = W[i][j]
        D = np.zeros((W.shape[0], W.shape[0]))
        D_inv_sqrt = np.zeros((W.shape[0], W.shape[0]))
        for i in range(W.shape[0]):
            D[i][i] = np.sum(W[i])
            D_inv_sqrt[i][i] = np.sqrt(1 / np.sum(W[i]))
        L = D - W
        norm_lap = np.matmul(D_inv_sqrt, np.matmul(L, D_inv_sqrt))

        return norm_lap

    def predict(self, X):
        lap = self.fit(X)
        eigvalues, eigvectors = np.linalg.eigh(lap)
        indx = [i for i in range(self.n_clusters)]
        cl_eigs = eigvectors[:, indx]
        print(cl_eigs)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(cl_eigs)

        return kmeans.labels_

