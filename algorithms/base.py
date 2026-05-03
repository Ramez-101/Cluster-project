from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist


class BaseClusterOptimizer(ABC):
    def __init__(self, n_clusters=4, max_iter=200, pop_size=30, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.labels_ = None
        self.centroids_ = None
        self.convergence_history_ = []
        self.best_wcss_ = None

    @abstractmethod
    def fit(self, X):
        """
        Returns
        -------
        labels            : np.ndarray (n_samples,)
        centroids         : np.ndarray (n_clusters, n_features)
        convergence_history : list of float WCSS, length == max_iter
        """

    def _wcss(self, X, centroids):
        dists = cdist(X, centroids, 'sqeuclidean')
        return float(np.sum(np.min(dists, axis=1)))

    def _assign_labels(self, X, centroids):
        dists = cdist(X, centroids, 'sqeuclidean')
        return np.argmin(dists, axis=1)

    def _decode(self, solution, n_features):
        return solution.reshape(self.n_clusters, n_features)

    def _encode(self, centroids):
        return centroids.flatten()

    def _init_population(self, X, size):
        n_features = X.shape[1]
        pop = np.zeros((size, self.n_clusters * n_features))
        for i in range(size):
            idxs = self.rng.choice(len(X), size=self.n_clusters, replace=False)
            pop[i] = self._encode(X[idxs])
        return pop

    def _refine_centroids(self, X, centroids):
        """One Lloyd's step; reinitialises empty clusters to random data points."""
        labels = self._assign_labels(X, centroids)
        new_c = np.copy(centroids)
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() == 0:
                new_c[k] = X[self.rng.integers(len(X))]
            else:
                new_c[k] = X[mask].mean(axis=0)
        return new_c
