from abc import ABC, abstractmethod
import numpy as np
from utils.gpu import GPU_AVAILABLE, cdist_sqeuclidean

if GPU_AVAILABLE:
    import cupy as cp


class BaseClusterOptimizer(ABC):
    def __init__(self, n_clusters=4, max_iter=200, pop_size=30, random_state=42, use_gpu=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.rng = np.random.default_rng(random_state)
        self.labels_ = None
        self.centroids_ = None
        self.convergence_history_ = []
        self.best_wcss_ = None
        self._X_gpu_cache = None   # (id(X), cupy_array) — refreshed when X changes
        self._X_sq_cache  = None   # precomputed ||x||² for cached X

    @abstractmethod
    def fit(self, X):
        """
        Returns
        -------
        labels            : np.ndarray (n_samples,)
        centroids         : np.ndarray (n_clusters, n_features)
        convergence_history : list of float WCSS, length == max_iter
        """

    def _cdist_sq(self, X, centroids):
        """Squared-Euclidean distance (n×k). Caches X on GPU across repeated calls."""
        if GPU_AVAILABLE and self.use_gpu:
            if self._X_gpu_cache is None or self._X_gpu_cache[0] != id(X):
                self._X_gpu_cache = (id(X), cp.asarray(X, dtype=cp.float32))
            _, X_gpu = self._X_gpu_cache
            C_gpu = cp.asarray(centroids, dtype=cp.float32)
            diff  = X_gpu[:, None, :] - C_gpu[None, :, :]  # (n, k, d)
            return cp.asnumpy(cp.sum(diff ** 2, axis=2))    # (n, k)
        from scipy.spatial.distance import cdist
        return cdist(X, centroids, 'sqeuclidean')

    def _wcss(self, X, centroids):
        dists = self._cdist_sq(X, centroids)
        return float(np.sum(np.min(dists, axis=1)))

    def _batch_wcss(self, X, all_centroids):
        """
        Evaluate WCSS for m centroid sets in one GPU call.

        all_centroids : (m, k, d)
        Returns       : (m,) WCSS values
        """
        from utils.gpu import batch_cdist_sqeuclidean
        if GPU_AVAILABLE and self.use_gpu:
            dists = batch_cdist_sqeuclidean(X, all_centroids)  # (n, m, k)
        else:
            from scipy.spatial.distance import cdist
            m, k = all_centroids.shape[:2]
            dists = np.empty((X.shape[0], m, k), dtype=np.float32)
            for j in range(m):
                dists[:, j, :] = cdist(X, all_centroids[j], 'sqeuclidean')
        return dists.min(axis=2).sum(axis=0)  # (m,)

    def _assign_labels(self, X, centroids):
        dists = self._cdist_sq(X, centroids)
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
