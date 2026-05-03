import numpy as np
from .base import BaseClusterOptimizer


class ExtremalOptimization(BaseClusterOptimizer):
    """Plain EO: always replaces the worst centroid each iteration."""

    def fit(self, X):
        n_samples, n_features = X.shape

        idxs = self.rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = self._refine_centroids(X, X[idxs].copy())

        best_centroids = centroids.copy()
        best_fitness = self._wcss(X, centroids)
        convergence_history = []

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centroids)

            component_costs = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    component_costs[k] = self._wcss(X[mask], centroids[k:k + 1])

            k_replace = int(np.argmax(component_costs))
            centroids[k_replace] = X[self.rng.integers(n_samples)]
            centroids = self._refine_centroids(X, centroids)

            fitness = self._wcss(X, centroids)
            if fitness < best_fitness:
                best_fitness = fitness
                best_centroids = centroids.copy()

            convergence_history.append(best_fitness)

        labels = self._assign_labels(X, best_centroids)
        self.labels_ = labels
        self.centroids_ = best_centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, best_centroids, convergence_history


class TauExtremalOptimization(BaseClusterOptimizer):
    """tau-EO: selects centroid to replace with probability proportional to rank^(-tau)."""

    def __init__(self, tau=1.5, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def fit(self, X):
        n_samples, n_features = X.shape

        idxs = self.rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = self._refine_centroids(X, X[idxs].copy())

        best_centroids = centroids.copy()
        best_fitness = self._wcss(X, centroids)
        convergence_history = []

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centroids)

            component_costs = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    component_costs[k] = self._wcss(X[mask], centroids[k:k + 1])

            # Rank 1 = best (lowest cost); rank n_clusters = worst
            sorted_idx = np.argsort(component_costs)   # ascending
            ranks = np.empty(self.n_clusters, dtype=int)
            ranks[sorted_idx] = np.arange(1, self.n_clusters + 1)

            weights = ranks.astype(float) ** (-self.tau)
            weights /= weights.sum()

            k_replace = int(self.rng.choice(self.n_clusters, p=weights))
            centroids[k_replace] = X[self.rng.integers(n_samples)]
            centroids = self._refine_centroids(X, centroids)

            fitness = self._wcss(X, centroids)
            if fitness < best_fitness:
                best_fitness = fitness
                best_centroids = centroids.copy()

            convergence_history.append(best_fitness)

        labels = self._assign_labels(X, best_centroids)
        self.labels_ = labels
        self.centroids_ = best_centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, best_centroids, convergence_history
