import numpy as np
from .base import BaseClusterOptimizer


class ParticleSwarmOptimization(BaseClusterOptimizer):
    def __init__(self, w_init=0.9, w_end=0.4, c1=2.0, c2=2.0, **kwargs):
        super().__init__(**kwargs)
        self.w_init = w_init
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

    def fit(self, X):
        n_samples, n_features = X.shape

        padding = 0.1 * (X.max(axis=0) - X.min(axis=0))
        lb = np.tile(X.min(axis=0) - padding, self.n_clusters)
        ub = np.tile(X.max(axis=0) + padding, self.n_clusters)
        v_max = 0.2 * (ub - lb)

        positions = self._init_population(X, self.pop_size)
        velocities = self.rng.uniform(-v_max, v_max, positions.shape)

        pbest = positions.copy()
        pbest_fitness = np.array([self._wcss(X, self._decode(p, n_features)) for p in pbest])

        gbest_idx = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        convergence_history = []

        for it in range(self.max_iter):
            w = self.w_init - (self.w_init - self.w_end) * (it / max(1, self.max_iter - 1))

            r1 = self.rng.random(positions.shape)
            r2 = self.rng.random(positions.shape)
            velocities = (w * velocities
                          + self.c1 * r1 * (pbest - positions)
                          + self.c2 * r2 * (gbest - positions))
            velocities = np.clip(velocities, -v_max, v_max)
            positions = np.clip(positions + velocities, lb, ub)

            fitness = np.array([self._wcss(X, self._decode(p, n_features)) for p in positions])

            improve = fitness < pbest_fitness
            pbest[improve] = positions[improve].copy()
            pbest_fitness[improve] = fitness[improve]

            best_i = int(np.argmin(pbest_fitness))
            if pbest_fitness[best_i] < gbest_fitness:
                gbest = pbest[best_i].copy()
                gbest_fitness = pbest_fitness[best_i]

            convergence_history.append(gbest_fitness)

        centroids = self._decode(gbest, n_features)
        labels = self._assign_labels(X, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = gbest_fitness
        return labels, centroids, convergence_history
