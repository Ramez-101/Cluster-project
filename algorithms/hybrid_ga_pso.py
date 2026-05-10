import numpy as np
from .base import BaseClusterOptimizer


class HybridGAPSO(BaseClusterOptimizer):
    """
    Cooperative hybrid: every iteration applies PSO updates to all particles,
    then GA operators (crossover + mutation) replace the worst 50%.
    """

    def __init__(self, w_init=0.9, w_end=0.4, c1=1.5, c2=1.5,
                 crossover_rate=0.8, mutation_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.w_init = w_init
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _tournament_select(self, pop, fitness, t=3):
        idxs = self.rng.choice(len(pop), size=t, replace=False)
        return pop[idxs[int(np.argmin(fitness[idxs]))]].copy()

    def _blx_crossover(self, p1, p2):
        alpha = 0.5
        lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
        hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
        return self.rng.uniform(lo, hi)

    def fit(self, X):
        n_samples, n_features = X.shape
        dim = self.n_clusters * n_features

        padding = 0.1 * (X.max(axis=0) - X.min(axis=0))
        lb = np.tile(X.min(axis=0) - padding, self.n_clusters)
        ub = np.tile(X.max(axis=0) + padding, self.n_clusters)
        v_max = 0.2 * (ub - lb)

        positions = self._init_population(X, self.pop_size)
        velocities = self.rng.uniform(-v_max, v_max, positions.shape)
        fitness = self._batch_wcss(X, positions.reshape(self.pop_size, self.n_clusters, n_features))

        pbest = positions.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        convergence_history = []

        for it in range(self.max_iter):
            w = self.w_init - (self.w_init - self.w_end) * (it / max(1, self.max_iter - 1))

            # PSO phase
            r1 = self.rng.random(positions.shape)
            r2 = self.rng.random(positions.shape)
            velocities = (w * velocities
                          + self.c1 * r1 * (pbest - positions)
                          + self.c2 * r2 * (gbest - positions))
            velocities = np.clip(velocities, -v_max, v_max)
            positions = np.clip(positions + velocities, lb, ub)
            fitness = self._batch_wcss(X, positions.reshape(self.pop_size, self.n_clusters, n_features))

            # GA phase: replace worst 50%
            half = self.pop_size // 2
            worst_idxs = np.argsort(fitness)[half:]

            for idx in worst_idxs:
                p1 = self._tournament_select(positions, fitness)
                p2 = self._tournament_select(positions, fitness)
                child = self._blx_crossover(p1, p2) if self.rng.random() < self.crossover_rate else p1
                mask = self.rng.random(dim) < self.mutation_rate
                if mask.any():
                    child[mask] += self.rng.normal(0, 0.1, mask.sum())
                child = np.clip(child, lb, ub)
                positions[idx] = child
                velocities[idx] = self.rng.uniform(-v_max, v_max)
                fitness[idx] = self._wcss(X, self._decode(child, n_features))

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
