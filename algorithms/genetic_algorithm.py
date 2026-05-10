import numpy as np
from .base import BaseClusterOptimizer


class GeneticAlgorithm(BaseClusterOptimizer):
    def __init__(self, crossover_rate=0.8, mutation_rate=0.1, mutation_scale=0.1,
                 tournament_size=3, elitism_count=2, **kwargs):
        super().__init__(**kwargs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

    def _tournament_select(self, pop, fitness):
        idxs = self.rng.choice(len(pop), size=self.tournament_size, replace=False)
        return pop[idxs[int(np.argmin(fitness[idxs]))]].copy()

    def _blx_crossover(self, p1, p2):
        alpha = 0.5
        lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
        hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
        return self.rng.uniform(lo, hi)

    def _mutate(self, ind):
        mask = self.rng.random(len(ind)) < self.mutation_rate
        if mask.any():
            ind[mask] += self.rng.normal(0, self.mutation_scale, mask.sum())
        return ind

    def fit(self, X):
        n_samples, n_features = X.shape

        padding = 0.1 * (X.max(axis=0) - X.min(axis=0))
        lb = np.tile(X.min(axis=0) - padding, self.n_clusters)
        ub = np.tile(X.max(axis=0) + padding, self.n_clusters)

        pop = self._init_population(X, self.pop_size)
        fitness = self._batch_wcss(X, pop.reshape(self.pop_size, self.n_clusters, n_features))

        best_fitness = fitness.min()
        convergence_history = []

        for _ in range(self.max_iter):
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]

            new_pop = [pop[i].copy() for i in range(self.elitism_count)]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitness)
                p2 = self._tournament_select(pop, fitness)
                child = self._blx_crossover(p1, p2) if self.rng.random() < self.crossover_rate else p1
                child = np.clip(self._mutate(child), lb, ub)
                new_pop.append(child)

            pop = np.array(new_pop)
            fitness = self._batch_wcss(X, pop.reshape(self.pop_size, self.n_clusters, n_features))
            best_fitness = float(fitness.min())
            convergence_history.append(best_fitness)

        best_idx = int(np.argmin(fitness))
        centroids = self._decode(pop[best_idx], n_features)
        labels = self._assign_labels(X, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, centroids, convergence_history
