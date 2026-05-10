"""
ACO variants for continuous clustering (ACOR framework).
All four classes share _BaseACO; each overrides only _compute_weights().
"""
import numpy as np
from .base import BaseClusterOptimizer


class _BaseACO(BaseClusterOptimizer):
    def __init__(self, q=0.5, xi=0.85, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.xi = xi

    def _compute_weights(self, fitness):
        raise NotImplementedError

    def _sample_solution(self, archive, weights, n_features):
        dim = archive.shape[1]
        solution = np.zeros(dim)
        for d in range(dim):
            l = int(self.rng.choice(len(archive), p=weights))
            mu = archive[l, d]
            diffs = np.abs(archive[:, d] - mu)
            sigma = max(self.xi * diffs.mean(), 1e-6)
            solution[d] = self.rng.normal(mu, sigma)
        return solution

    def fit(self, X):
        n_samples, n_features = X.shape

        archive = self._init_population(X, self.pop_size)
        fitness = self._batch_wcss(X, archive.reshape(self.pop_size, self.n_clusters, n_features))

        order = np.argsort(fitness)
        archive = archive[order]
        fitness = fitness[order]

        best_fitness = fitness[0]
        convergence_history = []

        for _ in range(self.max_iter):
            weights = self._compute_weights(fitness)

            new_ants = [self._sample_solution(archive, weights, n_features)
                        for _ in range(self.pop_size)]
            new_ants_arr = np.array(new_ants)
            new_fitness  = self._batch_wcss(
                X, new_ants_arr.reshape(self.pop_size, self.n_clusters, n_features))

            all_sols = np.vstack([archive, new_ants_arr])
            all_fit = np.concatenate([fitness, new_fitness])
            order = np.argsort(all_fit)[:self.pop_size]
            archive = all_sols[order]
            fitness = all_fit[order]

            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
            convergence_history.append(best_fitness)

        centroids = self._decode(archive[0], n_features)
        labels = self._assign_labels(X, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, centroids, convergence_history


class AntSystem(_BaseACO):
    """All archive entries contribute equally."""

    def _compute_weights(self, fitness):
        k = len(fitness)
        return np.ones(k) / k


class ElitistAntSystem(_BaseACO):
    """Best entry receives extra pheromone proportional to elitism_factor."""

    def __init__(self, elitism_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.elitism_factor = elitism_factor

    def _compute_weights(self, fitness):
        k = len(fitness)
        w = np.ones(k) / k
        w[0] += self.elitism_factor / k
        return w / w.sum()


class RankBasedAntSystem(_BaseACO):
    """Weight decays linearly with rank; rank 1 (index 0) is best."""

    def _compute_weights(self, fitness):
        k = len(fitness)
        w = np.array([k - i for i in range(k)], dtype=float)
        return w / w.sum()


class MAXMINAntSystem(_BaseACO):
    """Gaussian weights clamped to [tau_min, tau_max]; restarts on stagnation."""

    def __init__(self, tau_max=0.999, stagnation_limit=20, **kwargs):
        super().__init__(**kwargs)
        self.tau_max = tau_max
        self.stagnation_limit = stagnation_limit

    def _compute_weights(self, fitness):
        k = len(fitness)
        tau_min = self.tau_max / (2 * k)
        q = self.q
        w = np.array([
            (1.0 / (q * k * np.sqrt(2 * np.pi)))
            * np.exp(-(i ** 2) / (2 * q ** 2 * k ** 2))
            for i in range(k)
        ])
        w = np.clip(w, tau_min, self.tau_max)
        return w / w.sum()

    def fit(self, X):
        n_samples, n_features = X.shape

        archive = self._init_population(X, self.pop_size)
        fitness = self._batch_wcss(X, archive.reshape(self.pop_size, self.n_clusters, n_features))

        order = np.argsort(fitness)
        archive = archive[order]
        fitness = fitness[order]

        best_fitness = fitness[0]
        stagnation_count = 0
        convergence_history = []

        for _ in range(self.max_iter):
            weights = self._compute_weights(fitness)

            new_ants = [self._sample_solution(archive, weights, n_features)
                        for _ in range(self.pop_size)]
            new_ants_arr = np.array(new_ants)
            new_fitness  = self._batch_wcss(
                X, new_ants_arr.reshape(self.pop_size, self.n_clusters, n_features))

            all_sols = np.vstack([archive, new_ants_arr])
            all_fit = np.concatenate([fitness, new_fitness])
            order = np.argsort(all_fit)[:self.pop_size]
            archive = all_sols[order]
            fitness = all_fit[order]

            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Restart bottom half on stagnation
            if stagnation_count >= self.stagnation_limit:
                half = self.pop_size // 2
                new_sols = self._init_population(X, half)
                new_fits = self._batch_wcss(X, new_sols.reshape(half, self.n_clusters, n_features))
                archive[half:] = new_sols
                fitness[half:] = new_fits
                order = np.argsort(fitness)
                archive = archive[order]
                fitness = fitness[order]
                stagnation_count = 0

            convergence_history.append(best_fitness)

        centroids = self._decode(archive[0], n_features)
        labels = self._assign_labels(X, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, centroids, convergence_history
