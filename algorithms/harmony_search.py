import numpy as np
from .base import BaseClusterOptimizer


class HarmonySearch(BaseClusterOptimizer):
    def __init__(self, hmcr=0.9, par=0.3, bw_init=0.02, bw_final=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.hmcr = hmcr
        self.par = par
        self.bw_init = bw_init
        self.bw_final = bw_final

    def fit(self, X):
        n_samples, n_features = X.shape
        dim = self.n_clusters * n_features

        padding = 0.1 * (X.max(axis=0) - X.min(axis=0))
        lb_flat = np.tile(X.min(axis=0) - padding, self.n_clusters)
        ub_flat = np.tile(X.max(axis=0) + padding, self.n_clusters)

        hm = self._init_population(X, self.pop_size)
        hm_fitness = np.array([self._wcss(X, self._decode(h, n_features)) for h in hm])

        best_fitness = hm_fitness.min()
        convergence_history = []

        for it in range(self.max_iter):
            # Linearly decaying bandwidth
            t = it / max(1, self.max_iter - 1)
            bw = self.bw_init * (self.bw_final / self.bw_init) ** t

            new_harmony = np.zeros(dim)
            for d in range(dim):
                if self.rng.random() < self.hmcr:
                    new_harmony[d] = hm[self.rng.integers(self.pop_size), d]
                    if self.rng.random() < self.par:
                        new_harmony[d] += self.rng.uniform(-bw, bw)
                else:
                    new_harmony[d] = self.rng.uniform(lb_flat[d], ub_flat[d])

            new_harmony = np.clip(new_harmony, lb_flat, ub_flat)
            new_fitness = self._wcss(X, self._decode(new_harmony, n_features))

            worst_idx = int(np.argmax(hm_fitness))
            if new_fitness < hm_fitness[worst_idx]:
                hm[worst_idx] = new_harmony
                hm_fitness[worst_idx] = new_fitness

            best_fitness = hm_fitness.min()
            convergence_history.append(best_fitness)

        best_idx = int(np.argmin(hm_fitness))
        centroids = self._decode(hm[best_idx], n_features)
        labels = self._assign_labels(X, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, centroids, convergence_history
