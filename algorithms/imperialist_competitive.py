import numpy as np
from .base import BaseClusterOptimizer


class ImperialistCompetitiveAlgorithm(BaseClusterOptimizer):
    def __init__(self, n_empires=5, assimilation_coeff=2.0,
                 revolution_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_empires = min(n_empires, self.pop_size - 1)
        self.assimilation_coeff = assimilation_coeff
        self.revolution_rate = revolution_rate

    def fit(self, X):
        n_samples, n_features = X.shape
        dim = self.n_clusters * n_features

        padding = 0.1 * (X.max(axis=0) - X.min(axis=0))
        lb = np.tile(X.min(axis=0) - padding, self.n_clusters)
        ub = np.tile(X.max(axis=0) + padding, self.n_clusters)

        countries = self._init_population(X, self.pop_size)
        fitness = self._batch_wcss(X, countries.reshape(self.pop_size, self.n_clusters, n_features))

        order = np.argsort(fitness)
        countries = countries[order]
        fitness = fitness[order]

        imperialists = countries[:self.n_empires].copy()
        imp_fitness = fitness[:self.n_empires].copy()
        colonies = list(countries[self.n_empires:].copy())
        col_fitness = list(fitness[self.n_empires:].copy())

        # Distribute colonies proportional to imperialist power
        power = np.maximum(imp_fitness.max() - imp_fitness, 0.0)
        if power.sum() == 0:
            power = np.ones(self.n_empires)
        power = power / power.sum()

        empire_cols = [[] for _ in range(self.n_empires)]
        empire_fit = [[] for _ in range(self.n_empires)]
        assignments = self.rng.choice(self.n_empires, size=len(colonies), p=power)
        for i, e in enumerate(assignments):
            empire_cols[e].append(colonies[i].copy())
            empire_fit[e].append(col_fitness[i])

        best_fitness = imp_fitness.min()
        convergence_history = []

        for _ in range(self.max_iter):
            # Assimilation — collect all updated colonies, batch-evaluate, scatter back
            all_new_cols, col_addr = [], []
            for e in range(self.n_empires):
                if imp_fitness[e] == np.inf:
                    continue
                for c_idx in range(len(empire_cols[e])):
                    col = empire_cols[e][c_idx]
                    direction = imperialists[e] - col
                    step = self.rng.uniform(0, self.assimilation_coeff) * direction
                    noise_scale = np.linalg.norm(direction) * (np.pi / 4)
                    noise = self.rng.uniform(-noise_scale, noise_scale, size=dim)
                    new_col = np.clip(col + step + noise, lb, ub)
                    empire_cols[e][c_idx] = new_col
                    all_new_cols.append(new_col)
                    col_addr.append((e, c_idx))
            if all_new_cols:
                batch = np.array(all_new_cols).reshape(len(all_new_cols), self.n_clusters, n_features)
                batch_fits = self._batch_wcss(X, batch)
                for idx, (e, c_idx) in enumerate(col_addr):
                    empire_fit[e][c_idx] = float(batch_fits[idx])

            # Revolution — same collect-then-batch pattern
            rev_cols, rev_addr = [], []
            for e in range(self.n_empires):
                for c_idx in range(len(empire_cols[e])):
                    if self.rng.random() < self.revolution_rate:
                        new_sol = self._init_population(X, 1)[0]
                        empire_cols[e][c_idx] = new_sol
                        rev_cols.append(new_sol)
                        rev_addr.append((e, c_idx))
            if rev_cols:
                batch = np.array(rev_cols).reshape(len(rev_cols), self.n_clusters, n_features)
                batch_fits = self._batch_wcss(X, batch)
                for idx, (e, c_idx) in enumerate(rev_addr):
                    empire_fit[e][c_idx] = float(batch_fits[idx])

            # Colony beats its imperialist
            for e in range(self.n_empires):
                if not empire_cols[e]:
                    continue
                best_c_idx = int(np.argmin(empire_fit[e]))
                if empire_fit[e][best_c_idx] < imp_fitness[e]:
                    imperialists[e], empire_cols[e][best_c_idx] = (
                        empire_cols[e][best_c_idx].copy(), imperialists[e].copy()
                    )
                    imp_fitness[e], empire_fit[e][best_c_idx] = (
                        empire_fit[e][best_c_idx], imp_fitness[e]
                    )

            # Competition: weakest empire loses its worst colony to strongest
            total_power = np.full(self.n_empires, np.inf)
            for e in range(self.n_empires):
                if imp_fitness[e] < np.inf:
                    all_fits = [imp_fitness[e]] + list(empire_fit[e])
                    total_power[e] = float(np.mean(all_fits))

            weakest_e = int(np.argmax(total_power))
            strongest_e = int(np.argmin(total_power))

            if empire_cols[weakest_e]:
                worst_c = int(np.argmax(empire_fit[weakest_e]))
                stolen = empire_cols[weakest_e].pop(worst_c)
                stolen_f = empire_fit[weakest_e].pop(worst_c)
                empire_cols[strongest_e].append(stolen)
                empire_fit[strongest_e].append(stolen_f)

                # Empire collapse: no colonies left
                if not empire_cols[weakest_e]:
                    empire_cols[strongest_e].append(imperialists[weakest_e].copy())
                    empire_fit[strongest_e].append(imp_fitness[weakest_e])
                    imp_fitness[weakest_e] = np.inf

            valid = imp_fitness[imp_fitness < np.inf]
            best_fitness = float(valid.min()) if len(valid) > 0 else best_fitness
            convergence_history.append(best_fitness)

        best_e = int(np.argmin(imp_fitness))
        centroids = self._decode(imperialists[best_e], n_features)
        labels = self._assign_labels(X, centroids)

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = best_fitness
        return labels, centroids, convergence_history
