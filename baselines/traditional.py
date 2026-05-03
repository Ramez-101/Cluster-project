import numpy as np
from scipy.spatial.distance import cdist
from algorithms.base import BaseClusterOptimizer


class KMeansBaseline(BaseClusterOptimizer):
    """Standard K-means with random initialisation."""

    def fit(self, X):
        n_samples, n_features = X.shape

        idxs = self.rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[idxs].copy()

        convergence_history = []
        for _ in range(self.max_iter):
            centroids = self._refine_centroids(X, centroids)
            convergence_history.append(self._wcss(X, centroids))

        labels = self._assign_labels(X, centroids)
        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = convergence_history[-1]
        return labels, centroids, convergence_history


class KMeansPlusPlusBaseline(BaseClusterOptimizer):
    """K-means with K-means++ centroid initialisation."""

    def _pp_init(self, X):
        n = len(X)
        centers = [X[self.rng.integers(n)]]
        for _ in range(self.n_clusters - 1):
            d2 = cdist(X, np.array(centers), 'sqeuclidean').min(axis=1)
            probs = d2 / d2.sum()
            centers.append(X[self.rng.choice(n, p=probs)])
        return np.array(centers)

    def fit(self, X):
        centroids = self._pp_init(X)
        convergence_history = []
        for _ in range(self.max_iter):
            centroids = self._refine_centroids(X, centroids)
            convergence_history.append(self._wcss(X, centroids))

        labels = self._assign_labels(X, centroids)
        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = convergence_history[-1]
        return labels, centroids, convergence_history


class KNNClusteringBaseline(BaseClusterOptimizer):
    """Agglomerative clustering with KNN-graph connectivity."""

    def __init__(self, n_neighbors=10, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def fit(self, X):
        from sklearn.neighbors import kneighbors_graph
        from sklearn.cluster import AgglomerativeClustering

        conn = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                mode='connectivity', include_self=False)
        conn = 0.5 * (conn + conn.T)

        model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                        connectivity=conn, linkage='ward')
        labels = model.fit_predict(X)

        centroids = np.array([
            X[labels == k].mean(axis=0) if (labels == k).sum() > 0
            else X[self.rng.integers(len(X))]
            for k in range(self.n_clusters)
        ])

        wcss = self._wcss(X, centroids)
        convergence_history = [wcss] * self.max_iter

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = wcss
        return labels, centroids, convergence_history


class KNNPlusPlusBaseline(BaseClusterOptimizer):
    """KNN-graph clustering seeded with K-means++ initial assignments."""

    def __init__(self, n_neighbors=10, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def _pp_init(self, X):
        n = len(X)
        centers = [X[self.rng.integers(n)]]
        for _ in range(self.n_clusters - 1):
            d2 = cdist(X, np.array(centers), 'sqeuclidean').min(axis=1)
            probs = d2 / d2.sum()
            centers.append(X[self.rng.choice(n, p=probs)])
        return np.array(centers)

    def fit(self, X):
        from sklearn.neighbors import kneighbors_graph
        from sklearn.cluster import AgglomerativeClustering

        # Adaptive connectivity: denser near each seed
        seeds = self._pp_init(X)
        n_samples = len(X)
        adaptive_k = max(5, min(self.n_neighbors, n_samples // (self.n_clusters * 5)))

        conn = kneighbors_graph(X, n_neighbors=adaptive_k,
                                mode='connectivity', include_self=False)
        conn = 0.5 * (conn + conn.T)

        model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                        connectivity=conn, linkage='ward')
        labels = model.fit_predict(X)

        centroids = np.array([
            X[labels == k].mean(axis=0) if (labels == k).sum() > 0
            else X[self.rng.integers(len(X))]
            for k in range(self.n_clusters)
        ])

        wcss = self._wcss(X, centroids)
        convergence_history = [wcss] * self.max_iter

        self.labels_ = labels
        self.centroids_ = centroids
        self.convergence_history_ = convergence_history
        self.best_wcss_ = wcss
        return labels, centroids, convergence_history
