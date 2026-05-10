import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.metrics import confusion_matrix
from utils.gpu import cdist_sqeuclidean

_SILHOUETTE_SAMPLE_CAP = 20_000  # silhouette is O(n²) — sample above this size


def clustering_accuracy_score(y_true, y_pred):
    """Best-matched label accuracy for clustering results.

    This aligns cluster IDs to reference labels with the Hungarian algorithm.
    Returns NaN when no valid reference labels are available.
    """
    if y_true is None:
        return float('nan')

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return float('nan')

    true_ids, _ = np.unique(y_true, return_inverse=True)
    pred_ids, _ = np.unique(y_pred, return_inverse=True)
    true_map = {label: i for i, label in enumerate(true_ids)}
    pred_map = {label: i for i, label in enumerate(pred_ids)}
    true_encoded = np.array([true_map[v] for v in y_true], dtype=int)
    pred_encoded = np.array([pred_map[v] for v in y_pred], dtype=int)

    cm = confusion_matrix(true_encoded, pred_encoded)
    if cm.size == 0:
        return float('nan')

    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    return float(cm[row_ind, col_ind].sum() / cm.sum())


def compute_all_metrics(X, labels, centroids, y_true=None):
    """Returns dict with wcss, silhouette, davies_bouldin, calinski_harabasz, accuracy."""
    dists = cdist_sqeuclidean(X, centroids)
    wcss = float(np.sum(np.min(dists, axis=1)))
    accuracy = clustering_accuracy_score(y_true, labels)

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return {
            'wcss': wcss,
            'silhouette': float('nan'),
            'davies_bouldin': float('nan'),
            'calinski_harabasz': float('nan'),
            'accuracy': accuracy,
        }

    # Silhouette is O(n²) — use a random sample for large datasets
    if len(X) > _SILHOUETTE_SAMPLE_CAP:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=_SILHOUETTE_SAMPLE_CAP, replace=False)
        sil = float(silhouette_score(X[idx], labels[idx]))
    else:
        sil = float(silhouette_score(X, labels))

    return {
        'wcss': wcss,
        'silhouette': sil,
        'davies_bouldin': float(davies_bouldin_score(X, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X, labels)),
        'accuracy': accuracy,
    }
