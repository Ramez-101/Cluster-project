import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pca_scatter_grid(X, labels_dict, centroids_dict, metrics_df, save_dir='figures'):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    cmap = plt.get_cmap('tab10')

    names = list(labels_dict.keys())
    ncols = 4
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        labels = labels_dict[name]
        c2 = pca.transform(centroids_dict[name])
        k = centroids_dict[name].shape[0]

        for ki in range(k):
            mask = labels == ki
            ax.scatter(X2[mask, 0], X2[mask, 1], s=8, alpha=0.5, color=cmap(ki))
        ax.scatter(c2[:, 0], c2[:, 1], marker='X', s=150, c='black',
                   zorder=5, edgecolors='white', linewidths=0.5)

        sil = metrics_df.loc[name, 'silhouette'] if name in metrics_df.index else float('nan')
        ax.set_title(f'{name}\nsil={sil:.3f}', fontsize=8)
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=7)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('PCA Scatter — All Algorithms', fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'pca_scatter_all.png'))


def plot_convergence_curves(histories, save_dir='figures'):
    def _is_flat(h):
        return len(set(round(v, 6) for v in h)) <= 2

    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(13, 6))

    ci_ec = {k: v for k, v in histories.items() if not _is_flat(v)}
    bases = {k: v for k, v in histories.items() if _is_flat(v)}
    total = max(1, len(ci_ec) + len(bases))

    for i, (name, hist) in enumerate(ci_ec.items()):
        ax.plot(hist, label=name, color=cmap(i / total), linewidth=1.5)

    for i, (name, hist) in enumerate(bases.items()):
        ax.axhline(hist[-1], linestyle='--', linewidth=1.2,
                   label=f'{name}', color=cmap((len(ci_ec) + i) / total))

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('WCSS', fontsize=11)
    ax.set_title('Convergence Curves', fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'convergence_curves.png'))


def plot_metric_bars(metrics_df, save_dir='figures'):
    metrics_info = [
        ('wcss', 'WCSS (lower=better)', False),
        ('silhouette', 'Silhouette (higher=better)', True),
        ('davies_bouldin', 'Davies-Bouldin (lower=better)', False),
        ('calinski_harabasz', 'Calinski-Harabasz (higher=better)', True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    names = metrics_df.index.tolist()
    x = np.arange(len(names))

    for ax, (col, title, higher) in zip(axes, metrics_info):
        if col not in metrics_df.columns:
            continue
        vals = metrics_df[col].values.astype(float)
        colors = ['steelblue'] * len(vals)
        valid = ~np.isnan(vals)
        if valid.any():
            best = int(np.nanargmax(vals)) if higher else int(np.nanargmin(vals))
            colors[best] = 'gold'
        ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Clustering Metric Comparison', fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'metric_comparison.png'))


def plot_cluster_sizes(labels_dict, n_clusters, save_dir='figures'):
    names = list(labels_dict.keys())
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5 + 2)))

    proportions = []
    for name in names:
        counts = np.bincount(labels_dict[name], minlength=n_clusters).astype(float)
        proportions.append(counts / counts.sum())
    proportions = np.array(proportions)

    left = np.zeros(len(names))
    for k in range(n_clusters):
        ax.barh(names, proportions[:, k], left=left, color=cmap(k),
                label=f'Cluster {k}', edgecolor='white', linewidth=0.3)
        left += proportions[:, k]

    ax.set_xlabel('Proportion of Samples', fontsize=11)
    ax.set_title('Cluster Size Distribution', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'cluster_sizes.png'))


def plot_cluster_profiles(X, labels, centroids, feature_names, scaler, save_dir='figures'):
    orig_c = scaler.inverse_transform(centroids)
    n_clusters = centroids.shape[0]

    max_feat = min(15, len(feature_names))
    feat = feature_names[:max_feat]
    orig_c = orig_c[:, :max_feat]

    col_min = orig_c.min(axis=0)
    col_max = orig_c.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
    norm_c = (orig_c - col_min) / col_range

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(max_feat)

    for k in range(n_clusters):
        n = int(np.sum(labels == k))
        ax.plot(x, norm_c[k], marker='o', linewidth=2,
                color=cmap(k), label=f'Cluster {k} (n={n})')

    ax.set_xticks(x)
    ax.set_xticklabels(feat, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Normalised Value', fontsize=11)
    ax.set_title('Cluster Centroid Profiles', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'cluster_profiles.png'))


def plot_metrics_heatmap(metrics_df, save_dir='figures'):
    import seaborn as sns

    cols = [c for c in ['wcss', 'silhouette', 'davies_bouldin', 'calinski_harabasz']
            if c in metrics_df.columns]
    data = metrics_df[cols].astype(float)
    norm = data.copy()

    for col in cols:
        lo, hi = data[col].min(), data[col].max()
        rng = hi - lo if hi != lo else 1.0
        if col in ('wcss', 'davies_bouldin'):
            norm[col] = (hi - data[col]) / rng
        else:
            norm[col] = (data[col] - lo) / rng

    fig, ax = plt.subplots(figsize=(8, max(5, len(data) * 0.45)))
    sns.heatmap(norm, annot=data.round(3), fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Normalised Score (1=best)'})
    ax.set_title('Algorithm Comparison Heatmap\n(Green = Better)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Algorithm', fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'metrics_heatmap.png'))
