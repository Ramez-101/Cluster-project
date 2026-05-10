"""
Flask web interface for the CI/EC Clustering Dashboard.
Run: python app.py  then open http://localhost:5000
"""
import os, sys, io, base64, time, threading, uuid, traceback
import concurrent.futures
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)
jobs: dict = {}          # {job_id: {status, progress, current, results, error}}
DATA_PATH = 'data/marketing_campaign.csv'

# ── Algorithm registry ────────────────────────────────────────────────────────
ALGO_REGISTRY = {
    'harmony_search': ('Harmony Search',  'CI / EC'),
    'eo':             ('EO',              'CI / EC'),
    'tau_eo':         ('tau-EO',          'CI / EC'),
    'ica':            ('ICA',             'CI / EC'),
    'ant_system':     ('Ant System',      'ACO'),
    'elitist_as':     ('Elitist AS',      'ACO'),
    'rank_as':        ('Rank-Based AS',   'ACO'),
    'mmas':           ('MMAS',            'ACO'),
    'ga':             ('GA',              'Evolutionary'),
    'pso':            ('PSO',             'Swarm'),
    'hybrid_ga_pso':  ('Hybrid GA-PSO',   'Hybrid'),
    'kmeans':         ('K-means',         'Classical'),
    'kmeans_pp':      ('K-means++',       'Classical'),
    'knn':            ('KNN Clustering',  'Classical'),
    'knn_pp':         ('KNN++',           'Classical'),
}

GROUPS = {
    'CI / EC':      ['harmony_search', 'eo', 'tau_eo', 'ica'],
    'ACO':          ['ant_system', 'elitist_as', 'rank_as', 'mmas'],
    'Evolutionary': ['ga'],
    'Swarm':        ['pso'],
    'Hybrid':       ['hybrid_ga_pso'],
    'Classical':    ['kmeans', 'kmeans_pp', 'knn', 'knn_pp'],
}


def _build_algo(key, common):
    from algorithms.harmony_search       import HarmonySearch
    from algorithms.extremal_optimization import ExtremalOptimization, TauExtremalOptimization
    from algorithms.imperialist_competitive import ImperialistCompetitiveAlgorithm
    from algorithms.ant_colony           import (AntSystem, ElitistAntSystem,
                                                 RankBasedAntSystem, MAXMINAntSystem)
    from algorithms.genetic_algorithm    import GeneticAlgorithm
    from algorithms.pso                  import ParticleSwarmOptimization
    from algorithms.hybrid_ga_pso        import HybridGAPSO
    from baselines.traditional           import (KMeansBaseline, KMeansPlusPlusBaseline,
                                                 KNNClusteringBaseline, KNNPlusPlusBaseline)
    return {
        'harmony_search': lambda: HarmonySearch(**common),
        'eo':             lambda: ExtremalOptimization(**common),
        'tau_eo':         lambda: TauExtremalOptimization(tau=1.5, **common),
        'ica':            lambda: ImperialistCompetitiveAlgorithm(**common),
        'ant_system':     lambda: AntSystem(**common),
        'elitist_as':     lambda: ElitistAntSystem(**common),
        'rank_as':        lambda: RankBasedAntSystem(**common),
        'mmas':           lambda: MAXMINAntSystem(**common),
        'ga':             lambda: GeneticAlgorithm(**common),
        'pso':            lambda: ParticleSwarmOptimization(**common),
        'hybrid_ga_pso':  lambda: HybridGAPSO(**common),
        'kmeans':         lambda: KMeansBaseline(**common),
        'kmeans_pp':      lambda: KMeansPlusPlusBaseline(**common),
        'knn':            lambda: KNNClusteringBaseline(n_neighbors=10, **common),
        'knn_pp':         lambda: KNNPlusPlusBaseline(n_neighbors=10, **common),
    }[key]()


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


_ALLOWED_EXTS = ('.csv', '.xlsx', '.xls')


def _load_raw(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(filepath)
    try:
        df = pd.read_csv(filepath, sep='\t')
        if df.shape[1] < 5:
            df = pd.read_csv(filepath, sep=',')
    except Exception:
        df = pd.read_csv(filepath, sep=',')
    return df


def _list_datasets():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.isdir(data_dir):
        return []
    return [{'name': f, 'path': f'data/{f}'}
            for f in sorted(os.listdir(data_dir))
            if os.path.splitext(f)[1].lower() in _ALLOWED_EXTS]


def _build_per_seed_payload(per_seed_metrics, per_seed_convergence, seeds, numeric_cols):
    def _safe_local(v):
        if isinstance(v, float) and v != v:
            return None
        return round(float(v), 4) if isinstance(v, (float, np.floating)) else v

    by_seed = {}
    for i, seed_val in enumerate(seeds):
        seed_m, seed_lat, seed_conv = {}, {}, {}
        for algo, records in per_seed_metrics.items():
            if i >= len(records):
                continue
            rec = records[i]
            for col in numeric_cols:
                seed_m.setdefault(col, {})[algo] = _safe_local(rec.get(col))
            seed_lat[algo] = _safe_local(rec.get('time_s'))
            if algo in per_seed_convergence and i < len(per_seed_convergence[algo]):
                seed_conv[algo] = per_seed_convergence[algo][i]
        by_seed[seed_val] = {'metrics': seed_m, 'latency': seed_lat, 'convergence': seed_conv}
    return {'seeds': seeds, 'by_seed': by_seed}


def _run_job(job_id, selected_keys, k, max_iter, pop_size, seed, num_seeds,
             dataset_path=None, use_gpu=True, n_workers=3):
    try:
        jobs[job_id].update(status='running', progress=2, current='Preprocessing data…')
        from utils.preprocessing import load_and_preprocess
        from utils.evaluation    import compute_all_metrics

        effective_path = dataset_path or DATA_PATH
        X, feature_names, scaler, y_true, target_name = load_and_preprocess(effective_path)

        seeds       = list(range(seed, seed + num_seeds))
        total_tasks = len(selected_keys) * num_seeds
        done        = 0

        numeric_cols = ['wcss', 'silhouette', 'davies_bouldin',
                        'calinski_harabasz', 'accuracy', 'time_s']

        def _safe(v):
            if isinstance(v, float) and (v != v):
                return None
            return round(float(v), 4) if isinstance(v, (float, np.floating)) else v

        # per-display-name → list of metric dicts (one per seed)
        all_metrics_by_seed: dict[str, list[dict]] = {
            ALGO_REGISTRY[k][0]: [] for k in selected_keys
        }
        per_seed_metrics: dict[str, list[dict]] = {}
        per_seed_convergence: dict[str, list[list]] = {}

        # store algo_results from each seed; keep best-seed results for charts
        best_wcss_global   = float('inf')
        best_seed_algo_results: dict = {}
        best_seed_mdf      = None

        for seed_offset, cur_seed in enumerate(seeds):
            common = dict(n_clusters=k, max_iter=max_iter,
                          pop_size=pop_size, random_state=cur_seed, use_gpu=use_gpu)

            seed_algo_results: dict = {}
            for key in selected_keys:
                display_name = ALGO_REGISTRY[key][0]
                seed_label   = (f'Seed {cur_seed}/{seeds[-1]}  ·  {display_name}…'
                                if num_seeds > 1 else f'Running {display_name}…')
                jobs[job_id].update(
                    current=seed_label,
                    progress=5 + int(done / total_tasks * 80),
                )
                algo = _build_algo(key, common)
                t0   = time.perf_counter()
                labels, centroids, history = algo.fit(X)
                elapsed = time.perf_counter() - t0
                seed_algo_results[display_name] = dict(
                    labels=labels, centroids=centroids,
                    history=history, time_s=elapsed, key=key,
                )
                done += 1

            # Compute metrics for this seed
            seed_metrics_rows = []
            for name, res in seed_algo_results.items():
                m = compute_all_metrics(X, res['labels'], res['centroids'], y_true=y_true)
                m['algorithm'] = name
                m['time_s']    = res['time_s']
                seed_metrics_rows.append(m)
                all_metrics_by_seed[name].append(m)
                seed_record = {col: _safe(m.get(col)) for col in numeric_cols}
                seed_record['seed'] = cur_seed
                per_seed_metrics.setdefault(name, []).append(seed_record)
                per_seed_convergence.setdefault(name, []).append(
                    [round(float(v), 4) for v in res['history']]
                )

            seed_mdf = pd.DataFrame(seed_metrics_rows).set_index('algorithm')
            seed_best_wcss = seed_mdf['wcss'].min()
            if seed_best_wcss < best_wcss_global:
                best_wcss_global    = seed_best_wcss
                best_seed_algo_results = seed_algo_results
                best_seed_mdf       = seed_mdf

        jobs[job_id].update(current='Aggregating metrics…', progress=87)

        # ── Aggregate across seeds ─────────────────────────────────────────
        mean_rows, std_rows = [], []
        for name in all_metrics_by_seed:
            seed_vals = all_metrics_by_seed[name]
            mean_row  = {'algorithm': name}
            std_row   = {'algorithm': name}
            for col in numeric_cols:
                vals = [v[col] for v in seed_vals if col in v and not (
                    isinstance(v[col], float) and v[col] != v[col])]
                mean_row[col] = float(np.mean(vals)) if vals else float('nan')
                std_row[col]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean_rows.append(mean_row)
            std_rows.append(std_row)

        mdf     = pd.DataFrame(mean_rows).set_index('algorithm')
        mdf_std = pd.DataFrame(std_rows).set_index('algorithm')

        # Console summary
        print(flush=True)
        hdr = (f'=== Clustering metrics (avg of {num_seeds} seeds: {seeds[0]}–{seeds[-1]}) ==='
               if num_seeds > 1 else '=== Clustering metrics ===')
        print(hdr, flush=True)
        if target_name is not None:
            print(f'Accuracy aligned against {target_name}.', flush=True)
        for name, row in mdf.iterrows():
            acc    = row.get('accuracy')
            acc_t  = 'N/A' if pd.isna(acc) else f'{acc:.4f}'
            if num_seeds > 1:
                ws  = mdf_std.loc[name, 'wcss']
                ss  = mdf_std.loc[name, 'silhouette']
                print(
                    f"{name:24s} acc={acc_t} | "
                    f"wcss={row['wcss']:.2f}±{ws:.2f} | "
                    f"sil={row['silhouette']:.4f}±{ss:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"{name:24s} acc={acc_t} | "
                    f"wcss={row['wcss']:.2f} | sil={row['silhouette']:.4f} | "
                    f"db={row['davies_bouldin']:.4f} | ch={row['calinski_harabasz']:.2f} | "
                    f"time={row['time_s']:.3f}s",
                    flush=True,
                )
        if 'accuracy' in mdf.columns and mdf['accuracy'].notna().any():
            print('Best accuracy:', mdf['accuracy'].idxmax(),
                  f"({mdf['accuracy'].max():.4f})", flush=True)

        jobs[job_id].update(current='Rendering charts…', progress=92)

        # ── Use best-seed results for charts ───────────────────────────────
        algo_results = best_seed_algo_results
        chart_mdf    = best_seed_mdf if best_seed_mdf is not None else mdf

        # PCA scatter grid
        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)
        ev  = pca.explained_variance_ratio_
        cmap = plt.get_cmap('tab10')

        ncols = min(3, len(algo_results))
        nrows = (len(algo_results) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 3.8), squeeze=False)
        for idx, (name, res) in enumerate(algo_results.items()):
            ax = axes[idx // ncols][idx % ncols]
            for ki in range(k):
                m = res['labels'] == ki
                ax.scatter(X2[m, 0], X2[m, 1], s=7, alpha=0.45, color=cmap(ki))
            c2 = pca.transform(res['centroids'])
            ax.scatter(c2[:, 0], c2[:, 1], marker='X', s=160, c='black',
                       zorder=5, edgecolors='white', linewidths=0.5)
            sil = chart_mdf.loc[name, 'silhouette'] if name in chart_mdf.index else float('nan')
            ax.set_title(f'{name}\nSil={sil:.3f}', fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(len(algo_results), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.suptitle(f'PCA Scatter  (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)',
                     fontsize=11)
        fig.tight_layout()
        pca_img = _fig_to_b64(fig)

        # Cluster profiles (best-WCSS algorithm from best seed)
        best_wcss_name = chart_mdf['wcss'].idxmin()
        best_res = algo_results[best_wcss_name]
        orig_c = scaler.inverse_transform(best_res['centroids'])
        max_f  = min(14, len(feature_names))
        feat   = feature_names[:max_f]
        oc     = orig_c[:, :max_f]
        lo, hi = oc.min(0), oc.max(0)
        rng_f  = np.where(hi - lo == 0, 1, hi - lo)
        norm_c = (oc - lo) / rng_f

        fig2, ax2 = plt.subplots(figsize=(14, 4))
        for ki in range(k):
            n_ki = int(np.sum(best_res['labels'] == ki))
            ax2.plot(np.arange(max_f), norm_c[ki], marker='o', linewidth=2,
                     color=cmap(ki), label=f'Cluster {ki}  (n={n_ki})')
        ax2.set_xticks(np.arange(max_f))
        ax2.set_xticklabels(feat, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Normalised value', fontsize=10)
        ax2.set_title(f'Cluster Centroid Profiles — {best_wcss_name}', fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        fig2.tight_layout()
        profile_img = _fig_to_b64(fig2)

        # ── Serialisable payload ───────────────────────────────────────────
        metrics_out     = {}
        metrics_std_out = {}
        for col in mdf.columns:
            metrics_out[col]     = {idx: _safe(val) for idx, val in mdf[col].items()}
            metrics_std_out[col] = {idx: _safe(val) for idx, val in mdf_std[col].items()}

        jobs[job_id].update(
            status   = 'done',
            progress = 100,
            current  = 'Done',
            results  = {
                'metrics':        metrics_out,
                'metrics_std':    metrics_std_out,
                'num_seeds':      num_seeds,
                'seeds_used':     seeds,
                'convergence':    {n: [round(float(v), 4) for v in r['history']]
                                   for n, r in algo_results.items()},
                'cluster_sizes':  {n: np.bincount(r['labels'], minlength=k).tolist()
                                   for n, r in algo_results.items()},
                'latency':        {n: round(r['time_s'], 3)
                                   for n, r in algo_results.items()},
                'pca_img':        pca_img,
                'profile_img':    profile_img,
                'best_wcss':      mdf['wcss'].idxmin(),
                'best_acc':       mdf['accuracy'].idxmax() if mdf['accuracy'].notna().any() else None,
                'best_sil':       mdf['silhouette'].idxmax(),
                'best_ch':        mdf['calinski_harabasz'].idxmax(),
                'best_db':        mdf['davies_bouldin'].idxmin(),
                'n_samples':      int(X.shape[0]),
                'n_features':     int(X.shape[1]),
                'n_clusters':     k,
                'max_iter':       max_iter,
                'target_name':    target_name,
                'per_seed_data':  _build_per_seed_payload(
                                      per_seed_metrics, per_seed_convergence,
                                      seeds, numeric_cols
                                  ) if num_seeds > 1 else None,
            }
        )

    except Exception:
        jobs[job_id].update(status='error', error=traceback.format_exc())


def _run_job_parallel(job_id, selected_keys, k, max_iter, pop_size, seed, num_seeds,
                      dataset_path=None, use_gpu=True, n_workers=3):
    try:
        jobs[job_id].update(status='running', progress=2, current='Preprocessing data…')
        from utils.preprocessing import load_and_preprocess
        from utils.evaluation    import compute_all_metrics

        effective_path = dataset_path or DATA_PATH
        X, feature_names, scaler, y_true, target_name = load_and_preprocess(effective_path)

        seeds        = list(range(seed, seed + num_seeds))
        total_tasks  = len(selected_keys) * num_seeds
        actual_workers = min(n_workers, total_tasks)
        lock         = threading.Lock()
        done_count   = [0]

        numeric_cols = ['wcss', 'silhouette', 'davies_bouldin',
                        'calinski_harabasz', 'accuracy', 'time_s']

        def _safe(v):
            if isinstance(v, float) and (v != v):
                return None
            return round(float(v), 4) if isinstance(v, (float, np.floating)) else v

        def _run_single(key, cur_seed):
            common = dict(n_clusters=k, max_iter=max_iter,
                          pop_size=pop_size, random_state=cur_seed, use_gpu=use_gpu)
            algo = _build_algo(key, common)
            t0   = time.perf_counter()
            labels, centroids, history = algo.fit(X)
            elapsed = time.perf_counter() - t0
            with lock:
                done_count[0] += 1
                pct = 5 + int(done_count[0] / total_tasks * 80)
                display = ALGO_REGISTRY[key][0]
                jobs[job_id].update(
                    progress=pct,
                    current=f'Seed {cur_seed} · {display} done',
                )
            return key, cur_seed, labels, centroids, history, elapsed

        # Submit all (seed, algo) pairs; collect into per-seed dicts
        seed_raw: dict = defaultdict(dict)   # cur_seed → {display_name: result_dict}
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_run_single, key, cur_seed): (key, cur_seed)
                for cur_seed in seeds
                for key in selected_keys
            }
            for future in concurrent.futures.as_completed(futures):
                key, cur_seed, labels, centroids, history, elapsed = future.result()
                display_name = ALGO_REGISTRY[key][0]
                seed_raw[cur_seed][display_name] = dict(
                    labels=labels, centroids=centroids,
                    history=history, time_s=elapsed, key=key,
                )

        # Assemble metrics in seed order (same logic as _run_job)
        all_metrics_by_seed: dict[str, list[dict]] = {
            ALGO_REGISTRY[k][0]: [] for k in selected_keys
        }
        per_seed_metrics:     dict[str, list[dict]] = {}
        per_seed_convergence: dict[str, list[list]] = {}
        best_wcss_global      = float('inf')
        best_seed_algo_results: dict = {}
        best_seed_mdf         = None

        for cur_seed in seeds:
            seed_algo_results = seed_raw[cur_seed]
            seed_metrics_rows = []
            for name, res in seed_algo_results.items():
                m = compute_all_metrics(X, res['labels'], res['centroids'], y_true=y_true)
                m['algorithm'] = name
                m['time_s']    = res['time_s']
                seed_metrics_rows.append(m)
                all_metrics_by_seed[name].append(m)
                seed_record = {col: _safe(m.get(col)) for col in numeric_cols}
                seed_record['seed'] = cur_seed
                per_seed_metrics.setdefault(name, []).append(seed_record)
                per_seed_convergence.setdefault(name, []).append(
                    [round(float(v), 4) for v in res['history']]
                )
            seed_mdf = pd.DataFrame(seed_metrics_rows).set_index('algorithm')
            seed_best_wcss = seed_mdf['wcss'].min()
            if seed_best_wcss < best_wcss_global:
                best_wcss_global       = seed_best_wcss
                best_seed_algo_results = seed_algo_results
                best_seed_mdf          = seed_mdf

        jobs[job_id].update(current='Aggregating metrics…', progress=87)

        mean_rows, std_rows = [], []
        for name in all_metrics_by_seed:
            seed_vals = all_metrics_by_seed[name]
            mean_row  = {'algorithm': name}
            std_row   = {'algorithm': name}
            for col in numeric_cols:
                vals = [v[col] for v in seed_vals if col in v and not (
                    isinstance(v[col], float) and v[col] != v[col])]
                mean_row[col] = float(np.mean(vals)) if vals else float('nan')
                std_row[col]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean_rows.append(mean_row)
            std_rows.append(std_row)

        mdf     = pd.DataFrame(mean_rows).set_index('algorithm')
        mdf_std = pd.DataFrame(std_rows).set_index('algorithm')

        jobs[job_id].update(current='Rendering charts…', progress=92)

        algo_results = best_seed_algo_results
        chart_mdf    = best_seed_mdf if best_seed_mdf is not None else mdf

        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)
        ev  = pca.explained_variance_ratio_
        cmap = plt.get_cmap('tab10')

        ncols = min(3, len(algo_results))
        nrows = (len(algo_results) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 3.8), squeeze=False)
        for idx, (name, res) in enumerate(algo_results.items()):
            ax = axes[idx // ncols][idx % ncols]
            for ki in range(k):
                m = res['labels'] == ki
                ax.scatter(X2[m, 0], X2[m, 1], s=7, alpha=0.45, color=cmap(ki))
            c2 = pca.transform(res['centroids'])
            ax.scatter(c2[:, 0], c2[:, 1], marker='X', s=160, c='black',
                       zorder=5, edgecolors='white', linewidths=0.5)
            sil = chart_mdf.loc[name, 'silhouette'] if name in chart_mdf.index else float('nan')
            ax.set_title(f'{name}\nSil={sil:.3f}', fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(len(algo_results), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.suptitle(f'PCA Scatter  (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)',
                     fontsize=11)
        fig.tight_layout()
        pca_img = _fig_to_b64(fig)

        best_wcss_name = chart_mdf['wcss'].idxmin()
        best_res = algo_results[best_wcss_name]
        orig_c = scaler.inverse_transform(best_res['centroids'])
        max_f  = min(14, len(feature_names))
        feat   = feature_names[:max_f]
        oc     = orig_c[:, :max_f]
        lo, hi = oc.min(0), oc.max(0)
        rng_f  = np.where(hi - lo == 0, 1, hi - lo)
        norm_c = (oc - lo) / rng_f

        fig2, ax2 = plt.subplots(figsize=(14, 4))
        for ki in range(k):
            n_ki = int(np.sum(best_res['labels'] == ki))
            ax2.plot(np.arange(max_f), norm_c[ki], marker='o', linewidth=2,
                     color=cmap(ki), label=f'Cluster {ki}  (n={n_ki})')
        ax2.set_xticks(np.arange(max_f))
        ax2.set_xticklabels(feat, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Normalised value', fontsize=10)
        ax2.set_title(f'Cluster Centroid Profiles — {best_wcss_name}', fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        fig2.tight_layout()
        profile_img = _fig_to_b64(fig2)

        metrics_out     = {}
        metrics_std_out = {}
        for col in mdf.columns:
            metrics_out[col]     = {idx: _safe(val) for idx, val in mdf[col].items()}
            metrics_std_out[col] = {idx: _safe(val) for idx, val in mdf_std[col].items()}

        jobs[job_id].update(
            status   = 'done',
            progress = 100,
            current  = 'Done',
            results  = {
                'metrics':        metrics_out,
                'metrics_std':    metrics_std_out,
                'num_seeds':      num_seeds,
                'seeds_used':     seeds,
                'convergence':    {n: [round(float(v), 4) for v in r['history']]
                                   for n, r in algo_results.items()},
                'cluster_sizes':  {n: np.bincount(r['labels'], minlength=k).tolist()
                                   for n, r in algo_results.items()},
                'latency':        {n: round(r['time_s'], 3)
                                   for n, r in algo_results.items()},
                'pca_img':        pca_img,
                'profile_img':    profile_img,
                'best_wcss':      mdf['wcss'].idxmin(),
                'best_acc':       mdf['accuracy'].idxmax() if mdf['accuracy'].notna().any() else None,
                'best_sil':       mdf['silhouette'].idxmax(),
                'best_ch':        mdf['calinski_harabasz'].idxmax(),
                'best_db':        mdf['davies_bouldin'].idxmin(),
                'n_samples':      int(X.shape[0]),
                'n_features':     int(X.shape[1]),
                'n_clusters':     k,
                'max_iter':       max_iter,
                'target_name':    target_name,
                'per_seed_data':  _build_per_seed_payload(
                                      per_seed_metrics, per_seed_convergence,
                                      seeds, numeric_cols
                                  ) if num_seeds > 1 else None,
            }
        )

    except Exception:
        jobs[job_id].update(status='error', error=traceback.format_exc())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    datasets = _list_datasets()
    return render_template('index.html',
                           groups=GROUPS,
                           registry=ALGO_REGISTRY,
                           data_exists=len(datasets) > 0,
                           datasets=datasets,
                           default_dataset=DATA_PATH)


@app.route('/api/datasets')
def api_datasets():
    return jsonify(datasets=_list_datasets())


@app.route('/api/dataset-preview')
def api_dataset_preview():
    path = request.args.get('path', '')
    safe_dir  = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    full_path = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    if not full_path.startswith(safe_dir) or os.path.splitext(full_path)[1].lower() not in _ALLOWED_EXTS:
        return jsonify(error='Invalid path'), 400
    if not os.path.exists(full_path):
        return jsonify(error='File not found'), 404
    try:
        df = _load_raw(full_path)
    except Exception as exc:
        return jsonify(error=str(exc)), 500
    has_response = 'Response' in df.columns
    return jsonify(
        name=os.path.basename(full_path),
        columns=list(df.columns),
        n_rows=len(df),
        n_cols=len(df.columns),
        has_target=has_response,
        target_name='Response' if has_response else None,
        preview_rows=df.head(5).fillna('').astype(str).values.tolist(),
    )


@app.route('/api/run', methods=['POST'])
def api_run():
    body     = request.get_json(force=True)
    selected = body.get('algorithms', [])
    if not selected:
        return jsonify(error='Select at least one algorithm.'), 400

    k         = max(2,  min(10,  int(body.get('k', 4))))
    max_iter  = max(10, min(500, int(body.get('max_iter', 100))))
    pop_size  = max(5,  min(100, int(body.get('pop_size', 20))))
    seed      = int(body.get('seed', 42))
    num_seeds = max(1,  min(20,  int(body.get('num_seeds', 1))))

    dataset_path = body.get('dataset_path', DATA_PATH)
    safe_dir  = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    full_path = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path))
    if not full_path.startswith(safe_dir) or os.path.splitext(full_path)[1].lower() not in _ALLOWED_EXTS:
        return jsonify(error='Invalid dataset path.'), 400
    if not os.path.exists(full_path):
        return jsonify(error='Dataset not found.'), 404

    use_gpu   = bool(body.get('use_gpu', True))
    parallel  = bool(body.get('parallel', False))
    n_workers = max(1, min(12, int(body.get('n_workers', 3))))

    jid = str(uuid.uuid4())[:8]
    jobs[jid] = dict(status='pending', progress=0, current='Queued', results=None, error=None)
    target_fn = _run_job_parallel if parallel else _run_job
    threading.Thread(target=target_fn,
                     args=(jid, selected, k, max_iter, pop_size, seed, num_seeds,
                           dataset_path, use_gpu, n_workers),
                     daemon=True).start()
    return jsonify(job_id=jid)


@app.route('/api/gpu_info')
def api_gpu_info():
    from utils.gpu import get_gpu_info
    return jsonify(get_gpu_info())


@app.route('/api/status/<jid>')
def api_status(jid):
    j = jobs.get(jid)
    if not j:
        return jsonify(error='Not found'), 404
    return jsonify(status=j['status'], progress=j['progress'],
                   current=j['current'], error=j.get('error'))


@app.route('/api/results/<jid>')
def api_results(jid):
    j = jobs.get(jid)
    if not j:
        return jsonify(error='Not found'), 404
    if j['status'] != 'done':
        return jsonify(error='Not ready'), 202
    return jsonify(j['results'])


@app.route('/api/export/<jid>')
def api_export(jid):
    from flask import Response

    j = jobs.get(jid)
    if not j or j['status'] != 'done':
        return jsonify(error='Not ready'), 404

    res   = j['results']
    m     = res['metrics']
    m_std = res.get('metrics_std', {})
    lat   = res['latency']
    ns    = res.get('num_seeds', 1)
    seeds = res.get('seeds_used', [])
    algos = list(m.get('wcss', {}).keys())
    per_seed = res.get('per_seed_data')   # None when ns == 1

    def _fmt(v):
        if v is None:
            return ''
        try:
            f = float(v)
            return '' if f != f else f          # NaN → ''
        except (TypeError, ValueError):
            return v

    out = io.StringIO()

    # ── Section 1: Summary (mean ± std) ──────────────────────────────────────
    if ns > 1:
        out.write(f'SUMMARY — mean ± std across {ns} seeds ({seeds[0]}–{seeds[-1]})\n')
    else:
        out.write(f'RESULTS — seed {seeds[0] if seeds else "?"}\n')

    summary_cols = ['Algorithm', 'Num_Seeds',
                    'WCSS_mean', 'WCSS_std',
                    'Silhouette_mean', 'Silhouette_std',
                    'Davies_Bouldin_mean', 'Davies_Bouldin_std',
                    'Calinski_Harabasz_mean', 'Calinski_Harabasz_std',
                    'Accuracy_mean', 'Accuracy_std',
                    'Time_s_mean']
    summary_rows = []
    for a in algos:
        summary_rows.append({
            'Algorithm':               a,
            'Num_Seeds':               ns,
            'WCSS_mean':               _fmt(m.get('wcss', {}).get(a)),
            'WCSS_std':                _fmt(m_std.get('wcss', {}).get(a)),
            'Silhouette_mean':         _fmt(m.get('silhouette', {}).get(a)),
            'Silhouette_std':          _fmt(m_std.get('silhouette', {}).get(a)),
            'Davies_Bouldin_mean':     _fmt(m.get('davies_bouldin', {}).get(a)),
            'Davies_Bouldin_std':      _fmt(m_std.get('davies_bouldin', {}).get(a)),
            'Calinski_Harabasz_mean':  _fmt(m.get('calinski_harabasz', {}).get(a)),
            'Calinski_Harabasz_std':   _fmt(m_std.get('calinski_harabasz', {}).get(a)),
            'Accuracy_mean':           _fmt(m.get('accuracy', {}).get(a)),
            'Accuracy_std':            _fmt(m_std.get('accuracy', {}).get(a)),
            'Time_s_mean':             _fmt(lat.get(a)),
        })
    pd.DataFrame(summary_rows, columns=summary_cols).to_csv(out, index=False)

    # ── Section 2: Per-seed breakdown (only when multi-seed) ─────────────────
    if per_seed and ns > 1:
        out.write('\n')
        out.write(f'PER-SEED RESULTS\n')
        seed_cols = ['Algorithm', 'Seed',
                     'WCSS', 'Silhouette', 'Davies_Bouldin',
                     'Calinski_Harabasz', 'Accuracy', 'Time_s']
        seed_rows = []
        by_seed = per_seed.get('by_seed', {})
        for sv in seeds:
            seed_block = by_seed.get(sv) or by_seed.get(str(sv), {})
            seed_metrics = seed_block.get('metrics', {})
            seed_latency = seed_block.get('latency', {})
            for a in algos:
                seed_rows.append({
                    'Algorithm':         a,
                    'Seed':              sv,
                    'WCSS':              _fmt(seed_metrics.get('wcss', {}).get(a)),
                    'Silhouette':        _fmt(seed_metrics.get('silhouette', {}).get(a)),
                    'Davies_Bouldin':    _fmt(seed_metrics.get('davies_bouldin', {}).get(a)),
                    'Calinski_Harabasz': _fmt(seed_metrics.get('calinski_harabasz', {}).get(a)),
                    'Accuracy':          _fmt(seed_metrics.get('accuracy', {}).get(a)),
                    'Time_s':            _fmt(seed_latency.get(a)),
                })
        pd.DataFrame(seed_rows, columns=seed_cols).to_csv(out, index=False)

    return Response(
        out.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=clustering_results.csv'},
    )


if __name__ == '__main__':
    print("Starting Clustering Dashboard → http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
