"""
Flask web interface for the CI/EC Clustering Dashboard.
Run: python app.py  then open http://localhost:5000
"""
import os, sys, io, base64, time, threading, uuid, traceback

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


def _run_job(job_id, selected_keys, k, max_iter, pop_size, seed):
    try:
        jobs[job_id].update(status='running', progress=2, current='Preprocessing data…')
        from utils.preprocessing import load_and_preprocess
        from utils.evaluation    import compute_all_metrics

        X, feature_names, scaler, y_true, target_name = load_and_preprocess(DATA_PATH)
        common = dict(n_clusters=k, max_iter=max_iter, pop_size=pop_size, random_state=seed)

        algo_results = {}
        n = len(selected_keys)

        for i, key in enumerate(selected_keys):
            display_name = ALGO_REGISTRY[key][0]
            jobs[job_id].update(current=f'Running {display_name}…',
                                progress=5 + int(i / n * 80))
            algo = _build_algo(key, common)
            t0 = time.perf_counter()
            labels, centroids, history = algo.fit(X)
            elapsed = time.perf_counter() - t0
            algo_results[display_name] = dict(
                labels=labels, centroids=centroids,
                history=history, time_s=elapsed, key=key,
            )

        jobs[job_id].update(current='Computing metrics…', progress=87)

        metrics_rows = []
        for name, res in algo_results.items():
            m = compute_all_metrics(X, res['labels'], res['centroids'], y_true=y_true)
            m['algorithm'] = name
            m['time_s']    = res['time_s']
            metrics_rows.append(m)
        mdf = pd.DataFrame(metrics_rows).set_index('algorithm')

        print(flush=True)
        print('=== Clustering metrics ===', flush=True)
        if target_name is not None:
            print(f'Accuracy is label-aligned against {target_name}.', flush=True)
        for name, row in mdf.iterrows():
            acc = row.get('accuracy')
            acc_txt = 'N/A' if pd.isna(acc) else f'{acc:.4f}'
            print(
                f"{name:24s} "
                f"acc={acc_txt} | "
                f"wcss={row['wcss']:.2f} | "
                f"sil={row['silhouette']:.4f} | "
                f"db={row['davies_bouldin']:.4f} | "
                f"ch={row['calinski_harabasz']:.2f} | "
                f"time={row['time_s']:.3f}s",
                flush=True,
            )
        if 'accuracy' in mdf.columns and mdf['accuracy'].notna().any():
            print('Best accuracy:', mdf['accuracy'].idxmax(), f"({mdf['accuracy'].max():.4f})", flush=True)

        jobs[job_id].update(current='Rendering charts…', progress=92)

        # ── PCA scatter grid ────────────────────────────────────────────
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
            sil = mdf.loc[name, 'silhouette'] if name in mdf.index else float('nan')
            ax.set_title(f'{name}\nSil={sil:.3f}', fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(len(algo_results), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.suptitle(f'PCA Scatter  (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)',
                     fontsize=11)
        fig.tight_layout()
        pca_img = _fig_to_b64(fig)

        # ── Cluster profiles (best-WCSS algorithm) ───────────────────────
        best_wcss_name = mdf['wcss'].idxmin()
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

        # ── Serialisable payload ─────────────────────────────────────────
        def _safe(v):
            if isinstance(v, float) and (v != v):   # NaN
                return None
            return round(float(v), 4) if isinstance(v, (float, np.floating)) else v

        metrics_out = {}
        for col in mdf.columns:
            metrics_out[col] = {idx: _safe(val) for idx, val in mdf[col].items()}

        jobs[job_id].update(
            status   = 'done',
            progress = 100,
            current  = 'Done',
            results  = {
                'metrics':        metrics_out,
                'convergence':    {n: [round(float(v), 4) for v in r['history']]
                                   for n, r in algo_results.items()},
                'cluster_sizes':  {n: np.bincount(r['labels'], minlength=k).tolist()
                                   for n, r in algo_results.items()},
                'latency':        {n: round(r['time_s'], 3)
                                   for n, r in algo_results.items()},
                'pca_img':        pca_img,
                'profile_img':    profile_img,
                'best_wcss':      best_wcss_name,
                'best_acc':       mdf['accuracy'].idxmax() if mdf['accuracy'].notna().any() else None,
                'best_sil':       mdf['silhouette'].idxmax(),
                'best_ch':        mdf['calinski_harabasz'].idxmax(),
                'best_db':        mdf['davies_bouldin'].idxmin(),
                'n_samples':      int(X.shape[0]),
                'n_features':     int(X.shape[1]),
                'n_clusters':     k,
                'max_iter':       max_iter,
                'target_name':    target_name,
            }
        )

    except Exception:
        jobs[job_id].update(status='error', error=traceback.format_exc())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           groups=GROUPS,
                           registry=ALGO_REGISTRY,
                           data_exists=os.path.exists(DATA_PATH))


@app.route('/api/run', methods=['POST'])
def api_run():
    body     = request.get_json(force=True)
    selected = body.get('algorithms', [])
    if not selected:
        return jsonify(error='Select at least one algorithm.'), 400

    k        = max(2, min(10, int(body.get('k', 4))))
    max_iter = max(10, min(500, int(body.get('max_iter', 100))))
    pop_size = max(5,  min(100, int(body.get('pop_size', 20))))
    seed     = int(body.get('seed', 42))

    jid = str(uuid.uuid4())[:8]
    jobs[jid] = dict(status='pending', progress=0, current='Queued', results=None, error=None)
    threading.Thread(target=_run_job,
                     args=(jid, selected, k, max_iter, pop_size, seed),
                     daemon=True).start()
    return jsonify(job_id=jid)


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
    m = j['results']['metrics']
    lat = j['results']['latency']
    algos = list(m.get('wcss', {}).keys())
    rows = []
    for a in algos:
        rows.append({
            'Algorithm':          a,
            'Accuracy':           m.get('accuracy', {}).get(a),
            'WCSS':               m.get('wcss', {}).get(a),
            'Silhouette':         m.get('silhouette', {}).get(a),
            'Davies-Bouldin':     m.get('davies_bouldin', {}).get(a),
            'Calinski-Harabasz':  m.get('calinski_harabasz', {}).get(a),
            'Time_s':             lat.get(a),
        })
    out = io.StringIO()
    pd.DataFrame(rows).to_csv(out, index=False)
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':
                             'attachment; filename=clustering_results.csv'})


if __name__ == '__main__':
    print("Starting Clustering Dashboard → http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
