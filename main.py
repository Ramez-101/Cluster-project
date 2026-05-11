"""
Customer Segmentation with CI/EC Algorithms
Usage: python main.py [--data data/marketing_campaign.csv] [--k 4] [--seed 42]
                      [--max-iter 200] [--pop-size 30] [--num-seeds 1]
                      [--no-gpu] [--parallel] [--workers N]
"""
import argparse
import concurrent.futures
import os
import sys
import time

import numpy as np
import pandas as pd


def _build_algorithms(common):
    from algorithms.harmony_search import HarmonySearch
    from algorithms.extremal_optimization import ExtremalOptimization, TauExtremalOptimization
    from algorithms.imperialist_competitive import ImperialistCompetitiveAlgorithm
    from algorithms.ant_colony import (AntSystem, ElitistAntSystem,
                                       RankBasedAntSystem, MAXMINAntSystem)
    from algorithms.genetic_algorithm import GeneticAlgorithm
    from algorithms.pso import ParticleSwarmOptimization
    from algorithms.hybrid_ga_pso import HybridGAPSO
    from baselines.traditional import (KMeansBaseline, KMeansPlusPlusBaseline,
                                       KNNClusteringBaseline, KNNPlusPlusBaseline)
    return {
        'Harmony Search':  HarmonySearch(**common),
        'EO':              ExtremalOptimization(**common),
        'tau-EO':          TauExtremalOptimization(tau=1.5, **common),
        'ICA':             ImperialistCompetitiveAlgorithm(**common),
        'Ant System':      AntSystem(**common),
        'Elitist AS':      ElitistAntSystem(**common),
        'Rank-Based AS':   RankBasedAntSystem(**common),
        'MMAS':            MAXMINAntSystem(**common),
        'GA':              GeneticAlgorithm(**common),
        'PSO':             ParticleSwarmOptimization(**common),
        'Hybrid GA-PSO':   HybridGAPSO(**common),
        'K-means':         KMeansBaseline(**common),
        'K-means++':       KMeansPlusPlusBaseline(**common),
        'KNN Clustering':  KNNClusteringBaseline(n_neighbors=10, **common),
        'KNN++':           KNNPlusPlusBaseline(n_neighbors=10, **common),
    }


def _run_one_seed(X, k, max_iter, pop_size, seed, y_true, verbose=True,
                  use_gpu=True, parallel=False, workers=3):
    """Run all algorithms once with the given seed. Returns (results_dict, metrics_df)."""
    from utils.evaluation import compute_all_metrics

    common = dict(n_clusters=k, max_iter=max_iter, pop_size=pop_size,
                  random_state=seed, use_gpu=use_gpu)
    algorithms = _build_algorithms(common)

    results = {}

    def _run_one(name, algo):
        t0 = time.perf_counter()
        labels, centroids, history = algo.fit(X)
        elapsed = time.perf_counter() - t0
        return name, dict(labels=labels, centroids=centroids, history=history, time=elapsed)

    if parallel:
        actual_workers = min(workers, len(algorithms))
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_run_one, name, algo): name
                       for name, algo in algorithms.items()}
            for future in concurrent.futures.as_completed(futures):
                name, res = future.result()
                results[name] = res
                if verbose:
                    print(f"    {name} done ({res['time']:.1f}s)  WCSS={res['history'][-1]:.2f}")
        # Re-order to match algorithm dict order
        results = {name: results[name] for name in algorithms}
    else:
        for name, algo in algorithms.items():
            if verbose:
                print(f"    {name}...", end=' ', flush=True)
            _, res = _run_one(name, algo)
            results[name] = res
            if verbose:
                print(f"done ({res['time']:.1f}s)  WCSS={res['history'][-1]:.2f}")

    rows = []
    for name, res in results.items():
        m = compute_all_metrics(X, res['labels'], res['centroids'], y_true=y_true)
        m['algorithm'] = name
        m['time_s'] = res['time']
        rows.append(m)
    metrics_df = pd.DataFrame(rows).set_index('algorithm')
    return results, metrics_df


def _aggregate_runs(dfs):
    """Mean and std across multiple seed DataFrames."""
    numeric_cols = [c for c in ['wcss', 'silhouette', 'davies_bouldin',
                                 'calinski_harabasz', 'accuracy', 'time_s']
                    if c in dfs[0].columns]
    stacked = pd.concat(dfs)
    mean_df = stacked.groupby(level=0)[numeric_cols].mean()
    std_df  = stacked.groupby(level=0)[numeric_cols].std(ddof=1).fillna(0.0)
    return mean_df, std_df


def _print_single_table(metrics_df):
    print("\n" + "=" * 100)
    print("CLUSTERING PERFORMANCE COMPARISON")
    print("=" * 100)
    cols = [c for c in ['wcss', 'silhouette', 'davies_bouldin',
                         'calinski_harabasz', 'accuracy', 'time_s']
            if c in metrics_df.columns]
    print(metrics_df[cols].round(4).to_string())
    print("=" * 100)
    best_wcss = metrics_df['wcss'].idxmin()
    best_sil  = metrics_df['silhouette'].idxmax()
    print(f"\nBest by WCSS:       {best_wcss}  ({metrics_df.loc[best_wcss,'wcss']:.2f})")
    print(f"Best by Silhouette: {best_sil}  ({metrics_df.loc[best_sil,'silhouette']:.4f})")
    if 'accuracy' in metrics_df.columns and metrics_df['accuracy'].notna().any():
        best_acc = metrics_df['accuracy'].idxmax()
        print(f"Best by Accuracy:   {best_acc}  ({metrics_df.loc[best_acc,'accuracy']:.4f})")


def _print_multi_table(mean_df, std_df, num_seeds, seeds):
    SEP = "=" * 125
    print(f"\n{SEP}")
    print(f"CLUSTERING PERFORMANCE  —  AVERAGE OVER {num_seeds} SEEDS  ({seeds[0]} – {seeds[-1]})")
    print(SEP)

    cols = [c for c in ['wcss', 'silhouette', 'davies_bouldin',
                         'calinski_harabasz', 'accuracy', 'time_s']
            if c in mean_df.columns]

    rows = {}
    for name in mean_df.index:
        row = {}
        for col in cols:
            m = mean_df.loc[name, col]
            s = std_df.loc[name, col]
            if pd.isna(m):
                row[col] = "N/A"
            else:
                row[col] = f"{m:.4f} ± {s:.4f}"
        rows[name] = row

    display_df = pd.DataFrame(rows).T
    display_df.index.name = 'algorithm'
    print(display_df.to_string())
    print(SEP)

    best_wcss = mean_df['wcss'].idxmin()
    best_sil  = mean_df['silhouette'].idxmax()
    print(f"\nBest mean WCSS:       {best_wcss}  "
          f"({mean_df.loc[best_wcss,'wcss']:.4f} ± {std_df.loc[best_wcss,'wcss']:.4f})")
    print(f"Best mean Silhouette: {best_sil}  "
          f"({mean_df.loc[best_sil,'silhouette']:.4f} ± {std_df.loc[best_sil,'silhouette']:.4f})")
    if 'accuracy' in mean_df.columns and mean_df['accuracy'].notna().any():
        best_acc = mean_df['accuracy'].idxmax()
        print(f"Best mean Accuracy:   {best_acc}  "
              f"({mean_df.loc[best_acc,'accuracy']:.4f} ± {std_df.loc[best_acc,'accuracy']:.4f})")


def _save_figures(X, results, metrics_df, feature_names, scaler, k):
    print("\nGenerating visualisations...")
    from utils.visualization import (plot_pca_scatter_grid, plot_convergence_curves,
                                     plot_metric_bars, plot_cluster_sizes,
                                     plot_cluster_profiles, plot_metrics_heatmap)

    labels_dict    = {n: r['labels']    for n, r in results.items()}
    centroids_dict = {n: r['centroids'] for n, r in results.items()}
    histories      = {n: r['history']   for n, r in results.items()}
    best_wcss      = metrics_df['wcss'].idxmin()

    plot_pca_scatter_grid(X, labels_dict, centroids_dict, metrics_df, save_dir='figures')
    plot_convergence_curves(histories, save_dir='figures')
    plot_metric_bars(metrics_df, save_dir='figures')
    plot_cluster_sizes(labels_dict, k, save_dir='figures')
    plot_cluster_profiles(X, results[best_wcss]['labels'], results[best_wcss]['centroids'],
                          feature_names, scaler, save_dir='figures')
    plot_metrics_heatmap(metrics_df, save_dir='figures')


def _run_param_sweep(args):
    """Parameter sensitivity analysis: vary one parameter, run a representative subset."""
    import matplotlib  # noqa: F401

    from utils.preprocessing import load_and_preprocess
    from utils.visualization import plot_param_sweep

    os.makedirs('figures', exist_ok=True)

    use_gpu = not args.no_gpu
    from utils.gpu import GPU_AVAILABLE
    if use_gpu and not GPU_AVAILABLE:
        use_gpu = False

    print("Loading and preprocessing data...")
    X, feature_names, scaler, y_true, target_name = load_and_preprocess(args.data)
    print(f"Ready: {X.shape[0]} samples × {X.shape[1]} features")

    param = args.sweep_param
    default_values = {'k': [2, 3, 4, 5, 6],
                      'pop_size': [10, 20, 30, 50],
                      'mutation_rate': [0.01, 0.05, 0.10, 0.20, 0.30]}
    values = [int(v) if param in ('k', 'pop_size') else v
              for v in (args.sweep_values or default_values[param])]

    print(f"\nParameter sweep: {param} over {values}")
    print("Algorithms: Hybrid GA-PSO, PSO, GA, MMAS, ICA, K-means, K-means++\n")

    sweep_results = []
    for val in values:
        k        = int(val) if param == 'k'        else args.k
        pop_size = int(val) if param == 'pop_size' else 20
        mut_rate = float(val) if param == 'mutation_rate' else 0.10

        print(f"  {param}={val}: ", end='', flush=True)

        from algorithms.genetic_algorithm import GeneticAlgorithm
        from algorithms.pso import ParticleSwarmOptimization
        from algorithms.hybrid_ga_pso import HybridGAPSO
        from algorithms.ant_colony import MAXMINAntSystem
        from algorithms.imperialist_competitive import ImperialistCompetitiveAlgorithm
        from baselines.traditional import KMeansBaseline, KMeansPlusPlusBaseline
        from utils.evaluation import compute_all_metrics
        import time

        common = dict(n_clusters=k, max_iter=100, pop_size=pop_size,
                      random_state=args.seed, use_gpu=use_gpu)
        algorithms = {
            'Hybrid GA-PSO': HybridGAPSO(**common),
            'PSO':           ParticleSwarmOptimization(**common),
            'GA':            GeneticAlgorithm(mutation_rate=mut_rate, **common),
            'MMAS':          MAXMINAntSystem(**common),
            'ICA':           ImperialistCompetitiveAlgorithm(**common),
            'K-means':       KMeansBaseline(**common),
            'K-means++':     KMeansPlusPlusBaseline(**common),
        }

        rows = []
        for name, algo in algorithms.items():
            labels, centroids, _ = algo.fit(X)
            m = compute_all_metrics(X, labels, centroids, y_true=y_true)
            m['algorithm'] = name
            rows.append(m)
            print(f"{name}({m['wcss']:.0f}) ", end='', flush=True)
        print()

        import pandas as pd
        mdf = pd.DataFrame(rows).set_index('algorithm')
        sweep_results.append(mdf)

    plot_param_sweep(sweep_results, param, values, save_dir='figures')
    print(f"\nSweep figure saved: figures/param_sweep_{param}.png")


def main():
    parser = argparse.ArgumentParser(description='Customer Segmentation with CI/EC Algorithms')
    parser.add_argument('--data', default='data/marketing_campaign.csv')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Number of random seeds to run and average over (default: 1)')
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--pop-size', type=int, default=30)
    parser.add_argument('--no-gpu',   action='store_true',
                        help='Force CPU mode (disable GPU acceleration)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run algorithms in parallel using multiple threads')
    parser.add_argument('--workers',  type=int, default=3,
                        help='Number of parallel worker threads (default: 3)')
    # Parameter sweep
    parser.add_argument('--param-sweep', action='store_true',
                        help='Run parameter sensitivity analysis instead of standard run')
    parser.add_argument('--sweep-param', default='k',
                        choices=['k', 'pop_size', 'mutation_rate'],
                        help='Parameter to sweep (default: k)')
    parser.add_argument('--sweep-values', nargs='+', type=float,
                        metavar='V',
                        help='Values to sweep over (default depends on --sweep-param)')
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
        import seaborn     # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install matplotlib seaborn")
        sys.exit(1)

    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        print("\nDownload it with:")
        print("  pip install kaggle")
        print("  kaggle datasets download -d imakash3011/customer-personality-analysis -p data/ --unzip")
        print("\nOr place marketing_campaign.csv in the data/ folder manually.")
        sys.exit(1)

    if args.param_sweep:
        _run_param_sweep(args)
        return

    use_gpu = not args.no_gpu
    from utils.gpu import GPU_AVAILABLE
    if use_gpu and not GPU_AVAILABLE:
        print("No GPU detected — running on CPU.")
        use_gpu = False
    elif use_gpu:
        from utils.gpu import get_gpu_info
        info = get_gpu_info()
        print(f"GPU: {info['name']} ({info['memory_gb']} GB VRAM)")
    else:
        print("GPU disabled — running on CPU.")

    if args.parallel:
        print(f"Parallel mode: {args.workers} worker threads.")

    print("Loading and preprocessing data...")
    from utils.preprocessing import load_and_preprocess
    X, feature_names, scaler, y_true, target_name = load_and_preprocess(args.data)
    print(f"Ready: {X.shape[0]} samples x {X.shape[1]} features, k={args.k}")
    if target_name:
        print(f"Reference labels: '{target_name}' (accuracy will be computed)")

    num_seeds = max(1, args.num_seeds)
    seeds = list(range(args.seed, args.seed + num_seeds))

    run_kwargs = dict(use_gpu=use_gpu, parallel=args.parallel, workers=args.workers)

    if num_seeds == 1:
        # ── Single seed: original behaviour ──────────────────────────────────
        print(f"\nRunning algorithms (seed={args.seed}):")
        results, metrics_df = _run_one_seed(
            X, args.k, args.max_iter, args.pop_size, args.seed, y_true,
            verbose=True, **run_kwargs)
        _print_single_table(metrics_df)
        _save_figures(X, results, metrics_df, feature_names, scaler, args.k)

    else:
        # ── Multi-seed: collect + aggregate ──────────────────────────────────
        print(f"\nRunning algorithms with {num_seeds} seeds ({seeds[0]} – {seeds[-1]})…")

        all_dfs = []
        best_results_per_seed = []

        for i, seed in enumerate(seeds):
            print(f"\n  [Seed {seed}  ({i+1}/{num_seeds})]")
            results, mdf = _run_one_seed(
                X, args.k, args.max_iter, args.pop_size, seed, y_true,
                verbose=True, **run_kwargs)
            all_dfs.append(mdf)
            best_results_per_seed.append((seed, results, mdf))

        mean_df, std_df = _aggregate_runs(all_dfs)
        _print_multi_table(mean_df, std_df, num_seeds, seeds)

        # Figures from the seed whose best algorithm had the lowest WCSS
        best_seed_idx = int(np.argmin([mdf['wcss'].min() for _, _, mdf in best_results_per_seed]))
        best_seed, best_results, best_mdf = best_results_per_seed[best_seed_idx]
        print(f"\nSaving figures from best seed ({best_seed})…")
        _save_figures(X, best_results, best_mdf, feature_names, scaler, args.k)

    print("\nAll figures saved to figures/")
    print("Done.")


if __name__ == '__main__':
    main()
