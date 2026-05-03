"""
Customer Segmentation with CI/EC Algorithms
Usage: python main.py [--data data/marketing_campaign.csv] [--k 4] [--seed 42]
                      [--max-iter 200] [--pop-size 30]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Customer Segmentation with CI/EC Algorithms')
    parser.add_argument('--data', default='data/marketing_campaign.csv')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--pop-size', type=int, default=30)
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

    print("Loading and preprocessing data...")
    from utils.preprocessing import load_and_preprocess
    X, feature_names, scaler = load_and_preprocess(args.data)
    print(f"Ready: {X.shape[0]} samples x {X.shape[1]} features, k={args.k}")

    common = dict(n_clusters=args.k, max_iter=args.max_iter,
                  pop_size=args.pop_size, random_state=args.seed)

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

    algorithms = {
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

    print("\nRunning algorithms:")
    results = {}
    for name, algo in algorithms.items():
        print(f"  {name}...", end=' ', flush=True)
        t0 = time.perf_counter()
        labels, centroids, history = algo.fit(X)
        elapsed = time.perf_counter() - t0
        results[name] = dict(labels=labels, centroids=centroids,
                             history=history, time=elapsed)
        print(f"done ({elapsed:.1f}s)  WCSS={history[-1]:.2f}")

    from utils.evaluation import compute_all_metrics
    rows = []
    for name, res in results.items():
        m = compute_all_metrics(X, res['labels'], res['centroids'])
        m['algorithm'] = name
        m['time_s'] = res['time']
        rows.append(m)
    metrics_df = pd.DataFrame(rows).set_index('algorithm')

    print("\n" + "=" * 95)
    print("CLUSTERING PERFORMANCE COMPARISON")
    print("=" * 95)
    cols = [c for c in ['wcss', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'time_s']
            if c in metrics_df.columns]
    print(metrics_df[cols].round(4).to_string())
    print("=" * 95)

    best_wcss = metrics_df['wcss'].idxmin()
    best_sil = metrics_df['silhouette'].idxmax()
    print(f"\nBest by WCSS:       {best_wcss}  ({metrics_df.loc[best_wcss,'wcss']:.2f})")
    print(f"Best by Silhouette: {best_sil}  ({metrics_df.loc[best_sil,'silhouette']:.4f})")

    print("\nGenerating visualisations...")
    from utils.visualization import (plot_pca_scatter_grid, plot_convergence_curves,
                                     plot_metric_bars, plot_cluster_sizes,
                                     plot_cluster_profiles, plot_metrics_heatmap)

    labels_dict = {n: r['labels'] for n, r in results.items()}
    centroids_dict = {n: r['centroids'] for n, r in results.items()}
    histories = {n: r['history'] for n, r in results.items()}

    plot_pca_scatter_grid(X, labels_dict, centroids_dict, metrics_df, save_dir='figures')
    plot_convergence_curves(histories, save_dir='figures')
    plot_metric_bars(metrics_df, save_dir='figures')
    plot_cluster_sizes(labels_dict, args.k, save_dir='figures')
    plot_cluster_profiles(X, results[best_wcss]['labels'], results[best_wcss]['centroids'],
                          feature_names, scaler, save_dir='figures')
    plot_metrics_heatmap(metrics_df, save_dir='figures')

    print("\nAll figures saved to figures/")
    print("Done.")


if __name__ == '__main__':
    main()
