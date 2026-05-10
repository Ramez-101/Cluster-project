# Clustering-Based Customer Segmentation with CI/EC Algorithms

A complete Python implementation of **15 clustering algorithms** — including 11 Computational Intelligence / Evolutionary Computation (CI/EC) metaheuristics and 4 classical baselines — applied to customer personality segmentation. Includes both a command-line runner and a full interactive web dashboard with GPU acceleration and parallel execution support.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Datasets](#3-datasets)
4. [Setup & Installation](#4-setup--installation)
5. [How to Run](#5-how-to-run)
6. [All 15 Algorithms — Detailed](#6-all-15-algorithms--detailed)
   - [Harmony Search](#61-harmony-search)
   - [Extremal Optimization](#62-extremal-optimization-eo)
   - [τ-Extremal Optimization](#63-τ-extremal-optimization-τ-eo)
   - [Imperialist Competitive Algorithm](#64-imperialist-competitive-algorithm-ica)
   - [Ant System](#65-ant-system-as)
   - [Elitist Ant System](#66-elitist-ant-system-eas)
   - [Rank-Based Ant System](#67-rank-based-ant-system-rbas)
   - [MAX-MIN Ant System](#68-max-min-ant-system-mmas)
   - [Genetic Algorithm](#69-genetic-algorithm-ga)
   - [Particle Swarm Optimization](#610-particle-swarm-optimization-pso)
   - [Hybrid GA-PSO](#611-hybrid-ga-pso)
   - [K-means](#612-k-means)
   - [K-means++](#613-k-means)
   - [KNN Clustering](#614-knn-clustering)
   - [KNN++](#615-knn)
7. [Shared Architecture](#7-shared-architecture)
8. [GPU Acceleration](#8-gpu-acceleration)
9. [Parallel Execution](#9-parallel-execution)
10. [Preprocessing Pipeline](#10-preprocessing-pipeline)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Web Dashboard](#12-web-dashboard)
13. [Output & Visualisations](#13-output--visualisations)
14. [Parameters Reference](#14-parameters-reference)

---

## 1. Project Overview

**Problem:** Businesses need to segment customers into meaningful groups for targeted marketing.

**Approach:** Frame clustering as an optimisation problem — each algorithm searches for the best set of `k` cluster centroids by minimising **Within-Cluster Sum of Squares (WCSS)**. All 15 algorithms share an identical solution representation and fitness function, enabling fair comparison.

**Solution representation:** A flat `numpy` array of shape `(k × d,)` — the concatenated rows of the `k` centroid vectors in `d`-dimensional feature space.

**Fitness function:** `WCSS = Σᵢ min_k ‖xᵢ − cₖ‖²`

**Performance:** Population-based algorithms (PSO, GA, ACO×4, ICA, Hybrid GA-PSO) use a single batched GPU call per iteration to evaluate all `pop_size` candidate centroid sets simultaneously, drastically reducing round-trip overhead vs. individual evaluations.

---

## 2. Project Structure

```
Cluster-project/
│
├── algorithms/                     ← All CI/EC metaheuristic algorithms
│   ├── __init__.py
│   ├── base.py                     ← Abstract base class (shared utilities + _batch_wcss)
│   ├── harmony_search.py           ← Algorithm 1: Harmony Search
│   ├── extremal_optimization.py    ← Algorithms 2 & 3: EO + τ-EO
│   ├── imperialist_competitive.py  ← Algorithm 4: ICA
│   ├── ant_colony.py               ← Algorithms 5–8: AS, EAS, RBAS, MMAS
│   ├── genetic_algorithm.py        ← Algorithm 9: GA
│   ├── pso.py                      ← Algorithm 10: PSO
│   └── hybrid_ga_pso.py            ← Algorithm 11: Hybrid GA-PSO
│
├── baselines/                      ← Classical / baseline algorithms
│   ├── __init__.py
│   └── traditional.py              ← Algorithms 12–15: K-means, K-means++, KNN, KNN++
│
├── utils/                          ← Shared utilities
│   ├── __init__.py
│   ├── preprocessing.py            ← Data loading, cleaning, feature engineering
│   ├── evaluation.py               ← WCSS, Silhouette, Davies-Bouldin, CH score, accuracy
│   ├── gpu.py                      ← CuPy GPU acceleration (cdist + batch_cdist + gpu_info)
│   └── visualization.py            ← All matplotlib figures (6 figure types)
│
├── templates/
│   └── index.html                  ← Web dashboard (Bootstrap 5 + Chart.js, embedded CSS/JS)
│
├── data/
│   ├── marketing_campaign.csv      ← Customer Personality Analysis dataset (~2,240 rows)
│   └── telco_churn.csv             ← Telco Customer Churn dataset (7,032 rows, preprocessed)
│
├── figures/                        ← Auto-created; all saved PNGs go here
│
├── main.py                         ← CLI runner — runs all algorithms, saves figures
├── app.py                          ← Flask web server
└── requirements.txt
```

---

## 3. Datasets

### Marketing Campaign (default)

**Source:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

**File:** `data/marketing_campaign.csv` (tab-delimited, ~2,240 rows)

**Download:**
```bash
pip install kaggle
kaggle datasets download -d imakash3011/customer-personality-analysis -p data/ --unzip
```

**Ground truth:** `Response` column (binary campaign response) — used for accuracy computation.

### Telco Customer Churn (larger dataset)

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**File:** `data/telco_churn.csv` (7,032 rows × 20 numeric columns, preprocessed)

**Preprocessing applied:**
- Dropped `customerID`; coerced `TotalCharges` to numeric (dropped ~11 NaN rows)
- Renamed `Churn` → `Response` (Yes/No → 1/0) for ground-truth accuracy
- Saved as comma-delimited `telco_churn.csv` ready for the dashboard

**Ground truth:** `Response` column (binary churn label).

---

## 4. Setup & Installation

**Requirements:** Python 3.10+

```bash
# Activate the virtual environment (Windows)
.\.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
numpy>=2.4
scipy>=1.17
pandas>=3.0
scikit-learn>=1.8
matplotlib>=3.9
seaborn>=0.13
flask>=3.0
```

**Optional — GPU acceleration (NVIDIA):**
```bash
pip install cupy-cuda12x   # adjust suffix to match your CUDA version
```
GPU acceleration is auto-detected at startup; the app falls back to CPU if CuPy is unavailable.

---

## 5. How to Run

### CLI — Run all algorithms and save figures

```bash
python main.py
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `data/marketing_campaign.csv` | Path to dataset |
| `--k` | `4` | Number of clusters |
| `--max-iter` | `200` | Max iterations per algorithm |
| `--pop-size` | `30` | Population / swarm size |
| `--seed` | `42` | Base random seed |
| `--num-seeds` | `1` | Number of seeds to run and average |
| `--no-gpu` | — | Disable GPU; force CPU mode |
| `--parallel` | — | Run algorithms concurrently (ThreadPoolExecutor) |
| `--workers` | `3` | Number of parallel worker threads |

**Examples:**
```bash
# Basic run
python main.py --k 5 --max-iter 150 --pop-size 25

# Multi-seed average, Telco dataset
python main.py --data data/telco_churn.csv --k 2 --num-seeds 5

# GPU + parallel execution
python main.py --k 4 --parallel --workers 6

# CPU-only
python main.py --no-gpu
```

**CLI output:**
- Per-algorithm runtime and final WCSS printed to console
- Full comparison table printed (WCSS, Silhouette, DB Index, CH Score, Accuracy, Time)
- 6 figure PNGs saved to `figures/`

---

### Web Dashboard

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 6. All 15 Algorithms — Detailed

---

### 6.1 Harmony Search

| | |
|---|---|
| **File** | `algorithms/harmony_search.py` |
| **Class** | `HarmonySearch` |
| **Category** | CI / EC |
| **Type** | Population-based, memory-driven |

**Concept:** Inspired by musical improvisation. A set of "harmonies" (candidate centroid solutions) is stored in a Harmony Memory (HM). At each iteration, a new harmony is "improvised" by combining memory recall, pitch adjustment, and random exploration, then replaces the worst harmony if it is better.

**Key operators:**
- **Memory consideration (HMCR = 0.9):** With 90% probability, each dimension of the new harmony is drawn from an existing HM entry.
- **Pitch adjustment (PAR = 0.3):** A drawn value is perturbed by ±`bw` (bandwidth) with 30% probability.
- **Random selection:** With 10% probability, a completely random value within bounds is chosen.
- **Adaptive bandwidth:** `bw` decays linearly from `0.02 → 1e-4` over iterations, shifting from exploration to exploitation.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `hmcr` | `0.9` | Harmony Memory Consideration Rate |
| `par` | `0.3` | Pitch Adjustment Rate |
| `bw_init` | `0.02` | Initial bandwidth |
| `bw_final` | `1e-4` | Final bandwidth |

---

### 6.2 Extremal Optimization (EO)

| | |
|---|---|
| **File** | `algorithms/extremal_optimization.py` |
| **Class** | `ExtremalOptimization` |
| **Category** | CI / EC |
| **Type** | Single-solution, component-based |

**Concept:** Inspired by self-organised criticality in nature. Maintains one solution. Each iteration identifies which centroid contributes the most WCSS (the "worst" component) and replaces it with a random data point, followed by one Lloyd's refinement step.

**Algorithm steps:**
1. Assign all points to nearest centroid.
2. Compute WCSS contribution of each centroid (its cluster's inertia).
3. Always replace the centroid with the **highest** contribution.
4. Run one Lloyd's step (`_refine_centroids`) for local improvement.
5. Keep the best solution found so far.

**No population — single solution evolves.** Fast per iteration, good for quick escapes from local optima.

---

### 6.3 τ-Extremal Optimization (τ-EO)

| | |
|---|---|
| **File** | `algorithms/extremal_optimization.py` |
| **Class** | `TauExtremalOptimization` |
| **Category** | CI / EC |
| **Type** | Single-solution, rank-based stochastic |

**Concept:** Extends plain EO by introducing a power-law selection probability. Instead of always replacing the worst centroid, centroid `k` is selected for replacement with probability proportional to `rank(k)^(−τ)`, where rank 1 = best (lowest cost) and rank `n_clusters` = worst.

**Effect of τ:**
- High τ → behaves like plain EO (almost always replaces worst)
- Low τ → more random exploration (any centroid can be replaced)
- Typical range: `τ ∈ [1.0, 2.5]`

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tau` | `1.5` | Selection pressure exponent |

---

### 6.4 Imperialist Competitive Algorithm (ICA)

| | |
|---|---|
| **File** | `algorithms/imperialist_competitive.py` |
| **Class** | `ImperialistCompetitiveAlgorithm` |
| **Category** | CI / EC |
| **Type** | Population-based, socio-political metaphor |

**Concept:** Simulates imperialist competition. The population is divided into **imperialists** (elite solutions) and **colonies** (weaker solutions). Colonies are distributed to empires proportional to imperialist power (inverse WCSS). Three mechanisms drive search:

1. **Assimilation:** Each colony moves toward its imperialist with a random step + directional noise (angle ξ = π/4), modelling cultural assimilation.
2. **Revolution:** Each colony is randomly reinitialised with probability `revolution_rate`, modelling social upheaval.
3. **Competition:** The weakest empire's worst colony is stolen by the strongest empire. If an empire loses all colonies, its imperialist becomes a colony too.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_empires` | `5` | Number of imperialists (capped at `pop_size - 1`) |
| `assimilation_coeff` | `2.0` | Step size toward imperialist |
| `revolution_rate` | `0.1` | Probability of colony reset per iteration |

---

### 6.5 Ant System (AS)

| | |
|---|---|
| **File** | `algorithms/ant_colony.py` |
| **Class** | `AntSystem` |
| **Category** | ACO |
| **Type** | Population-based, archive-based continuous ACO (ACOR) |

**Concept:** Adapts the classic Ant System to continuous centroid optimisation using the **ACOR (Ant Colony Optimisation for Continuous Domains)** framework. A sorted archive of `pop_size` solutions acts as the pheromone model. New "ants" sample solutions from a Gaussian kernel:

- Select archive entry `l` with probability `w_l`
- For each dimension `d`: sample from `N(archive[l, d], σ_d)` where `σ_d = ξ × mean(|archive[j,d] − archive[l,d]|)` (floor `1e-6`)

**AS weight rule:** All archive entries receive **equal weight** `w_i = 1/k`. No preference for better solutions — full diversity.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `q` | `0.5` | Locality parameter for Gaussian kernel |
| `xi` | `0.85` | Speed / spread parameter |

---

### 6.6 Elitist Ant System (EAS)

| | |
|---|---|
| **File** | `algorithms/ant_colony.py` |
| **Class** | `ElitistAntSystem` |
| **Category** | ACO |
| **Type** | Population-based, archive-based continuous ACO |

**Concept:** Same ACOR framework as AS, but the **best archive entry** (lowest WCSS, index 0) receives extra pheromone weight proportional to `elitism_factor`. This biases sampling toward the global best solution found so far.

**EAS weight rule:**
```
w[0] += elitism_factor / k
weights = normalise(weights)
```

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `elitism_factor` | `2.0` | Extra weight multiplier for best solution |
| `q` | `0.5` | Locality parameter |
| `xi` | `0.85` | Speed parameter |

---

### 6.7 Rank-Based Ant System (RBAS)

| | |
|---|---|
| **File** | `algorithms/ant_colony.py` |
| **Class** | `RankBasedAntSystem` |
| **Category** | ACO |
| **Type** | Population-based, archive-based continuous ACO |

**Concept:** Same ACOR framework, but weights decay **linearly** with rank. The best solution (rank 1, index 0 in sorted archive) gets the most weight; the worst gets the least.

**RBAS weight rule:**
```
w[i] = (k − i) / (k(k+1)/2)    for i = 0, 1, …, k−1
```
Creates a smooth gradient of influence without the sharp elitist spike of EAS.

---

### 6.8 MAX-MIN Ant System (MMAS)

| | |
|---|---|
| **File** | `algorithms/ant_colony.py` |
| **Class** | `MAXMINAntSystem` |
| **Category** | ACO |
| **Type** | Population-based, archive-based continuous ACO with stagnation restart |

**Concept:** The most sophisticated ACO variant. Uses a Gaussian-shaped weight distribution (`q`-Gaussian kernel) and **clamps pheromone weights** to `[τ_min, τ_max]` to prevent stagnation:

```
τ_max = 0.999
τ_min = τ_max / (2 × k)
```

Additionally includes a **stagnation detection and restart mechanism**: if the best WCSS does not improve for `stagnation_limit` consecutive iterations, the bottom 50% of the archive is reinitialised with fresh random solutions.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tau_max` | `0.999` | Upper pheromone clamp |
| `stagnation_limit` | `20` | Iterations without improvement before restart |

---

### 6.9 Genetic Algorithm (GA)

| | |
|---|---|
| **File** | `algorithms/genetic_algorithm.py` |
| **Class** | `GeneticAlgorithm` |
| **Category** | Evolutionary |
| **Type** | Population-based, generational |

**Concept:** Evolves a population of centroid solutions through selection, crossover, and mutation — mimicking biological evolution.

**Operators:**

| Operator | Implementation |
|----------|----------------|
| **Selection** | Tournament selection (size 3): pick 3 random individuals, take the fittest |
| **Crossover** | BLX-α (Blend Crossover, α=0.5): `child[d] ~ Uniform(min(p1,p2) − 0.5·Δ, max(p1,p2) + 0.5·Δ)` |
| **Mutation** | Gaussian perturbation: `gene += N(0, σ)` applied per gene with probability `mutation_rate` |
| **Elitism** | Top `elitism_count` individuals pass unchanged to the next generation |

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `crossover_rate` | `0.8` | Probability of crossover vs. cloning |
| `mutation_rate` | `0.1` | Per-gene mutation probability |
| `mutation_scale` | `0.1` | Gaussian mutation std dev |
| `tournament_size` | `3` | Number of candidates in tournament |
| `elitism_count` | `2` | Number of elites preserved each generation |

---

### 6.10 Particle Swarm Optimization (PSO)

| | |
|---|---|
| **File** | `algorithms/pso.py` |
| **Class** | `ParticleSwarmOptimization` |
| **Category** | Swarm Intelligence |
| **Type** | Population-based, velocity-driven |

**Concept:** Each particle (candidate centroid set) has a position and a velocity. Particles are attracted toward their personal best position and the global best position found by the swarm.

**Velocity update:**
```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x ← x + v
```

- **Inertia weight `w`** decays linearly from `w_init → w_end` over iterations (early exploration → late exploitation)
- **Velocity clamping:** `v_max = 0.2 × (ub − lb)` per dimension

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `w_init` | `0.9` | Initial inertia weight |
| `w_end` | `0.4` | Final inertia weight |
| `c1` | `2.0` | Cognitive coefficient (personal best attraction) |
| `c2` | `2.0` | Social coefficient (global best attraction) |

---

### 6.11 Hybrid GA-PSO

| | |
|---|---|
| **File** | `algorithms/hybrid_ga_pso.py` |
| **Class** | `HybridGAPSO` |
| **Category** | Hybrid |
| **Type** | Cooperative population-based hybrid |

**Concept:** Combines PSO's exploitation capability with GA's exploration operators in a **cooperative per-iteration scheme**:

1. **PSO phase** (all particles): Apply standard velocity and position update.
2. **GA phase** (worst 50% only): Replace the weakest half of the swarm with GA-generated offspring via BLX-0.5 crossover and Gaussian mutation.

This keeps the swarm converging efficiently while continuously injecting genetic diversity to avoid premature convergence. Each individual carries both a position and a velocity.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `w_init` | `0.9` | Initial PSO inertia |
| `w_end` | `0.4` | Final PSO inertia |
| `c1` | `1.5` | PSO cognitive coefficient (reduced vs. pure PSO) |
| `c2` | `1.5` | PSO social coefficient (reduced vs. pure PSO) |
| `crossover_rate` | `0.8` | GA crossover probability |
| `mutation_rate` | `0.05` | GA mutation rate (lower: PSO handles perturbation) |

---

### 6.12 K-means

| | |
|---|---|
| **File** | `baselines/traditional.py` |
| **Class** | `KMeansBaseline` |
| **Category** | Classical Baseline |
| **Type** | Iterative, Lloyd's algorithm |

**Concept:** Standard Lloyd's algorithm. Centroids are initialised by sampling `k` random data points, then alternates between assigning each point to its nearest centroid and recomputing centroids as cluster means.

**Note:** Implemented manually (not via scikit-learn) so that per-iteration WCSS convergence history is recorded consistently with all other algorithms.

---

### 6.13 K-means++

| | |
|---|---|
| **File** | `baselines/traditional.py` |
| **Class** | `KMeansPlusPlusBaseline` |
| **Category** | Classical Baseline |
| **Type** | Iterative with smart initialisation |

**Concept:** Identical to K-means but uses the **K-means++ initialisation** strategy to select starting centroids. Each successive centroid is chosen with probability proportional to its squared distance from the nearest already-chosen centroid, spreading initial centroids across the data.

**Initialisation:**
```
Pick first center uniformly at random
For each subsequent center:
    d²[i] = min distance² from x[i] to existing centers
    P[i]  = d²[i] / Σd²
    Sample next center ∝ P
```

This typically yields significantly lower final WCSS than random initialisation.

---

### 6.14 KNN Clustering

| | |
|---|---|
| **File** | `baselines/traditional.py` |
| **Class** | `KNNClusteringBaseline` |
| **Category** | Classical Baseline |
| **Type** | Graph-based hierarchical |

**Concept:** Builds a **K-Nearest Neighbour graph** over the data using `sklearn.neighbors.kneighbors_graph`, then applies **Agglomerative Clustering with Ward linkage** constrained to that connectivity structure. Ward linkage minimises within-cluster variance at each merge step.

**Steps:**
1. Compute KNN graph with `n_neighbors=10`
2. Symmetrise: `conn = 0.5 × (conn + connᵀ)`
3. Run `AgglomerativeClustering(linkage='ward', connectivity=conn)`

**Non-iterative** — convergence history is a flat line at the final WCSS.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_neighbors` | `10` | KNN graph connectivity |

---

### 6.15 KNN++

| | |
|---|---|
| **File** | `baselines/traditional.py` |
| **Class** | `KNNPlusPlusBaseline` |
| **Category** | Classical Baseline |
| **Type** | Graph-based hierarchical with adaptive connectivity |

**Concept:** Enhances KNN Clustering with **K-means++ seeding** to determine an adaptive connectivity radius. The K-means++ seeds give a rough estimate of cluster density, which is used to compute `adaptive_k`:

```
adaptive_k = max(5, min(n_neighbors, n_samples / (n_clusters × 5)))
```

The rest follows KNN Clustering (KNN graph → Ward agglomerative). The adaptive `k` ensures that the graph connectivity reflects the true cluster structure rather than a fixed global neighbourhood size.

---

## 7. Shared Architecture

All 15 algorithms inherit from `BaseClusterOptimizer` in `algorithms/base.py`.

### Abstract base class

```python
class BaseClusterOptimizer(ABC):
    def __init__(self, n_clusters, max_iter, pop_size, random_state, use_gpu=True): ...

    @abstractmethod
    def fit(self, X) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Returns: labels, centroids, convergence_history"""
```

### Shared utility methods

| Method | Description |
|--------|-------------|
| `_wcss(X, centroids)` | Single centroid set evaluation — squared distance to nearest centroid, summed |
| `_batch_wcss(X, all_centroids)` | Batch evaluation: `(m, k, d)` → `(m,)` WCSS in one GPU call |
| `_assign_labels(X, centroids)` | Assigns each sample to its nearest centroid |
| `_decode(solution, n_features)` | Reshapes flat `(k×d,)` vector → `(k, d)` centroid matrix |
| `_encode(centroids)` | Flattens `(k, d)` centroid matrix → `(k×d,)` vector |
| `_init_population(X, size)` | Initialises `size` solutions by sampling `k` data points each |
| `_refine_centroids(X, centroids)` | One Lloyd's step; reinitialises empty clusters to random data points |

All randomness uses `np.random.default_rng(random_state)` — fully reproducible and independent per instance.

---

## 8. GPU Acceleration

**File:** `utils/gpu.py`

GPU acceleration is powered by **CuPy** (NVIDIA CUDA). The system auto-detects GPU availability at startup and falls back to CPU (scipy) transparently.

### How it works

Population-based algorithms previously called `_wcss` once per population member per iteration (30 separate evaluations = 30 GPU round-trips). Now `_batch_wcss` stacks all `pop_size` centroid sets into a `(m, k, d)` array and evaluates them in **a single GPU kernel call**, returning `(m,)` WCSS values.

```python
# batch_cdist_sqeuclidean in utils/gpu.py:
# X: (n, d), all_C: (m, k, d) → dists: (n, m, k)
diff  = X_gpu[:, None, None, :] - C_gpu[None, :, :, :]   # broadcast
dists = cp.sum(diff ** 2, axis=3)
return dists.min(axis=2).sum(axis=0)  # (m,) WCSS
```

**VRAM-aware chunking:** For datasets with many rows, the intermediate `(n, m, k, d)` tensor is chunked along `n` to stay within 50% of available VRAM.

### Which algorithms use batch GPU eval

| Algorithm | GPU-batched operations |
|-----------|----------------------|
| PSO | Initial pbest fitness + per-iteration full population |
| GA | Initial population + per-generation offspring |
| ACO ×4 | Initial archive + per-iteration new ants |
| ICA | Initial countries + assimilation + revolution |
| Hybrid GA-PSO | Initial population + PSO phase (GA phase kept sequential) |
| Harmony Search, EO, τ-EO | Per-call `cdist_sqeuclidean` (single centroid set) |

### GPU info API

```bash
GET /api/gpu_info
# → {"available": true, "name": "NVIDIA GeForce RTX 3080 Ti", "memory_gb": 12.0}
```

### Disabling GPU

```bash
# CLI
python main.py --no-gpu

# Web dashboard
# Use the GPU toggle in the sidebar
```

---

## 9. Parallel Execution

All 15 algorithms are **mutually independent** — they can run concurrently. `ThreadPoolExecutor` is used because CuPy releases the GIL on CUDA kernel launches, allowing true overlap on GPU-accelerated algorithms.

### Web dashboard

Enable the **Parallel** toggle in the sidebar and adjust the **Workers** slider (1–12). Each worker thread handles one algorithm; all threads share the same GPU.

### CLI

```bash
python main.py --parallel --workers 6
```

### Determinism

Each `(algo, seed)` pair uses its own `numpy.default_rng(seed)` instance. Parallel execution produces **identical results** to sequential execution — thread scheduling does not affect RNG state.

---

## 10. Preprocessing Pipeline

**File:** `utils/preprocessing.py` — `load_and_preprocess(filepath)`

The pipeline auto-detects the dataset format and extracts a `Response` column as ground truth if present.

### Marketing Campaign dataset

| Step | Action |
|------|--------|
| 1 | Load CSV with `sep='\t'`; fallback to `sep=','` |
| 2 | Extract `Response` column as ground truth; drop `ID`, `Dt_Customer`, `Z_*`, `AcceptedCmp*`, `Complain` |
| 3 | Fill missing `Income` with column median |
| 4 | Engineer `Age`, `TotalSpend`, `NumPurchases` |
| 5 | Remove `Age ≥ 100`; clip `Income` at 99th percentile |
| 6 | One-hot encode `Education` and `Marital_Status` |
| 7 | Cast to `float64`; apply `StandardScaler` |

**Output:** `(X_scaled, feature_names, scaler, y_true, target_name)` — shape `≈ (2215, 20)`.

### Telco Churn dataset

| Step | Action |
|------|--------|
| 1 | Load comma-delimited CSV |
| 2 | `Response` column (0/1) auto-detected as ground truth |
| 3 | Drop non-numeric / ID columns |
| 4 | Fill missing values; cast to `float64`; apply `StandardScaler` |

**Output:** shape `(7032, 19)`.

---

## 11. Evaluation Metrics

**File:** `utils/evaluation.py` — `compute_all_metrics(X, labels, centroids, y_true=None)`

| Metric | Direction | Description |
|--------|-----------|-------------|
| **WCSS** | Lower = better | Primary optimisation target; total squared distance to assigned centroid |
| **Silhouette Score** | Higher = better | `[-1, 1]`; measures how similar a point is to its own cluster vs. neighbouring clusters |
| **Davies-Bouldin Index** | Lower = better | `≥ 0`; ratio of within-cluster scatter to between-cluster separation |
| **Calinski-Harabasz Score** | Higher = better | `≥ 0`; ratio of between-cluster to within-cluster dispersion |
| **Accuracy** | Higher = better | Hungarian-algorithm-matched label accuracy vs. ground truth; only when `y_true` is provided |

If fewer than 2 non-empty clusters are produced, Silhouette / DB / CH are returned as `NaN`. Accuracy is `NaN` when no ground truth is available.

---

## 12. Web Dashboard

**File:** `app.py` + `templates/index.html`

**Start the server:**
```bash
python app.py
# → http://localhost:5000
```

**Flask API endpoints:**

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Main dashboard page |
| `GET` | `/api/gpu_info` | GPU status: `{available, name, memory_gb}` |
| `POST` | `/api/run` | Start a job; body: `{algorithms, k, max_iter, pop_size, seed, num_seeds, dataset, use_gpu, parallel, n_workers}` |
| `GET` | `/api/status/<job_id>` | Poll job progress: `{status, progress, current}` |
| `GET` | `/api/results/<job_id>` | Fetch full results JSON once done |
| `GET` | `/api/export/<job_id>` | Download results as CSV (summary + per-seed breakdown) |

**Dashboard features:**
- Algorithm checkboxes grouped by category with quick-select buttons (All / None / CI/EC / ACO / Classical)
- Dataset picker (dropdown) with a preview modal showing sample rows
- GPU toggle: enable/disable GPU acceleration; badge shows GPU name and VRAM
- Parallel toggle: run algorithms concurrently; workers slider (1–12) controls thread count
- Topbar GPU badge: green (GPU active) or grey (CPU only)
- Live progress bar showing which algorithm is currently running
- 5 summary metric cards (WCSS, Silhouette, DB Index, CH Score, Accuracy) highlighting the best algorithm
- Per-seed state viewer (when `num_seeds > 1`)
- **6 interactive tabs** — see Section 13
- CSV export button (summary + per-seed raw data)

---

## 13. Output & Visualisations

### CLI — Saved to `figures/`

| File | Contents |
|------|----------|
| `pca_scatter_all.png` | 4×4 grid of PCA 2D scatter plots, one per algorithm; centroids marked with ✕ |
| `convergence_curves.png` | WCSS vs. iteration for all metaheuristics; baselines as dashed horizontal lines |
| `metric_comparison.png` | 2×2 bar chart grid, one panel per metric; gold bar = best algorithm |
| `cluster_sizes.png` | Horizontal stacked bars showing cluster-size proportions per algorithm |
| `cluster_profiles.png` | Parallel-coordinates centroid profiles for the best-WCSS algorithm |
| `metrics_heatmap.png` | Normalised heatmap (green = better) across all algorithms and metrics |

### Web Dashboard Tabs

| Tab | Chart type | Contents |
|-----|-----------|----------|
| **Overview** | Table + horizontal bar | Full metrics table with colour-coded best cells; latency comparison |
| **Convergence** | Interactive line chart | WCSS per iteration for all selected algorithms |
| **Metrics** | 4 bar charts | One per metric; gold = winner |
| **Scatter** | Static PNG | PCA 2D scatter grid |
| **Clusters** | Stacked horizontal bar | Cluster size distribution per algorithm |
| **Profiles** | Static PNG | Centroid feature profiles for best algorithm |

### CSV Export

The exported CSV contains two sections:
1. **Summary** — mean ± std per algorithm across all seeds (WCSS, Silhouette, DB Index, CH Score, Accuracy, Time)
2. **Per-seed breakdown** — one row per `(algorithm, seed)` pair (only present when `num_seeds > 1`)

---

## 14. Parameters Reference

| Parameter | CLI flag | Web control | Default | Range |
|-----------|----------|-------------|---------|-------|
| Number of clusters | `--k` | k slider | `4` | 2 – 10 |
| Max iterations | `--max-iter` | Iterations slider | `200` (CLI) / `100` (web) | 10 – 500 |
| Population size | `--pop-size` | Population slider | `30` (CLI) / `20` (web) | 5 – 100 |
| Random seed | `--seed` | Seed input | `42` | any int |
| Number of seeds | `--num-seeds` | Seeds input | `1` | 1 – 20 |
| GPU acceleration | `--no-gpu` | GPU toggle | on (if available) | — |
| Parallel execution | `--parallel` | Parallel toggle | off | — |
| Worker threads | `--workers` | Workers slider | `3` | 1 – 12 |

All algorithms receive the same `n_clusters`, `max_iter`, `pop_size`, and `random_state` to ensure a fair comparison.
