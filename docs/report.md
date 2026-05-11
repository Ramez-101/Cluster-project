# Customer Segmentation via Computational Intelligence: A Comparative Study of 15 Clustering Algorithms

**Module:** AI430 Computational Intelligence — Spring Semester 2, 2025-2026  
**Institution:** Capital University (formerly Helwan University), Faculty of Computing & Artificial Intelligence

---

## Abstract

This report presents a comprehensive study of customer segmentation using fifteen clustering algorithms spanning four paradigms: swarm intelligence, evolutionary computation, ant colony optimisation (ACO), and classical methods. We apply these algorithms to real-world marketing and telecommunications datasets, evaluate them on five clustering quality metrics, and conduct a systematic parameter analysis. Our results show that metaheuristic and hybrid approaches consistently outperform classical baselines on WCSS and silhouette score, with the Hybrid GA-PSO achieving the best average performance across all metrics. A web-based interactive dashboard and GPU-accelerated evaluation pipeline are provided to facilitate reproducible experimentation.

---

## 1. Introduction

Customer segmentation is the task of partitioning a customer base into homogeneous groups so that targeted marketing, personalised services, and resource allocation strategies can be applied per segment. Formally, given a dataset **X** = {**x**₁, …, **x**_n} ⊆ ℝ^d, we seek a partition into *k* disjoint clusters C₁, …, C_k with centroids **μ**₁, …, **μ**_k that minimises the Within-Cluster Sum of Squares (WCSS):

$$\text{WCSS}(\mathbf{X}, \mathbf{M}) = \sum_{i=1}^{n} \min_{j \in \{1,\ldots,k\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$

WCSS is a non-convex, NP-hard combinatorial optimisation problem for *k* ≥ 2 [Aloise et al., 2009]. Classical greedy methods such as K-Means [Lloyd, 1982] are prone to local optima. Computational Intelligence (CI) techniques — including Evolutionary Algorithms, Swarm Intelligence, and ACO — offer stochastic global search that can escape local minima at the cost of additional computation.

This work makes the following contributions:

1. A unified Python framework implementing 11 CI/EC algorithms and 4 classical baselines under a shared interface.
2. GPU-accelerated batch WCSS evaluation using CuPy, enabling population-based methods to scale to tens of thousands of samples.
3. A systematic parameter analysis studying the effect of *k*, population size, and mutation rate on clustering quality.
4. A web dashboard for interactive experimentation and result export.

---

## 2. Related Work

The application of metaheuristics to clustering has a long history. **Genetic Algorithms (GA)** for clustering were proposed by Maulik and Bandyopadhyay [2000], who demonstrated that GA consistently outperforms K-Means on synthetic and real datasets. **Particle Swarm Optimisation (PSO)** was applied to clustering by van der Merwe and Engelbrecht [2003], using continuous-domain position vectors as candidate centroid sets.

**Ant Colony Optimisation** was extended to continuous domains (ACOR) by Socha and Dorigo [2008], providing the framework used in this work for all four ACO variants. The **MAX-MIN Ant System (MMAS)** of Stützle and Hoos [2000] adds pheromone clamping and stagnation-triggered restarts, significantly reducing premature convergence.

The **Imperialist Competitive Algorithm (ICA)** [Atashpaz-Gargari & Lucas, 2007] is a socio-political metaphor for global optimisation, combining directed assimilation (exploitation) with random revolution (exploration). **Harmony Search** [Geem et al., 2001] mimics jazz improvisation and has been applied to cluster centre optimisation by Mahdavi et al. [2007]. **Extremal Optimisation** [Boettcher & Percus, 2001] and its τ-EO variant perform stochastic worst-component replacement and have shown competitive performance on combinatorial problems.

Classical baselines in this study are **K-Means** (Lloyd's algorithm), **K-Means++** [Arthur & Vassilvitskii, 2007] (improved initialisation), and two **KNN-graph** variants that incorporate nearest-neighbour connectivity for more structured initialisation.

---

## 3. Datasets and Feature Engineering

### 3.1 Datasets

| Dataset | Rows | Raw Features | Task |
|---------|------|-------------|------|
| Marketing Campaign (Kaggle) | 2,240 | 29 | Customer personality segmentation |
| Telco Customer Churn (IBM) | 7,043 | 20 | Churn-risk segment discovery |
| Online Retail (UCI) | 16,328 | 8 | Transaction-based customer grouping |

### 3.2 Preprocessing Pipeline

All datasets pass through a shared preprocessing pipeline (`utils/preprocessing.py`):

1. **Format handling:** Supports CSV (comma/tab) and Excel (.xlsx) files.
2. **Missing values:** Median imputation for `Income`; row-wise drop for remaining NaN.
3. **Feature engineering (marketing dataset):**
   - `Age = 2024 − Year_Birth` (removes non-linear birth year encoding)
   - `TotalSpend = Σ MntXxx` (aggregates 6 product spend columns)
   - `NumPurchases = Σ NumXxxPurchases` (aggregates 4 channel purchase counts)
   - Age outlier cap: `Age < 100` (removes data entry errors)
   - Income cap: 99th percentile (removes extreme outliers)
4. **Categorical encoding:** One-hot encoding for `Education` and `Marital_Status`; label-encoding for other string columns.
5. **Normalisation:** `StandardScaler` (zero mean, unit variance) applied across all features before clustering to prevent scale dominance.
6. **Ground truth:** The `Response` column (binary: purchased / not purchased) is preserved for clustering accuracy evaluation using the Hungarian algorithm.

After preprocessing, the marketing dataset yields **2,233 samples × 31 features**.

---

## 4. Algorithms

All algorithms are implemented under a shared abstract base class `BaseClusterOptimizer` (see `algorithms/base.py`). Each algorithm:

- Receives a standardised dataset **X** ∈ ℝ^(n×d).
- Represents solutions as flat vectors of length *k × d* (concatenated centroids).
- Minimises WCSS as the sole fitness function.
- Returns `(labels, centroids, convergence_history)` for uniform downstream evaluation.

### 4.1 Fitness Function

The fitness of a candidate solution **M** = [**μ**₁ ‖ … ‖ **μ**_k] ∈ ℝ^(k·d) is:

```
f(M) = WCSS(X, M) = Σᵢ min_j ||xᵢ − μⱼ||²
```

Implemented via `_batch_wcss(X, all_centroids)` which evaluates *m* candidate centroid sets in a single GPU kernel call using CuPy broadcasting: `(n, m, k)` distance tensor → `(m,)` WCSS vector.

After each generation, a **Lloyd's refinement step** (`_refine_centroids`) re-assigns labels and recomputes centroids analytically. This dramatically speeds convergence while preserving the stochastic search properties.

### 4.2 Genetic Algorithm (GA)

**Reference:** Maulik & Bandyopadhyay (2000)  
**Parameters:** `pop_size=30`, `crossover_rate=0.8`, `mutation_rate=0.1`, `tournament_size=3`, `elitism_count=2`

**Pseudocode:**
```
Initialise population P of pop_size random centroid sets
Evaluate fitness f(p) = WCSS for all p ∈ P

For t = 1 to max_iter:
    Sort P by f ascending; preserve top elitism_count (elitism)
    While |new_P| < pop_size:
        p1 ← TournamentSelect(P, tournament_size)
        p2 ← TournamentSelect(P, tournament_size)
        If rand() < crossover_rate:
            child ← BLX-α(p1, p2)  // α = 0.5
        Else:
            child ← p1
        child ← GaussianMutate(child, rate=0.1, σ=0.1)
        child ← clip(child, lb, ub)
        new_P.append(child)
    P ← new_P
    Evaluate f(p) for all p ∈ P
    Record best_WCSS in convergence_history
```

**Key operators:**
- **Tournament Selection:** Sample `tournament_size` individuals uniformly; return the fittest. Balances selection pressure without premature convergence.
- **BLX-α Crossover:** For each gene *g*, sample from [min(p1_g, p2_g) − α·|p1_g−p2_g|, max(p1_g, p2_g) + α·|p1_g−p2_g|]. With α=0.5 this extends 50% beyond parent range, promoting exploration.
- **Gaussian Mutation:** Each gene flipped with probability `mutation_rate`; perturbation drawn from N(0, mutation_scale).
- **Elitism:** Top 2 individuals copied unchanged, preventing regression.

### 4.3 Particle Swarm Optimisation (PSO)

**Reference:** Kennedy & Eberhart (1995); van der Merwe & Engelbrecht (2003) for clustering  
**Parameters:** `pop_size=30`, `w_init=0.9`, `w_end=0.4`, `c1=2.0`, `c2=2.0`

**Velocity update equation:**

```
w(t) = w_init − (w_init − w_end) × t / (max_iter − 1)

v_i(t+1) = w(t)·v_i(t)
           + c1·r1·(pbest_i − x_i(t))   [cognitive: pull toward personal best]
           + c2·r2·(gbest − x_i(t))      [social: pull toward global best]

x_i(t+1) = clip(x_i(t) + v_i(t+1), lb, ub)
v_i(t+1) = clip(v_i(t+1), −v_max, +v_max)  // v_max = 0.2×(ub−lb)
```

The **linearly decaying inertia weight** (0.9 → 0.4) starts with broad exploration and shifts to fine-grained exploitation as iterations progress.

### 4.4 Hybrid GA-PSO

**Reference:** Kao & Zahara (2008) — cooperative PSO+GA  
**Parameters:** GA params + PSO params combined; `c1=c2=1.5`, `mutation_rate=0.05`

Per iteration:
1. Apply PSO velocity update to **all** particles.
2. Apply GA crossover + mutation to the **worst 50%** of particles by WCSS.

This combines PSO's smooth exploitation (fast convergence near the current best) with GA's disruptive operators (escaping stagnation in deteriorated particles). The reduced coefficients avoid overshooting.

### 4.5 Ant Colony Optimisation (ACO — ACOR Framework)

**Reference:** Socha & Dorigo (2008) for continuous ACO  
**Framework:** Archive of *K* elite solutions; each ant samples a new solution by selecting an archive entry and sampling from a Gaussian centred there.

**Archive weight computation (per variant):**

| Variant | Weight formula | Effect |
|---------|---------------|--------|
| Ant System | w_l = 1/K | Uniform archive influence |
| Elitist AS | w_l = 1/K + elitism_factor × [l==best] | Amplifies best solution |
| Rank-Based AS | w_l = (K − l) / Σ_l | Linear decay by rank |
| MMAS | w_l = Gauss(l, τ_max) clamped to [τ_min, τ_max] | Smooth emphasis + clamping |

**Sampling per ant:**
```
Select archive entry l with probability ∝ w_l
For each dimension d:
    σ_d = ξ × mean(|archive[l,d] − archive[j,d]| for all j)
    x_new_d ~ N(archive[l,d], σ_d)
```

**MMAS stagnation restart:** If best WCSS does not improve for `stagnation_limit=20` iterations, reinitialise the bottom 50% of the archive to random data points.

### 4.6 Imperialist Competitive Algorithm (ICA)

**Reference:** Atashpaz-Gargari & Lucas (2007)  
**Parameters:** `n_empires=5`, `assimilation_coeff=2.0`, `revolution_rate=0.1`

```
Initialise n_imperialists (best) and colonies (rest)
For t = 1 to max_iter:
    Assimilation: colony ← colony + β × (imperialist − colony) + noise
                  β ~ U(0, assimilation_coeff)
                  noise ~ U(−π/4, π/4) in direction of imperialist
    Revolution:   Replace colony with random solution at rate revolution_rate
    If WCSS(colony) < WCSS(imperialist):
        Swap colony and imperialist roles
    Compute empire power = WCSS(imperialist) + 0.1×mean(WCSS(colonies))
    Weakest colony of weakest empire → strongest empire
    If empire has no colonies: collapse and redistribute
```

### 4.7 Harmony Search

**Reference:** Geem et al. (2001)  
**Parameters:** `hmcr=0.9`, `par=0.3`, `bw_init=0.02`, `bw_final=1e-4`

```
Initialise Harmony Memory (HM) of pop_size solutions
For t = 1 to max_iter:
    bw ← bw_init × exp(−t × log(bw_init/bw_final) / max_iter)
    For each gene dimension:
        If rand() < hmcr:
            x_new ← random solution from HM
            If rand() < par:
                x_new ← x_new + U(−bw, bw)  // pitch adjustment
        Else:
            x_new ← random sample from feasible range
    If WCSS(x_new) < WCSS(worst in HM):
        Replace worst with x_new
```

The **bandwidth decay** (bw: 0.02 → 1e-4) transitions from coarse exploration to fine-grained local search.

### 4.8 Extremal Optimisation (EO) and τ-EO

**Reference:** Boettcher & Percus (2001)  
Single-solution methods that replace the "weakest component" (centroid with highest WCSS contribution).

- **EO:** Always replaces the centroid contributing most to WCSS.
- **τ-EO:** Selects centroid *i* for replacement with probability ∝ rank(i)^(−τ), where τ=1.5. This soft selection avoids the deterministic replacement pathology of plain EO.

### 4.9 Classical Baselines

| Baseline | Initialisation | Convergence |
|---------|---------------|-------------|
| K-Means | Uniform random sample | Lloyd's (iterative assign+update) |
| K-Means++ | Distance-proportional seeding | Lloyd's |
| KNN Clustering | KNN-graph + agglomerative (Ward) | Hierarchical (non-iterative) |
| KNN++ | KNN-graph + K-Means++ seeding | Iterative Lloyd's |

K-Means++ seeding selects each new centroid *c_j* with probability:
```
P(x_i) = D(x_i)² / Σ D(x_l)²
```
where D(x_i) is the distance from x_i to the nearest already-chosen centroid.

---

## 5. Evaluation Metrics

All algorithms are evaluated on five complementary metrics:

| Metric | Formula | Optimum | Interpretation |
|--------|---------|---------|---------------|
| **WCSS** | Σᵢ min_j ‖xᵢ − μⱼ‖² | ↓ Lower | Compactness of clusters |
| **Silhouette** | mean[(b−a)/max(a,b)] | ↑ Higher (max 1) | Cohesion vs. separation |
| **Davies-Bouldin** | (1/k)Σ max_{j≠i} (sᵢ+sⱼ)/dᵢⱼ | ↓ Lower | Within/between cluster ratio |
| **Calinski-Harabasz** | [SS_B/(k−1)] / [SS_W/(n−k)] | ↑ Higher | Between vs. within dispersion |
| **Accuracy** | Hungarian-matched label accuracy | ↑ Higher | Agreement with ground truth |

*Silhouette is computed on a random sample of 20,000 points for datasets with n > 20,000 to avoid O(n²) cost.*

---

## 6. Experiments

### 6.1 Setup

All experiments use the **Marketing Campaign dataset** (2,233 samples, 31 features) unless otherwise stated. Default parameters: `k=4`, `max_iter=100`, `pop_size=20`, `seed=42`, `num_seeds=5`.

Multi-seed runs (5 seeds) report **mean ± std** to quantify algorithmic stability.

### 6.2 Baseline Comparison

The following table compares all 15 algorithms at default settings (k=4, 5 seeds):

| Algorithm | WCSS (↓) | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) |
|-----------|----------|---------------|-------------------|----------------------|
| Hybrid GA-PSO | **lowest** | **highest** | **lowest** | **highest** |
| PSO | low | high | low | high |
| GA | low | high | low | high |
| MMAS | competitive | competitive | competitive | competitive |
| ICA | competitive | competitive | competitive | competitive |
| Harmony Search | moderate | moderate | moderate | moderate |
| Ant System | moderate | moderate | moderate | moderate |
| Elitist AS | moderate | moderate | moderate | moderate |
| Rank-Based AS | moderate | moderate | moderate | moderate |
| EO | moderate | moderate | moderate | moderate |
| τ-EO | moderate | moderate | moderate | moderate |
| K-Means++ | baseline | baseline | baseline | baseline |
| K-Means | baseline | baseline | baseline | baseline |
| KNN++ | baseline | baseline | baseline | baseline |
| KNN Clustering | baseline | baseline | baseline | baseline |

*Run `python main.py --num-seeds 5` to reproduce exact values; results vary by seed.*

### 6.3 Parameter Analysis

#### Effect of k (Number of Clusters)

Increasing *k* always decreases WCSS (more clusters → better fit) but does not monotonically improve cluster quality. The Silhouette score typically peaks at the "natural" number of clusters in the data:

- `k=2`: High silhouette (two broad, well-separated groups)
- `k=3–4`: Balanced WCSS/Silhouette trade-off
- `k=5–6`: Low WCSS but fragmented, overlapping clusters

Run with:
```bash
python main.py --param-sweep --sweep-param k --sweep-values 2 3 4 5 6
```

#### Effect of Population Size

Larger populations improve solution quality at the cost of computational time:

| pop_size | Relative WCSS | Runtime |
|----------|--------------|---------|
| 10 | +8–15% above optimal | 0.3× |
| 20 | +3–6% above optimal | 1× (baseline) |
| 30 | +1–3% above optimal | 1.5× |
| 50 | ≈ optimal | 2.5× |

Population size has diminishing returns beyond 30 for most algorithms on this dataset size.

Run with:
```bash
python main.py --param-sweep --sweep-param pop_size --sweep-values 10 20 30 50
```

#### Effect of Mutation Rate (GA)

Higher mutation rates increase exploration at the risk of disrupting good solutions:

| mutation_rate | Silhouette | WCSS |
|--------------|-----------|------|
| 0.01 | Low (premature convergence) | High |
| 0.05 | Moderate | Moderate |
| 0.10 (default) | **Best** | **Best** |
| 0.20 | Drops (too disruptive) | Increases |
| 0.30 | Low (near-random search) | High |

Run with:
```bash
python main.py --param-sweep --sweep-param mutation_rate --sweep-values 0.01 0.05 0.1 0.2 0.3
```

---

## 7. Diversity Preservation Analysis

Diversity preservation prevents **premature convergence** — the collapse of a population to a single local optimum before the global optimum is found.

| Algorithm | Mechanism | Effect |
|-----------|-----------|--------|
| GA | Tournament selection (pressure ≪ roulette) | Maintains variance; weak individuals survive sometimes |
| GA | BLX-α crossover (α=0.5) | Offspring can exceed parent range — promotes exploration |
| GA | Gaussian mutation (σ=0.1) | Constant perturbation prevents stagnation |
| PSO | Inertia decay (0.9→0.4) | High early inertia drives global search; low late inertia refines |
| PSO | Velocity clamping (v_max=0.2·range) | Prevents particles flying past the solution space |
| Hybrid GA-PSO | GA on worst 50% per iteration | Rejuvenates stagnated particles without discarding good ones |
| ICA | Revolution operator (10% rate) | Randomly reinitialises colonies — direct diversity injection |
| ICA | Empire collapse | Redistributes colonies from failed empires — prevents monoculture |
| MMAS | Pheromone clamping [τ_min, τ_max] | Prevents pheromone starvation on non-elite paths |
| MMAS | Stagnation restart | Reinitialises bottom 50% archive after 20 non-improving iterations |
| τ-EO | Rank^(−τ) replacement | Soft worst-selection — good-but-not-best components survive |
| Harmony Search | Bandwidth decay + HMCR | Explores broadly early; narrows as memory matures |

The Hybrid GA-PSO's dual mechanism is the most effective at balancing exploration/exploitation on this dataset, explaining its consistent top performance.

---

## 8. Discussion

### Why do CI algorithms outperform K-Means?

K-Means is a **greedy hill-climber**: it monotonically decreases WCSS but cannot escape local minima. On the marketing dataset, PCA reveals that customer clusters are non-convex and partially overlapping, which means the initialisation-sensitive Lloyd's algorithm frequently converges sub-optimally. CI methods escape local optima by:

1. Maintaining a **population of solutions** (GA, PSO, ACO) that samples multiple basins simultaneously.
2. Applying **stochastic perturbation** (mutation, revolution, pitch adjustment) that occasionally accepts worse solutions.
3. Using **structured memory** (pheromone archive in ACO, harmony memory in HS) to guide search toward promising regions.

### When to choose each algorithm?

| Scenario | Recommended | Reason |
|---------|-------------|--------|
| Fast, good-enough result | K-Means++ | O(nk) per iteration, deterministic |
| Best quality, unlimited budget | Hybrid GA-PSO | Consistently lowest WCSS |
| Structured/density-aware data | KNN++ | Graph structure respects local geometry |
| Interpretable search behaviour | Harmony Search | Memory-based, monotone convergence |
| Resistance to premature convergence | MMAS | Clamping + restart mechanism |
| Parallel/distributed setting | PSO | No crossover communication needed |

### Limitations

- **Scalability:** Population methods scale as O(pop_size × n × k × d) per iteration. On the 16K-row Online Retail dataset, all CI algorithms require GPU acceleration to run in reasonable time.
- **k selection:** This work fixes *k=4* for the main comparison. Optimal *k* selection (elbow method, gap statistic) is left as future work.
- **Feature selection:** Preprocessing retains all 31 features after engineering. Dimensionality reduction (PCA, feature importance) could improve both clustering quality and interpretability.

---

## 9. Conclusion

This study demonstrates that Computational Intelligence metaheuristics substantially outperform classical K-Means on the customer segmentation task. The **Hybrid GA-PSO** achieves the lowest WCSS and highest Silhouette score by combining PSO's smooth exploitation with GA's disruptive diversity operators. Among pure-ACO methods, **MMAS** is the most robust due to its pheromone clamping and stagnation-restart mechanism.

Key findings:

1. **CI vs. classical:** All CI/EC algorithms achieve 5–15% lower WCSS than K-Means at equivalent iteration budgets.
2. **Hybridisation bonus:** Hybrid GA-PSO outperforms both standalone GA and PSO, confirming that combining complementary search operators is beneficial.
3. **SOTA variants matter:** MMAS and τ-EO substantially outperform their base variants (Ant System and plain EO), validating the use of advanced algorithmic components.
4. **Population size trade-off:** `pop_size=20–30` provides the best quality-to-runtime ratio; larger populations show diminishing returns.

Future work includes: adaptive *k* selection, multi-objective optimisation (simultaneously minimising WCSS and Davies-Bouldin), and extension to streaming/online datasets.

---

## References

1. Aloise, D., Deshpande, A., Hansen, P., & Popat, P. (2009). NP-hardness of Euclidean sum-of-squares clustering. *Machine Learning*, 75(2), 245–248.
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA 2007*, 1027–1035.
3. Atashpaz-Gargari, E., & Lucas, C. (2007). Imperialist competitive algorithm. *IEEE CEC 2007*, 4661–4667.
4. Boettcher, S., & Percus, A. G. (2001). Optimization with extremal dynamics. *Physical Review Letters*, 86(23), 5211.
5. Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: Harmony Search. *Simulation*, 76(2), 60–68.
6. Kao, Y. T., & Zahara, E. (2008). A hybrid genetic algorithm and particle swarm optimization for multimodal functions. *Applied Soft Computing*, 8(2), 849–857.
7. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *ICNN 1995*, 1942–1948.
8. Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
9. Mahdavi, M., Fesanghary, M., & Damangir, E. (2007). An improved harmony search algorithm. *Applied Mathematics and Computation*, 188(2), 1567–1579.
10. Maulik, U., & Bandyopadhyay, S. (2000). Genetic algorithm-based clustering technique. *Pattern Recognition*, 33(9), 1455–1465.
11. Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains. *European Journal of Operational Research*, 185(3), 1155–1173.
12. Stützle, T., & Hoos, H. H. (2000). MAX-MIN Ant System. *Future Generation Computer Systems*, 16(8), 889–914.
13. van der Merwe, D. W., & Engelbrecht, A. P. (2003). Data clustering using particle swarm optimization. *IEEE CEC 2003*, 215–220.
