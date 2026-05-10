"""
GPU acceleration utility.
Provides squared-Euclidean distance functions that run on the RTX 3080 Ti
when CuPy is available and fall back to scipy on CPU otherwise.
"""
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()  # raises if no GPU
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def cdist_sqeuclidean(X, C):
    """
    Squared Euclidean distance matrix between rows of X (n×d) and C (k×d).
    Returns numpy (n, k).
    """
    if GPU_AVAILABLE:
        X_gpu = cp.asarray(X, dtype=cp.float32)   # (n, d)
        C_gpu = cp.asarray(C, dtype=cp.float32)   # (k, d)
        diff  = X_gpu[:, None, :] - C_gpu[None, :, :]  # (n, k, d)
        dists = cp.sum(diff ** 2, axis=2)               # (n, k)
        return cp.asnumpy(dists)
    from scipy.spatial.distance import cdist
    return cdist(X, C, 'sqeuclidean')


def batch_cdist_sqeuclidean(X, all_C):
    """
    Batch squared-Euclidean distances using element-wise broadcasting (no cuBLAS).

    X     : numpy (n, d)
    all_C : numpy (m, k, d) — m centroid sets

    Returns numpy (n, m, k).
    """
    if GPU_AVAILABLE:
        X_gpu = cp.asarray(X,     dtype=cp.float32)   # (n, d)
        C_gpu = cp.asarray(all_C, dtype=cp.float32)   # (m, k, d)
        n, d  = X_gpu.shape
        m, k  = C_gpu.shape[:2]

        # Check if full (n, m, k, d) intermediate tensor fits in free VRAM
        mem_needed = n * m * k * d * 4  # bytes (float32)
        free_mem   = cp.cuda.Device(0).mem_info[0]

        if mem_needed < free_mem * 0.5:
            # Full broadcast: (n,1,1,d) - (1,m,k,d) → (n,m,k,d)
            diff  = X_gpu[:, None, None, :] - C_gpu[None, :, :, :]
            dists = cp.sum(diff ** 2, axis=3)   # (n, m, k)
            cp.maximum(dists, 0.0, out=dists)
            return cp.asnumpy(dists)

        # Large dataset: chunk along n to stay within VRAM
        chunk_n = max(1, int(free_mem * 0.4 / (m * k * d * 4)))
        out = np.empty((n, m, k), dtype=np.float32)
        for i in range(0, n, chunk_n):
            sl   = slice(i, min(i + chunk_n, n))
            diff = X_gpu[sl, None, None, :] - C_gpu[None, :, :, :]
            out[i : i + diff.shape[0]] = cp.asnumpy(
                cp.maximum(cp.sum(diff ** 2, axis=3), 0.0)
            )
        return out

    from scipy.spatial.distance import cdist
    m = all_C.shape[0]
    out = np.empty((X.shape[0], m, all_C.shape[1]), dtype=np.float32)
    for j in range(m):
        out[:, j, :] = cdist(X, all_C[j], 'sqeuclidean')
    return out


def get_gpu_info():
    """Return dict: available (bool), name (str), memory_gb (float)."""
    if not GPU_AVAILABLE:
        return {'available': False, 'name': 'None', 'memory_gb': 0.0}
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name  = props['name']
        if isinstance(name, bytes):
            name = name.decode('utf-8').strip('\x00')
        return {'available': True, 'name': name,
                'memory_gb': round(props['totalGlobalMem'] / 1024 ** 3, 1)}
    except Exception:
        return {'available': True, 'name': 'Unknown GPU', 'memory_gb': 0.0}
