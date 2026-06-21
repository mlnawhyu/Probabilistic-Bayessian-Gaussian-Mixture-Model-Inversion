"""
gpu_utils.py — Abstraction layer untuk komputasi GPU/CPU.

Auto-detect CuPy + CUDA. Jika GPU tersedia → operasi besar dilakukan di GPU.
Fallback otomatis ke NumPy/SciPy jika CuPy tidak ada atau CUDA tidak aktif.

Pattern penggunaan:
    from gpu_utils import xp, GPU_AVAILABLE, asnumpy, asgpu, gpu_cholesky, gpu_solve

    # Transfer data ke GPU sekali
    arr_gpu = asgpu(arr_cpu)

    # Lakukan operasi
    res_gpu = gpu_cholesky(arr_gpu)

    # Transfer hasil kembali
    res_cpu = asnumpy(res_gpu)
"""
from __future__ import annotations
import warnings

import numpy as np
import scipy.linalg as scipy_linalg

# ─────────────────────────────────────────────────────────────────────
#                    GPU AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────
GPU_AVAILABLE = False
GPU_DEVICE_NAME = "CPU"
_cupy = None
_cupy_linalg = None

try:
    import cupy as _cupy
    import cupy.linalg as _cupy_linalg
    if _cupy.cuda.is_available():
        try:
            _devid = _cupy.cuda.runtime.getDevice()
            _props = _cupy.cuda.runtime.getDeviceProperties(_devid)
            GPU_DEVICE_NAME = _props["name"].decode() if isinstance(
                _props["name"], bytes) else _props["name"]
            GPU_AVAILABLE = True
            print(f"[gpu_utils] GPU detected: {GPU_DEVICE_NAME}")
        except Exception as e:
            warnings.warn(f"[gpu_utils] CuPy import OK tapi device error: {e}")
            GPU_AVAILABLE = False
    else:
        # CuPy installed but no CUDA runtime
        GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False
except Exception as e:
    warnings.warn(f"[gpu_utils] CuPy import error: {e}")
    GPU_AVAILABLE = False

# Pilih array module
xp = _cupy if GPU_AVAILABLE else np
xpl = _cupy_linalg if GPU_AVAILABLE else np.linalg


def status_msg() -> str:
    if GPU_AVAILABLE:
        return f"GPU MODE: {GPU_DEVICE_NAME} (CuPy {_cupy.__version__})"
    else:
        return "CPU MODE: NumPy/SciPy fallback"


# ─────────────────────────────────────────────────────────────────────
#                    Transfer helpers
# ─────────────────────────────────────────────────────────────────────
def asgpu(arr):
    """Transfer array ke GPU jika tersedia, else return as-is."""
    if GPU_AVAILABLE:
        return _cupy.asarray(arr)
    return np.asarray(arr)


def asnumpy(arr):
    """Transfer array dari GPU ke NumPy. Aman dipanggil pada array CPU."""
    if GPU_AVAILABLE and isinstance(arr, _cupy.ndarray):
        return _cupy.asnumpy(arr)
    return np.asarray(arr)


def free_gpu_memory():
    """Free unused GPU memory pool. Panggil setelah inversi besar selesai."""
    if GPU_AVAILABLE:
        try:
            _cupy.get_default_memory_pool().free_all_blocks()
            _cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
#                    GPU/CPU linear algebra wrappers
# ─────────────────────────────────────────────────────────────────────
def gpu_cholesky(A, lower=True, ridge=0.0):
    """
    Cholesky factorization: A = L @ L^T (lower=True) atau L^T @ L.

    Returns (L,) — bisa lower atau upper sesuai parameter.
    Untuk solve gunakan gpu_cho_solve(L, b, lower).
    """
    if ridge > 0:
        n = A.shape[0]
        if GPU_AVAILABLE and isinstance(A, _cupy.ndarray):
            A = A + ridge * _cupy.eye(n, dtype=A.dtype)
        else:
            A = A + ridge * np.eye(n, dtype=A.dtype)

    if GPU_AVAILABLE and isinstance(A, _cupy.ndarray):
        L = _cupy_linalg.cholesky(A)              # CuPy returns lower-triangular
        if not lower:
            L = L.T
    else:
        # SciPy returns (L, lower_flag)
        L_sc, low_flag = scipy_linalg.cho_factor(A, lower=lower, overwrite_a=False)
        # Zero out the upper (or lower) triangle so behaviour matches CuPy
        if lower:
            L = np.tril(L_sc) if low_flag else np.tril(L_sc.T)
        else:
            L = np.triu(L_sc) if not low_flag else np.triu(L_sc.T)
    return L


def gpu_cho_solve(L, b, lower=True):
    """Solve A @ x = b given Cholesky factor L (A = L L^T)."""
    if GPU_AVAILABLE and isinstance(L, _cupy.ndarray):
        # CuPy: solve via solve_triangular twice
        from cupyx.scipy.linalg import solve_triangular as cp_solve_tri
        if lower:
            y = cp_solve_tri(L, b, lower=True)
            x = cp_solve_tri(L.T, y, lower=False)
        else:
            y = cp_solve_tri(L.T, b, lower=True)
            x = cp_solve_tri(L, y, lower=False)
        return x
    else:
        # SciPy
        return scipy_linalg.cho_solve((L, lower), b)


def gpu_solve(A, b):
    """General linear solve A @ x = b."""
    if GPU_AVAILABLE and isinstance(A, _cupy.ndarray):
        return _cupy_linalg.solve(A, b)
    return scipy_linalg.solve(A, b)


def gpu_inv(A):
    """Matrix inverse — gunakan jika A kecil atau perlu inverse eksplisit."""
    if GPU_AVAILABLE and isinstance(A, _cupy.ndarray):
        return _cupy_linalg.inv(A)
    return np.linalg.inv(A)


def gpu_slogdet(A):
    """Sign and log-determinant."""
    if GPU_AVAILABLE and isinstance(A, _cupy.ndarray):
        return _cupy_linalg.slogdet(A)
    return np.linalg.slogdet(A)


# ─────────────────────────────────────────────────────────────────────
#                    Convenience: percentile + sampling
# ─────────────────────────────────────────────────────────────────────
def gpu_percentile(arr, q, axis=None):
    if GPU_AVAILABLE and isinstance(arr, _cupy.ndarray):
        return _cupy.percentile(arr, q, axis=axis)
    return np.percentile(arr, q, axis=axis)


def gpu_default_rng(seed=None):
    """Random generator — CuPy or NumPy."""
    if GPU_AVAILABLE:
        return _cupy.random.default_rng(seed)
    return np.random.default_rng(seed)
