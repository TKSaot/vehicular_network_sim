# common/backend.py
from __future__ import annotations
import os

# Always have NumPy
import numpy as _np

# Try CuPy
try:
    import cupy as _cp  # type: ignore
except Exception:
    _cp = None

def _env_flag(name: str, default: str = "1") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

# Decide backend
_USE_CUDA = _env_flag("VC_USE_CUDA", "1") and (_cp is not None)

# Public aliases
is_cupy = bool(_USE_CUDA)
np = _np
cp = _cp
xp = _cp if is_cupy else _np

def asnumpy(a):
    """Return a NumPy array view/copy of 'a' (handles CuPy/NumPy)."""
    if is_cupy and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return _np.asarray(a)

def to_xp(a, dtype=None):
    """Convert arbitrary array-like to backend array."""
    if is_cupy:
        return _cp.asarray(a, dtype=dtype) if not isinstance(a, _cp.ndarray) else (a.astype(dtype) if dtype else a)
    return _np.asarray(a, dtype=dtype) if not isinstance(a, _np.ndarray) else (a.astype(dtype) if dtype else a)

def default_rng(seed=None):
    """
    Return a backend RNG object.
    - For NumPy: np.random.default_rng(seed)
    - For CuPy:  cp.random.RandomState(seed) (Generator is not guaranteed in all envs)
    """
    if is_cupy:
        # RandomState offers .normal/.randint etc. and is reproducible by seed.
        return _cp.random.RandomState(seed)  # type: ignore[attr-defined]
    return _np.random.default_rng(seed)
