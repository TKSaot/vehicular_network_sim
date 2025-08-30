# common/byte_mapping.py
from __future__ import annotations
import numpy as np

def _ensure_len(b: np.ndarray, N: int) -> np.ndarray:
    """Trim or zero-pad to exactly N bytes (uint8)."""
    if b.size < N:
        return np.pad(b, (0, N - b.size), mode="constant", constant_values=0)
    if b.size > N:
        return b[:N]
    return b

def _perm_indices(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) ^ 0xA5C3_19D3)
    return rng.permutation(N)

def map_bytes(data: bytes, mtu_bytes: int, scheme: str = "none", seed: int | None = None) -> bytes:
    """
    Apply a byte mapping across the whole payload BEFORE segmentation.
    - "none": identity
    - "permute": global permutation by RNG(seed)
    - "frame_block": treat payload as matrix [rows=ceil(N/mtu), cols=mtu] and read column-wise
    """
    if scheme == "none":
        return data
    arr = np.frombuffer(data, dtype=np.uint8)
    N = arr.size
    if N == 0:
        return data

    if scheme == "permute":
        if seed is None:
            raise ValueError("permute mapping requires a seed")
        perm = _perm_indices(N, seed)
        return arr[perm].tobytes()

    if scheme == "frame_block":
        M = (N + mtu_bytes - 1) // mtu_bytes  # number of DATA frames (no headers)
        order = []
        for c in range(mtu_bytes):
            for r in range(M):
                j = r * mtu_bytes + c
                if j < N:
                    order.append(j)
        order = np.asarray(order, dtype=np.int64)
        return arr[order].tobytes()

    raise ValueError(f"Unknown byte mapping scheme: {scheme}")

def unmap_bytes(data_rx: bytes, mtu_bytes: int, scheme: str = "none",
                seed: int | None = None, original_len: int | None = None) -> bytes:
    """
    Invert the mapping after reassembly. We assume drop_bad_frames=False so length is preserved.
    If lengths mismatch, we trim/pad to original_len (if provided).
    """
    arr = np.frombuffer(data_rx, dtype=np.uint8)
    N = arr.size if original_len is None else int(original_len)
    arr = _ensure_len(arr, N)

    if scheme == "none":
        return arr.tobytes()

    if scheme == "permute":
        if seed is None:
            raise ValueError("permute mapping requires a seed")
        perm = _perm_indices(N, seed)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(N, dtype=perm.dtype)
        return arr[inv].tobytes()

    if scheme == "frame_block":
        M = (N + mtu_bytes - 1) // mtu_bytes
        order = []
        for c in range(mtu_bytes):
            for r in range(M):
                j = r * mtu_bytes + c
                if j < N:
                    order.append(j)
        order = np.asarray(order, dtype=np.int64)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=order.dtype)
        return arr[inv].tobytes()

    raise ValueError(f"Unknown byte mapping scheme: {scheme}")
