# common/utils.py
"""
General helpers: bit/byte conversions, CRC, interleaving, padding.
"""

from __future__ import annotations
import numpy as np
import binascii

def set_seed(seed: int | None):
    if seed is not None:
        np.random.seed(seed)

# ----------------- Bits/Bytes -----------------
def bytes_to_bits(b: bytes) -> np.ndarray:
    """Convert bytes to a 1-D np.uint8 array of 0/1 bits."""
    if isinstance(b, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(b, dtype=np.uint8)
    elif isinstance(b, np.ndarray) and b.dtype == np.uint8:
        arr = b
    else:
        arr = np.array(list(b), dtype=np.uint8)
    return np.unpackbits(arr)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a 1-D bits array (0/1) to bytes. Pads to a multiple of 8 with zeros."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

def pad_bits_to_multiple(bits: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    """Pad with zeros so len(bits) is a multiple of m. Return (padded_bits, pad_len)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(bits)) % m
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bits, pad

def remove_tail_bits(bits: np.ndarray, n_remove: int) -> np.ndarray:
    if n_remove <= 0:
        return bits
    return bits[:-n_remove] if n_remove <= len(bits) else np.array([], dtype=np.uint8)

# ----------------- CRC -----------------
def crc32_bytes(data: bytes) -> int:
    """CRC-32 for a bytes payload (unsigned)."""
    return binascii.crc32(data) & 0xFFFFFFFF

def append_crc32(payload: bytes) -> bytes:
    c = crc32_bytes(payload)
    return payload + c.to_bytes(4, 'big')

def verify_and_strip_crc32(data_with_crc: bytes) -> tuple[bool, bytes]:
    if len(data_with_crc) < 4:
        return False, b""
    payload, crc = data_with_crc[:-4], data_with_crc[-4:]
    ok = (crc32_bytes(payload) == int.from_bytes(crc, 'big'))
    return ok, payload

# ----------------- Interleaving -----------------
def block_interleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """Simple block interleaver: write row-wise, read column-wise."""
    depth = max(1, int(depth))
    if depth == 1:
        return np.asarray(bits, dtype=np.uint8)
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = len(bits)
    cols = int(np.ceil(L / depth))
    pad = (depth * cols) - L
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    mat = bits.reshape(depth, cols)
    out = mat.T.reshape(-1)
    return out

def block_deinterleave(bits: np.ndarray, depth: int, original_len: int | None = None) -> np.ndarray:
    """Inverse of block_interleave. Optionally trim to original_len."""
    depth = max(1, int(depth))
    if depth == 1:
        out = np.asarray(bits, dtype=np.uint8).reshape(-1)
        return out[:original_len] if original_len is not None else out
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = len(bits)
    cols = int(np.ceil(L / depth))
    mat = bits.reshape(cols, depth).T
    out = mat.reshape(-1)
    if original_len is not None:
        out = out[:original_len]
    return out

# ----------------- PSI Metrics -----------------
def psnr(a: np.ndarray, b: np.ndarray, data_range: int = 255) -> float:
    """Peak Signal-to-Noise Ratio between two images (uint8 arrays)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)
