
# multimodal_ofdm/utils.py
from __future__ import annotations
import numpy as np
import binascii

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

# ----------------- CRC -----------------
def append_crc32(payload: bytes) -> bytes:
    c = (binascii.crc32(payload) & 0xFFFFFFFF)
    return payload + c.to_bytes(4, "big")

def verify_and_strip_crc32(data_with_crc: bytes) -> tuple[bool, bytes]:
    if len(data_with_crc) < 4:
        return False, b""
    pl, crc_b = data_with_crc[:-4], data_with_crc[-4:]
    ok = (binascii.crc32(pl) & 0xFFFFFFFF) == int.from_bytes(crc_b, "big")
    return ok, pl

# ----------------- Interleaving -----------------
def block_interleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """
    Simple block interleaver: write row-wise [depth x cols], read column-wise.
    We zero-pad so reshape is always safe.
    """
    depth = max(1, int(depth))
    if depth == 1:
        return np.asarray(bits, dtype=np.uint8).reshape(-1)
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = len(b)
    cols = int(np.ceil(L / depth))
    pad = depth * cols - L
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    mat = b.reshape(depth, cols)
    out = mat.T.reshape(-1)
    return out

def block_deinterleave(
    bits: np.ndarray,
    depth: int,
    original_len_bits: int | None = None,
    original_len: int | None = None,
) -> np.ndarray:
    """
    Inverse of block_interleave. Accepts either 'original_len_bits' or 'original_len'.
    We zero-pad incoming length to a multiple of 'depth' before reshaping to avoid
    ValueError on environments where partial columns can appear.
    """
    # prefer 'original_len' if provided (to match まとめ.py style), else fallback
    Lout = original_len if original_len is not None else original_len_bits

    depth = max(1, int(depth))
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if depth == 1:
        return b[:Lout] if Lout is not None else b

    if b.size == 0:
        return b

    L = len(b)
    cols = int(np.ceil(L / depth))
    need = cols * depth - L
    if need > 0:
        b = np.concatenate([b, np.zeros(need, dtype=np.uint8)])

    mat = b.reshape(cols, depth).T
    out = mat.reshape(-1)
    return out[:Lout] if Lout is not None else out
