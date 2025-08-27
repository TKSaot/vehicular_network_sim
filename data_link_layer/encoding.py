import numpy as np
import zlib
from typing import Tuple

def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) == 0:
        return b""
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    by = np.packbits(bits.astype(np.uint8))
    return by.tobytes()

# ---- フレーミング：len(4) + payload + crc32(4) ----
def build_length_crc_frame(payload: bytes) -> bytes:
    L = len(payload).to_bytes(4, 'big')
    crc = zlib.crc32(payload).to_bytes(4, 'big', signed=False)
    return L + payload + crc

def parse_length_crc_frame(frame: bytes) -> Tuple[bytes, bool]:
    if len(frame) < 8:
        return b"", False
    L = int.from_bytes(frame[0:4], 'big')
    if len(frame) < 8 + L:
        return b"", False
    payload = frame[4:4+L]
    crc_got = int.from_bytes(frame[4+L:8+L], 'big', signed=False)
    crc_calc = zlib.crc32(payload) & 0xffffffff
    return payload, (crc_got == crc_calc)

# ---- インターリーブ ----
def interleave(bits: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(bits))
    rng.shuffle(idx)
    return bits[idx], idx

def deinterleave(bits: np.ndarray, perm):
    if perm is None:
        return bits
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out = bits[inv]
    return out
