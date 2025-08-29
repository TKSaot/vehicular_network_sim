# data_link_layer/error_correction.py
"""
Error correction schemes:
- None
- Repetition (k)
- Hamming(7,4)
- Reed-Solomon(255,223)  [optional, requires 'reedsolo' package]

All operate on bit arrays (0/1) and return bit arrays.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# ----------------- Base -----------------
class FECBase:
    name = "base"
    code_rate = 1.0
    def encode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.uint8)
    def decode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.uint8)

# ----------------- None -----------------
class NoFEC(FECBase):
    name = "none"
    code_rate = 1.0

# ----------------- Repetition -----------------
class RepetitionFEC(FECBase):
    def __init__(self, k: int = 3):
        assert k >= 1 and isinstance(k, int)
        self.k = k
        self.name = f"repeat{self.k}"
        self.code_rate = 1.0 / k
    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1, 1)
        return np.repeat(bits, self.k, axis=1).reshape(-1)
    def decode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) % self.k != 0:
            bits = bits[: (len(bits) // self.k) * self.k]
        grp = bits.reshape(-1, self.k)
        s = np.sum(grp, axis=1)
        return (s >= (self.k // 2 + 1)).astype(np.uint8)

# ----------------- Hamming(7,4) -----------------
class Hamming74FEC(FECBase):
    """
    Systematic Hamming(7,4) with codeword [d1 d2 d3 d4 p1 p2 p3]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4

    Parity-check H columns (for positions 1..7):
      pos1(d1): [1,1,0]
      pos2(d2): [1,0,1]
      pos3(d3): [0,1,1]
      pos4(d4): [1,1,1]
      pos5(p1): [1,0,0]
      pos6(p2): [0,1,0]
      pos7(p3): [0,0,1]
    """
    name = "hamming74"
    code_rate = 4/7

    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        pad = (-len(b)) % 4
        if pad:
            b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        if len(b) == 0:
            return b
        D = b.reshape(-1, 4)
        d1, d2, d3, d4 = D[:,0], D[:,1], D[:,2], D[:,3]
        p1 = (d1 ^ d2 ^ d4).astype(np.uint8)
        p2 = (d1 ^ d3 ^ d4).astype(np.uint8)
        p3 = (d2 ^ d3 ^ d4).astype(np.uint8)
        C = np.stack([d1, d2, d3, d4, p1, p2, p3], axis=1)
        return C.reshape(-1)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        c = np.asarray(bits, dtype=np.uint8).reshape(-1)
        L = (len(c) // 7) * 7
        c = c[:L]
        if L == 0:
            return np.zeros(0, dtype=np.uint8)
        C = c.reshape(-1, 7)
        d1, d2, d3, d4, p1, p2, p3 = [C[:,i] for i in range(7)]
        s1 = (d1 ^ d2 ^ d4 ^ p1).astype(np.uint8)
        s2 = (d1 ^ d3 ^ d4 ^ p2).astype(np.uint8)
        s3 = (d2 ^ d3 ^ d4 ^ p3).astype(np.uint8)

        # syndrome value (s1 + 2*s2 + 4*s3) → error position
        # mapping derived from H columns shown in class docstring
        synd_val = (s1 + (s2 << 1) + (s3 << 2)).astype(np.uint8)
        # map: 0->0(no flip), 1->5, 2->6, 3->1, 4->7, 5->2, 6->3, 7->4
        map_arr = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)
        err_pos = map_arr[synd_val]  # 0..7

        # flip indicated bit (1-based position)
        for i in range(C.shape[0]):
            pos = int(err_pos[i])
            if pos != 0:
                C[i, pos-1] ^= 1

        # extract data bits
        data = C[:, :4].reshape(-1)
        return data

# ----------------- Reed-Solomon(255,223) optional -----------------
class RS255223FEC(FECBase):
    """
    Byte-oriented RS(255,223) over GF(256). Requires 'reedsolo' package.
    Encodes/decodes bytes, converts to/from bits at the boundary.
    Code rate 223/255 ~ 0.8745. Corrects up to 16 byte errors per 255-byte block.
    """
    name = "rs255_223"
    code_rate = 223/255

    def __init__(self):
        try:
            import reedsolo  # type: ignore
        except Exception as e:
            raise ImportError("reedsolo package is required for RS255_223 FEC. Install via: pip install reedsolo") from e
        self.rs = reedsolo.RSCodec(32)  # n-k = 32 parity bytes

    def encode(self, bits: np.ndarray) -> np.ndarray:
        from common.utils import bits_to_bytes, bytes_to_bits
        data_bytes = bits_to_bytes(bits)
        if len(data_bytes) % 223 != 0:
            pad = 223 - (len(data_bytes) % 223)
            data_bytes += bytes([0])*pad
        out = bytearray()
        for i in range(0, len(data_bytes), 223):
            out.extend(self.rs.encode(data_bytes[i:i+223]))
        return bytes_to_bits(bytes(out))

    def decode(self, bits: np.ndarray) -> np.ndarray:
        from common.utils import bits_to_bytes, bytes_to_bits
        data_bytes = bits_to_bytes(bits)
        L = (len(data_bytes) // 255) * 255
        data_bytes = data_bytes[:L]
        out = bytearray()
        for i in range(0, len(data_bytes), 255):
            block = data_bytes[i:i+255]
            try:
                dec = self.rs.decode(block)  # -> 223 bytes
            except Exception:
                dec = bytes([0])*223
            out.extend(dec)
        return bytes_to_bits(bytes(out))

def make_fec(scheme: str, repeat_k: int = 3) -> FECBase:
    s = scheme.lower()
    if s == "none":
        return NoFEC()
    if s == "repeat":
        return RepetitionFEC(k=repeat_k)
    if s == "hamming74":
        return Hamming74FEC()
    if s == "rs255_223":
        try:
            return RS255223FEC()
        except ImportError:
            print("[WARN] RS255_223 selected but 'reedsolo' not installed. Falling back to NoFEC.")
            return NoFEC()
    raise ValueError(f"Unknown FEC scheme: {scheme}")
