# data_link_layer/error_correction.py
"""
Error correction schemes:
- None
- Repetition (k)
- Hamming(7,4)
- Reed-Solomon(255,223)  [optional]
- NEW: Convolutional K=7 (g0=133_o, g1=171_o) with puncturing: R=1/2, 2/3, 3/4

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
        synd_val = (s1 + (s2 << 1) + (s3 << 2)).astype(np.uint8)
        map_arr = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)  # pos map (1-based)
        err_pos = map_arr[synd_val]
        for i in range(C.shape[0]):
            pos = int(err_pos[i])
            if pos != 0:
                C[i, pos-1] ^= 1
        data = C[:, :4].reshape(-1)
        return data

# ----------------- Reed-Solomon(255,223) optional -----------------
class RS255223FEC(FECBase):
    """
    Byte-oriented RS(255,223) over GF(256). Requires 'reedsolo'.
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

# ----------------- NEW: Convolutional (K=7) with puncturing -----------------
class ConvK7FEC(FECBase):
    """
    Rate-1/2 mother code with K=7 (g0=133_o, g1=171_o) + puncturing:
      R=1/2: p0=[1],        p1=[1]
      R=2/3: p0=[1,1,0],    p1=[1,0,1]
      R=3/4: p0=[1,1,0,1],  p1=[1,0,1,1]
    Hard-decision Viterbi (64 states). Tail-bitingは用いず，ゼロ終端（m=6個の0）．
    """
    def __init__(self, rate: str = "1/2"):
        rate = str(rate).lower().replace("r","").replace("_","/")  # "3/4" 等へ正規化
        if rate in ("1/2", "12"):
            p0, p1 = [1], [1]; R = 1/2
            self.name = "conv_k7_r12"
        elif rate in ("2/3", "23"):
            p0, p1 = [1,1,0], [1,0,1]; R = 2/3
            self.name = "conv_k7_r23"
        elif rate in ("3/4", "34"):
            p0, p1 = [1,1,0,1], [1,0,1,1]; R = 3/4
            self.name = "conv_k7_r34"
        else:
            raise ValueError(f"Unsupported conv. rate: {rate}")
        self.p0 = np.array(p0, dtype=np.uint8)
        self.p1 = np.array(p1, dtype=np.uint8)
        self.period = len(p0)
        self.code_rate = float(R)

        # mother code params
        self.m = 6
        self.K = 7
        self.g0 = 0o133
        self.g1 = 0o171

        # Precompute trellis
        S = 1 << self.m
        self.next_state = np.zeros((S, 2), dtype=np.int32)
        self.out0 = np.zeros((S, 2), dtype=np.uint8)
        self.out1 = np.zeros((S, 2), dtype=np.uint8)
        for s in range(S):
            for u in (0, 1):
                reg = (u << self.m) | s
                y0 = bin(reg & self.g0).count("1") & 1
                y1 = bin(reg & self.g1).count("1") & 1
                ns = (s >> 1) | (u << (self.m - 1))
                self.next_state[s, u] = ns
                self.out0[s, u] = y0
                self.out1[s, u] = y1

    # --- encode/decode ---
    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        s = 0
        y0_list, y1_list = [], []
        for u in b:
            y0 = int(self.out0[s, int(u)]); y1 = int(self.out1[s, int(u)])
            y0_list.append(y0); y1_list.append(y1)
            s = int(self.next_state[s, int(u)])
        # tail zeros (zero-termination)
        for _ in range(self.m):
            y0 = int(self.out0[s, 0]); y1 = int(self.out1[s, 0])
            y0_list.append(y0); y1_list.append(y1)
            s = int(self.next_state[s, 0])

        # puncture
        out = []
        P = self.period
        for t in range(len(y0_list)):
            if self.p0[t % P]:
                out.append(y0_list[t])
            if self.p1[t % P]:
                out.append(y1_list[t])
        return np.array(out, dtype=np.uint8)

    def _depuncture_to_pairs(self, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs0, obs1) of length T; missing positions = -1."""
        r = np.asarray(bits, dtype=np.uint8).reshape(-1)
        obs0, obs1 = [], []
        i = 0
        P = self.period
        while i < len(r):
            # stream 0
            if self.p0[len(obs0) % P]:
                if i < len(r): b0 = int(r[i]); i += 1
                else:          b0 = -1
            else:
                b0 = -1
            # stream 1
            if self.p1[len(obs1) % P]:
                if i < len(r): b1 = int(r[i]); i += 1
                else:          b1 = -1
            else:
                b1 = -1
            obs0.append(b0); obs1.append(b1)
        return np.array(obs0, dtype=np.int16), np.array(obs1, dtype=np.int16)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        # Depuncture
        obs0, obs1 = self._depuncture_to_pairs(bits)
        T = int(len(obs0))
        S = 1 << self.m
        INF = 10**9

        pm = np.full(S, INF, dtype=np.int64); pm[0] = 0
        prev_state = np.full((T, S), -1, dtype=np.int16)
        prev_bit   = np.zeros((T, S), dtype=np.uint8)

        for t in range(T):
            new_pm = np.full(S, INF, dtype=np.int64)
            o0, o1 = int(obs0[t]), int(obs1[t])
            for ps in range(S):
                m0 = pm[ps]
                if m0 >= INF:  # unreachable
                    continue
                for u in (0, 1):
                    ns = int(self.next_state[ps, u])
                    y0 = int(self.out0[ps, u]); y1 = int(self.out1[ps, u])
                    dist = 0
                    if o0 != -1: dist += (o0 ^ y0)
                    if o1 != -1: dist += (o1 ^ y1)
                    cost = m0 + dist
                    if cost < new_pm[ns]:
                        new_pm[ns] = cost
                        prev_state[t, ns] = ps
                        prev_bit[t, ns] = u
            pm = new_pm

        # prefer zero state (due to zero-termination); fallback to argmin
        end_state = 0 if pm[0] < INF else int(np.argmin(pm))
        s = end_state
        bits_rev = []
        for t in range(T - 1, -1, -1):
            u = int(prev_bit[t, s])
            bits_rev.append(u)
            s = int(prev_state[t, s])
            if s < 0:  # break in path; pad
                s = 0
        seq = bits_rev[::-1]
        # remove tail bits (m zeros appended at encoder)
        if len(seq) >= self.m:
            seq = seq[:len(seq) - self.m]
        return np.array(seq, dtype=np.uint8)

# ----------------- Factory -----------------
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
    if s in ("conv_k7_r12", "convk7_r12", "conv_k7_1_2"):
        return ConvK7FEC(rate="1/2")
    if s in ("conv_k7_r23", "convk7_r23", "conv_k7_2_3"):
        return ConvK7FEC(rate="2/3")
    if s in ("conv_k7_r34", "convk7_r34", "conv_k7_3_4"):
        return ConvK7FEC(rate="3/4")
    raise ValueError(f"Unknown FEC scheme: {scheme}")
