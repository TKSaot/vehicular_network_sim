# ==========================
# ./data_link_layer/error_correction.py
# ==========================
# Error correction schemes:
#   None / Repeat / Hamming(7,4) / RS(255,223)
#   NEW: Conv(K=7) with vectorized hard/soft Viterbi (GPU optional)
# ==========================
from __future__ import annotations
import numpy as np
from typing import Optional
from common.backend import xp, to_xp, asnumpy

# ---------- Base ----------
class FECBase:
    name = "base"
    code_rate = 1.0
    def encode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.uint8)
    def decode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.uint8)

# ---------- None ----------
class NoFEC(FECBase):
    name = "none"
    code_rate = 1.0

# ---------- Repetition ----------
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

# ---------- Hamming(7,4) ----------
class Hamming74FEC(FECBase):
    """
    Systematic Hamming(7,4) with codeword [d1 d2 d3 d4 p1 p2 p3]
    p1 = d1 ^ d2 ^ d4; p2 = d1 ^ d3 ^ d4; p3 = d2 ^ d3 ^ d4
    """
    name = "hamming74"
    code_rate = 4/7
    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        pad = (-len(b)) % 4
        if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        if len(b) == 0: return b
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
        if L == 0: return np.zeros(0, dtype=np.uint8)
        C = c.reshape(-1, 7)
        d1,d2,d3,d4,p1,p2,p3 = [C[:,i] for i in range(7)]
        s1 = (d1 ^ d2 ^ d4 ^ p1).astype(np.uint8)
        s2 = (d1 ^ d3 ^ d4 ^ p2).astype(np.uint8)
        s3 = (d2 ^ d3 ^ d4 ^ p3).astype(np.uint8)
        synd = (s1 + (s2<<1) + (s3<<2)).astype(np.uint8)
        pos_map = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)
        err_pos = pos_map[synd]
        for i in range(C.shape[0]):
            p = int(err_pos[i])
            if p != 0:
                C[i, p-1] ^= 1
        return C[:, :4].reshape(-1)

# ---------- RS(255,223) (optional) ----------
class RS255223FEC(FECBase):
    name = "rs255_223"
    code_rate = 223/255
    def __init__(self):
        try:
            import reedsolo  # type: ignore
        except Exception as e:
            raise ImportError("reedsolo is required for RS255_223 (pip install reedsolo)") from e
        self.rs = reedsolo.RSCodec(32)
    def encode(self, bits: np.ndarray) -> np.ndarray:
        from common.utils import bits_to_bytes, bytes_to_bits
        data = bits_to_bytes(bits)
        if len(data) % 223 != 0:
            data += bytes([0]) * (223 - (len(data) % 223))
        out = bytearray()
        for i in range(0, len(data), 223):
            out.extend(self.rs.encode(data[i:i+223]))
        return bytes_to_bits(bytes(out))
    def decode(self, bits: np.ndarray) -> np.ndarray:
        from common.utils import bits_to_bytes, bytes_to_bits
        data = bits_to_bytes(bits)
        L = (len(data) // 255) * 255
        data = data[:L]
        out = bytearray()
        for i in range(0, len(data), 255):
            block = data[i:i+255]
            try:
                dec = self.rs.decode(block)
            except Exception:
                dec = bytes([0]) * 223
            out.extend(dec)
        return bytes_to_bits(bytes(out))

# ---------- NEW: Conv(K=7) vectorized hard/soft Viterbi ----------
class ConvK7FEC(FECBase):
    """
    Mother code (K=7, m=6) with generators g0=133_o, g1=171_o.
    Puncturing:
      R=1/2: p0=[1],        p1=[1]
      R=2/3: p0=[1,1,0],    p1=[1,0,1]
      R=3/4: p0=[1,1,0,1],  p1=[1,0,1,1]
    Provides:
      - encode(bits)  -> np.uint8 coded bits
      - decode(bits)  -> np.uint8 hard-decision Viterbi (vectorized)
      - decode_soft(hard_bits, rel_bits) -> np.uint8 soft-aided Viterbi (uses per-coded-bit reliabilities)
    """
    def __init__(self, rate: str = "1/2"):
        rate = str(rate).lower().replace("r","").replace("_","/").replace(" ", "")
        if rate in ("1/2","12"):
            self.p0, self.p1 = [1], [1]; self.name = "conv_k7_r12"; self.code_rate = 1/2
        elif rate in ("2/3","23"):
            self.p0, self.p1 = [1,1,0], [1,0,1]; self.name = "conv_k7_r23"; self.code_rate = 2/3
        elif rate in ("3/4","34"):
            self.p0, self.p1 = [1,1,0,1], [1,0,1,1]; self.name = "conv_k7_r34"; self.code_rate = 3/4
        else:
            raise ValueError(f"Unsupported rate: {rate}")
        self.P = len(self.p0)
        self.m = 6
        self.K = 7
        self.S = 1 << self.m
        self.g0 = 0o133
        self.g1 = 0o171

        # Precompute trellis tables on CPU (small)
        self.next_state = np.zeros((self.S, 2), dtype=np.int16)
        self.out0 = np.zeros((self.S, 2), dtype=np.uint8)
        self.out1 = np.zeros((self.S, 2), dtype=np.uint8)
        for s in range(self.S):
            for u in (0,1):
                reg = (u << self.m) | s
                y0 = bin(reg & self.g0).count("1") & 1
                y1 = bin(reg & self.g1).count("1") & 1
                ns = (s >> 1) | (u << (self.m - 1))
                self.next_state[s,u] = ns
                self.out0[s,u] = y0
                self.out1[s,u] = y1

        # Invert trellis: for each ns, two (prev_state, input_bit) pairs
        prev0 = np.zeros(self.S, dtype=np.int16)
        prev1 = np.zeros(self.S, dtype=np.int16)
        inb0  = np.zeros(self.S, dtype=np.uint8)
        inb1  = np.zeros(self.S, dtype=np.uint8)
        out0a = np.zeros(self.S, dtype=np.uint8)
        out1a = np.zeros(self.S, dtype=np.uint8)
        out0b = np.zeros(self.S, dtype=np.uint8)
        out1b = np.zeros(self.S, dtype=np.uint8)
        fill = np.zeros(self.S, dtype=np.uint8)
        for ps in range(self.S):
            for u in (0,1):
                ns = int(self.next_state[ps,u])
                if fill[ns] == 0:
                    prev0[ns] = ps; inb0[ns] = u; out0a[ns] = self.out0[ps,u]; out1a[ns] = self.out1[ps,u]; fill[ns] = 1
                else:
                    prev1[ns] = ps; inb1[ns] = u; out0b[ns] = self.out0[ps,u]; out1b[ns] = self.out1[ps,u]
        self.prev0 = prev0; self.prev1 = prev1
        self.inb0  = inb0;  self.inb1  = inb1
        self.o0a   = out0a; self.o1a   = out1a
        self.o0b   = out0b; self.o1b   = out1b

    # --- Encoding (simple & fast enough) ---
    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        s = 0; y0_list=[]; y1_list=[]
        for u in b:
            u = int(u)
            y0 = int(self.out0[s,u]); y1 = int(self.out1[s,u])
            y0_list.append(y0); y1_list.append(y1)
            s = int(self.next_state[s,u])
        for _ in range(self.m):  # zero-termination
            y0 = int(self.out0[s,0]); y1 = int(self.out1[s,0])
            y0_list.append(y0); y1_list.append(y1)
            s = int(self.next_state[s,0])
        # puncture
        out = []
        for t in range(len(y0_list)):
            if self.p0[t % self.P]: out.append(y0_list[t])
            if self.p1[t % self.P]: out.append(y1_list[t])
        return np.array(out, dtype=np.uint8)

    # --- Helper: depuncture to two streams (obs, and optional weights) ---
    def _depuncture_pairs(self, obs_bits: np.ndarray, rel_bits: Optional[np.ndarray] = None):
        r = np.asarray(obs_bits, dtype=np.uint8).reshape(-1)
        has_rel = rel_bits is not None
        if has_rel:
            w = np.asarray(rel_bits, dtype=np.float32).reshape(-1)
        P = self.P
        o0 = []; o1 = []; w0 = []; w1 = []
        i = 0
        while i < len(r):
            if self.p0[len(o0) % P]:
                o0.append(int(r[i])); w0.append(float(w[i]) if has_rel else 1.0); i += 1
            else:
                o0.append(-1); w0.append(0.0)
            if self.p1[len(o1) % P]:
                o1.append(int(r[i])); w1.append(float(w[i]) if has_rel else 1.0); i += 1
            else:
                o1.append(-1); w1.append(0.0)
        return np.array(o0, dtype=np.int16), np.array(o1, dtype=np.int16), \
               np.array(w0, dtype=np.float32), np.array(w1, dtype=np.float32)

    # --- Vectorized forward pass (hard or weighted-soft) ---
    def _viterbi_forward(self, o0, o1, w0, w1):
        # move to backend
        prev0 = to_xp(self.prev0); prev1 = to_xp(self.prev1)
        inb0  = to_xp(self.inb0);  inb1  = to_xp(self.inb1)
        o0a   = to_xp(self.o0a);   o1a   = to_xp(self.o1a)
        o0b   = to_xp(self.o0b);   o1b   = to_xp(self.o1b)

        T = int(len(o0)); S = self.S
        INF = xp.float32(1e9)

        pm = xp.full(S, INF, dtype=xp.float32); pm[0] = 0.0  # start from state 0
        prev_state = xp.empty((T, S), dtype=xp.uint8)  # store predecessors
        prev_bit   = xp.empty((T, S), dtype=xp.uint8)

        # Prepare obs/weights on backend
        o0x = to_xp(o0); o1x = to_xp(o1); w0x = to_xp(w0); w1x = to_xp(w1)

        for t in range(T):
            # costs from two predecessor branches into each next state
            # mismatch cost: weight * XOR(predicted, observed); punctured -> weight=0
            obs0 = o0x[t]; obs1 = o1x[t]
            ww0  = w0x[t]; ww1  = w1x[t]

            # Branch A
            cA = pm[prev0]
            if int(obs0) != -1:
                cA = cA + ww0 * xp.abs(o0a - obs0)
            if int(obs1) != -1:
                cA = cA + ww1 * xp.abs(o1a - obs1)

            # Branch B
            cB = pm[prev1]
            if int(obs0) != -1:
                cB = cB + ww0 * xp.abs(o0b - obs0)
            if int(obs1) != -1:
                cB = cB + ww1 * xp.abs(o1b - obs1)

            chooseA = cA <= cB
            new_pm = xp.where(chooseA, cA, cB)
            prev_state[t,:] = xp.where(chooseA, prev0, prev1).astype(xp.uint8)
            prev_bit[t,:]   = xp.where(chooseA, inb0,  inb1).astype(xp.uint8)
            pm = new_pm

        return pm, prev_state, prev_bit

    def _backtrack(self, pm, prev_state, prev_bit):
        # End at state 0 if possible (zero-termination), else best state
        end_state = int(xp.argmin(pm).item())
        s = end_state
        T = prev_bit.shape[0]
        out_bits = []
        # backtrack on CPU (cheap) for portability
        PS = asnumpy(prev_state)
        PB = asnumpy(prev_bit)
        for t in range(T-1, -1, -1):
            u = int(PB[t, s]); out_bits.append(u); s = int(PS[t, s])
        out_bits = out_bits[::-1]
        # remove tail bits (m)
        if len(out_bits) >= self.m:
            out_bits = out_bits[:len(out_bits) - self.m]
        return np.array(out_bits, dtype=np.uint8)

    # --- Public decoders ---
    def decode(self, bits: np.ndarray) -> np.ndarray:
        # hard-decision: unit weights for non-punctured bits
        o0, o1, w0, w1 = self._depuncture_pairs(bits, rel_bits=None)
        pm, ps, pb = self._viterbi_forward(o0, o1, w0, w1)
        return self._backtrack(pm, ps, pb)

    def decode_soft(self, hard_bits: np.ndarray, rel_bits: np.ndarray) -> np.ndarray:
        # soft-aided: weights = reliability; still uses hard observed bits, but weighted
        o0, o1, w0, w1 = self._depuncture_pairs(hard_bits, rel_bits)
        pm, ps, pb = self._viterbi_forward(o0, o1, w0, w1)
        return self._backtrack(pm, ps, pb)

# ---------- Factory ----------
def make_fec(scheme: str, repeat_k: int = 3) -> FECBase:
    s = scheme.lower()
    if s == "none":          return NoFEC()
    if s == "repeat":        return RepetitionFEC(k=repeat_k)
    if s == "hamming74":     return Hamming74FEC()
    if s == "rs255_223":
        try: return RS255223FEC()
        except ImportError:
            print("[WARN] RS255_223 selected but 'reedsolo' not installed. Falling back to NoFEC.")
            return NoFEC()
    if s in ("conv_k7_r12","convk7_r12","conv_k7_1_2"): return ConvK7FEC(rate="1/2")
    if s in ("conv_k7_r23","convk7_r23","conv_k7_2_3"): return ConvK7FEC(rate="2/3")
    if s in ("conv_k7_r34","convk7_r34","conv_k7_3_4"): return ConvK7FEC(rate="3/4")
    raise ValueError(f"Unknown FEC scheme: {scheme}")
