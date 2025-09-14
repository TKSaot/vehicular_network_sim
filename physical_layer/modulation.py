# physical_layer/modulation.py
# GPU/CPU backend 両対応版：LLR（信頼度）を返すデモジュレーションを追加
from __future__ import annotations
from typing import Tuple
from common.backend import xp, np, to_xp, asnumpy

# ---------- BPSK ----------
def _bpsk_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    return (2.0 * b.astype(xp.float64) - 1.0).astype(xp.complex128)

def _bpsk_demod_with_rel(symbols) -> Tuple["np.ndarray", "np.ndarray"]:
    z = to_xp(symbols).astype(xp.complex128, copy=False).real
    bits = (z >= 0).astype(xp.uint8)
    rel = xp.abs(z).astype(xp.float32)
    return asnumpy(bits), asnumpy(rel)

def _bpsk_demod(symbols) -> "np.ndarray":
    bits, _ = _bpsk_demod_with_rel(symbols)
    return bits

# ---------- QPSK(Gray) ----------
def _qpsk_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    if (b.size % 2) != 0:
        b = xp.concatenate([b, xp.zeros(1, dtype=xp.uint8)])
    pairs = b.reshape(-1, 2).astype(xp.float64)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]
    return ((i + 1j*q) / xp.sqrt(2.0)).astype(xp.complex128)

def _qpsk_demod_with_rel(symbols) -> Tuple["np.ndarray", "np.ndarray"]:
    z = to_xp(symbols).astype(xp.complex128, copy=False)
    re = z.real.astype(xp.float64, copy=False)
    im = z.imag.astype(xp.float64, copy=False)
    b0 = (re < 0).astype(xp.uint8)  # I
    b1 = (im < 0).astype(xp.uint8)  # Q
    rel0 = xp.abs(re)
    rel1 = xp.abs(im)
    bits = xp.vstack([b0, b1]).T.reshape(-1)
    rel  = xp.vstack([rel0, rel1]).T.reshape(-1).astype(xp.float32, copy=False)
    return asnumpy(bits), asnumpy(rel)

def _qpsk_demod(symbols) -> "np.ndarray":
    bits, _ = _qpsk_demod_with_rel(symbols)
    return bits

# ---------- 16QAM(Gray) ----------
def _16qam_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    pad = (-b.size) % 4
    if pad:
        b = xp.concatenate([b, xp.zeros(pad, dtype=xp.uint8)])
    quads = b.reshape(-1, 4).astype(xp.float64)
    I = (1.0 - 2.0 * quads[:, 0]) * (3.0 - 2.0 * quads[:, 1])
    Q = (1.0 - 2.0 * quads[:, 2]) * (3.0 - 2.0 * quads[:, 3])
    return ((I + 1j*Q) / xp.sqrt(10.0)).astype(xp.complex128)

def _16qam_demod_with_rel(symbols) -> Tuple["np.ndarray", "np.ndarray"]:
    z = to_xp(symbols).astype(xp.complex128, copy=False) * xp.sqrt(10.0)
    I = z.real.astype(xp.float64, copy=False)
    Q = z.imag.astype(xp.float64, copy=False)

    b_i1 = (I < 0).astype(xp.uint8)
    b_i0 = (xp.abs(I) < 2.0).astype(xp.uint8)
    b_q1 = (Q < 0).astype(xp.uint8)
    b_q0 = (xp.abs(Q) < 2.0).astype(xp.uint8)

    r_i1 = xp.abs(I)
    r_i0 = xp.abs(xp.abs(I) - 2.0)
    r_q1 = xp.abs(Q)
    r_q0 = xp.abs(xp.abs(Q) - 2.0)

    bits = xp.vstack([b_i1, b_i0, b_q1, b_q0]).T.reshape(-1)
    rel  = xp.vstack([r_i1, r_i0, r_q1, r_q0]).T.reshape(-1).astype(xp.float32, copy=False)
    return asnumpy(bits), asnumpy(rel)

def _16qam_demod(symbols) -> "np.ndarray":
    bits, _ = _16qam_demod_with_rel(symbols)
    return bits

# ---------- Modulator wrapper ----------
class Modulator:
    def __init__(self, scheme: str = "qpsk"):
        s = scheme.lower()
        if s not in ("bpsk", "qpsk", "16qam"):
            raise ValueError("Unsupported modulation")
        self.scheme = s

    @property
    def bits_per_symbol(self) -> int:
        return {"bpsk": 1, "qpsk": 2, "16qam": 4}[self.scheme]

    def modulate(self, bits) -> "xp.ndarray":
        if self.scheme == "bpsk":  return _bpsk_mod(bits)
        if self.scheme == "qpsk":  return _qpsk_mod(bits)
        if self.scheme == "16qam": return _16qam_mod(bits)
        raise RuntimeError

    def demodulate(self, symbols) -> "np.ndarray":
        if self.scheme == "bpsk":  return _bpsk_demod(symbols)
        if self.scheme == "qpsk":  return _qpsk_demod(symbols)
        if self.scheme == "16qam": return _16qam_demod(symbols)
        raise RuntimeError

    def demodulate_with_reliability(self, symbols) -> Tuple["np.ndarray","np.ndarray"]:
        if self.scheme == "bpsk":  return _bpsk_demod_with_rel(symbols)
        if self.scheme == "qpsk":  return _qpsk_demod_with_rel(symbols)
        if self.scheme == "16qam": return _16qam_demod_with_rel(symbols)
        raise RuntimeError

# ---------- PHY frame builder ----------
def build_phy_frame(bits, mod: Modulator,
                    preamble_len_bits: int = 32,
                    pilot_len_symbols: int = 16
                    ) -> tuple["xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    # preamble: 1010...(BPSK)
    base = xp.array([1,0], dtype=xp.uint8)
    pre_bits = xp.tile(base, preamble_len_bits // 2)
    if (preamble_len_bits % 2) == 1:
        pre_bits = xp.concatenate([pre_bits, xp.array([1], dtype=xp.uint8)])
    pre_sym = _bpsk_mod(pre_bits).astype(xp.complex128)

    pilot = xp.ones(pilot_len_symbols, dtype=xp.complex128) * (1.0 + 1j) / xp.sqrt(2.0)
    data_syms = mod.modulate(bits).astype(xp.complex128)
    all_syms = xp.concatenate([pre_sym, pilot, data_syms])
    return all_syms, pilot, data_syms
