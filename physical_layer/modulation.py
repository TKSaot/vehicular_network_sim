# physical_layer/modulation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
from common.backend import xp, np, asnumpy, to_xp

# -------------------------
# BPSK
# -------------------------
def _bpsk_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    return (2.0 * b.astype(xp.float64) - 1.0).astype(xp.complex128)

def _bpsk_demod(symbols) -> "np.ndarray":
    s = to_xp(symbols).real
    out = (s >= 0).astype(xp.uint8)
    return asnumpy(out).astype(np.uint8)

# -------------------------
# QPSK (Gray)
# -------------------------
def _qpsk_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    if (b.size % 2) != 0:
        b = xp.concatenate([b, xp.zeros(1, dtype=xp.uint8)], axis=0)
    pairs = b.reshape(-1, 2).astype(xp.float64)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]
    return ((i + 1j * q) / xp.sqrt(2.0)).astype(xp.complex128)

def _qpsk_demod(symbols) -> "np.ndarray":
    z = to_xp(symbols)
    i = (z.real < 0).astype(xp.uint8)
    q = (z.imag < 0).astype(xp.uint8)
    out = xp.vstack([i, q]).T.reshape(-1)
    return asnumpy(out).astype(np.uint8)

# -------------------------
# 16QAM (Gray)
# -------------------------
def _16qam_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    pad = (-b.size) % 4
    if pad:
        b = xp.concatenate([b, xp.zeros(pad, dtype=xp.uint8)], axis=0)
    quads = b.reshape(-1, 4).astype(xp.float64)
    I = (1.0 - 2.0 * quads[:, 0]) * (3.0 - 2.0 * quads[:, 1])
    Q = (1.0 - 2.0 * quads[:, 2]) * (3.0 - 2.0 * quads[:, 3])
    return ((I + 1j * Q) / xp.sqrt(10.0)).astype(xp.complex128)

def _16qam_demod(symbols) -> "np.ndarray":
    z = to_xp(symbols) * xp.sqrt(10.0)
    I = z.real
    Q = z.imag

    def level_to_bits(vals):
        b1 = (vals < 0).astype(xp.uint8)          # sign
        b0 = (xp.abs(vals) < 2).astype(xp.uint8)  # inner(1) / outer(0)
        return b1, b0

    i1, i0 = level_to_bits(I)
    q1, q0 = level_to_bits(Q)
    bits = xp.vstack([i1, i0, q1, q0]).T.reshape(-1)
    return asnumpy(bits).astype(np.uint8)

# -------------------------
# Modulator wrapper
# -------------------------
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
        if self.scheme == "bpsk":
            return _bpsk_mod(bits)
        if self.scheme == "qpsk":
            return _qpsk_mod(bits)
        if self.scheme == "16qam":
            return _16qam_mod(bits)
        raise RuntimeError

    def demodulate(self, symbols) -> "np.ndarray":
        if self.scheme == "bpsk":
            return _bpsk_demod(symbols)
        if self.scheme == "qpsk":
            return _qpsk_demod(symbols)
        if self.scheme == "16qam":
            return _16qam_demod(symbols)
        raise RuntimeError

# -------------------------
# PHY frame builder (preamble + pilots + data)
# -------------------------
def build_phy_frame(bits, mod: Modulator,
                    preamble_len_bits: int = 32,
                    pilot_len_symbols: int = 16
                    ) -> tuple["xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    # Preamble: alternating 1010... (BPSK)
    if preamble_len_bits <= 0:
        pre_sym = xp.zeros(0, dtype=xp.complex128)
    else:
        base = xp.array([1, 0], dtype=xp.uint8)
        pre_bits = xp.tile(base, preamble_len_bits // 2)
        if (preamble_len_bits % 2) == 1:
            pre_bits = xp.concatenate([pre_bits, xp.array([1], dtype=xp.uint8)])
        pre_sym = _bpsk_mod(pre_bits)

    # Pilots: constant QPSK @45°
    pilot = xp.ones(pilot_len_symbols, dtype=xp.complex128) * (1.0 + 1j) / xp.sqrt(2.0)

    # Data
    data_syms = mod.modulate(bits).astype(xp.complex128)

    all_syms = xp.concatenate([pre_sym, pilot, data_syms]) if data_syms.size else xp.concatenate([pre_sym, pilot])
    return all_syms, pilot, data_syms
