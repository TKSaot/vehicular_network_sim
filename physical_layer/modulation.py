# physical_layer/modulation.py
"""
Mapping bits to complex baseband symbols (BPSK, QPSK, 16QAM) and back (hard decisions).
Also builds per-frame symbol sequences by adding a preamble and pilots.

⚠️ 重要: 整数 (uint8) 演算のまま I/Q を計算するとアンダーフローします。
必ず float にキャストしてから 1 - 2*b などの演算を行ってください。
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# -------------------------
# BPSK
# -------------------------
def _bpsk_mod(bits: np.ndarray) -> np.ndarray:
    # 0 -> -1, 1 -> +1 (real)
    return 2.0 * bits.astype(np.float64) - 1.0

def _bpsk_demod(symbols: np.ndarray) -> np.ndarray:
    return (symbols.real >= 0).astype(np.uint8)

# -------------------------
# QPSK (Gray)
# ビット [b0, b1] -> (I, Q) = (1-2*b0, 1-2*b1) / sqrt(2)
# -------------------------
def _qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(b) % 2 != 0:
        b = np.concatenate([b, np.zeros(1, dtype=np.uint8)])
    # ⚠️ float化してから演算（uint8のままだと 1-2*b でアンダーフロー）
    pairs = b.reshape(-1, 2).astype(np.float64)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]
    syms = (i + 1j * q) / np.sqrt(2.0)
    return syms

def _qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    i = (symbols.real < 0).astype(np.uint8)
    q = (symbols.imag < 0).astype(np.uint8)
    out = np.vstack([i, q]).T.reshape(-1)
    return out

# -------------------------
# 16QAM (Gray)
# 2ビット -> レベル in {-3, -1, +1, +3}
# (b1,b0) -> amp = (1 - 2*b1) * (3 - 2*b0)
# -------------------------
def _16qam_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(b)) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    quads = b.reshape(-1, 4).astype(np.float64)  # float化
    # I: (b1,b0) = (q[:,0], q[:,1]), Q: (q[:,2], q[:,3])
    I = (1.0 - 2.0 * quads[:, 0]) * (3.0 - 2.0 * quads[:, 1])
    Q = (1.0 - 2.0 * quads[:, 2]) * (3.0 - 2.0 * quads[:, 3])
    syms = (I + 1j * Q) / np.sqrt(10.0)
    return syms

def _16qam_demod(symbols: np.ndarray) -> np.ndarray:
    x = symbols * np.sqrt(10.0)
    I = x.real
    Q = x.imag
    # しきい値: -2, 0, +2 （レベル: -3, -1, +1, +3）
    def level_to_bits(vals):
        b1 = (vals < 0).astype(np.uint8)            # MSB
        b0 = (np.abs(vals) < 2).astype(np.uint8)    # 内側:1, 外側:0
        return b1, b0
    i1, i0 = level_to_bits(I)
    q1, q0 = level_to_bits(Q)
    bits = np.vstack([i1, i0, q1, q0]).T.reshape(-1)
    return bits.astype(np.uint8)

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

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if self.scheme == "bpsk":
            return _bpsk_mod(bits)
        if self.scheme == "qpsk":
            return _qpsk_mod(bits)
        if self.scheme == "16qam":
            return _16qam_mod(bits)
        raise RuntimeError

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
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
def build_phy_frame(bits: np.ndarray, mod: Modulator,
                    preamble_len_bits: int = 32,
                    pilot_len_symbols: int = 16
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce a sequence of symbols for one link-layer frame:
    - preamble (BPSK on 1010...) for visibility (not used for sync here)
    - pilot (QPSK constant (1+j)/√2) for 1-tap channel estimation
    - data symbols (per selected modulation)
    Returns (symbols_concat, tx_pilot_symbols, data_symbols_only)
    """
    # Preamble: alternating 1010... (BPSK, real)
    pre_bits = np.tile(np.array([1, 0], dtype=np.uint8), preamble_len_bits // 2)
    pre_sym = _bpsk_mod(pre_bits).astype(np.complex128)  # 実軸 -> 複素

    # Pilots: constant QPSK @45°
    pilot = np.ones(pilot_len_symbols, dtype=np.complex128) * (1.0 + 1j) / np.sqrt(2.0)

    # Data
    data_syms = mod.modulate(bits).astype(np.complex128)

    all_syms = np.concatenate([pre_sym, pilot, data_syms])
    return all_syms, pilot, data_syms
