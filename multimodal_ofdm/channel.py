# multimodal_ofdm/channel.py
# ---------------------------------------------------------------------
# Channel models with per-sample i.i.d. noise/fading to avoid horizontal
# banding artifacts. Provides AWGN, Rayleigh, and a helper equalizer.
# This file is self-contained (NumPy only) and fits the current project
# layout: ~/work/vehicular_network_sim/{multimodal_ofdm,examples}
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

ComplexND = np.ndarray


def _es_avg(x: ComplexND) -> float:
    """Average symbol energy Es (per complex sample)."""
    x = np.asarray(x, dtype=np.complex128)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) ** 2))


# ----------------------------- AWGN ---------------------------------- #
def awgn_channel(x: ComplexND,
                 snr_db: float,
                 seed: Optional[int] = None,
                 snr_reference: str = "tx") -> ComplexND:
    """
    Add complex AWGN with variance chosen from SNR.

    Args
    ----
    x : np.ndarray (complex128)
        Time‑domain samples to be corrupted by noise.
    snr_db : float
        Target Es/N0 in dB. By default measured at TX (before channel).
    seed : int | None
        Seed for RNG. Different from symbol index → no patterning.
    snr_reference : {"tx","rx"}
        When "rx" is selected the average channel power is assumed 1,
        so this behaves the same as "tx" for AWGN.

    Returns
    -------
    y : np.ndarray (complex128)
        x + n, where n ~ CN(0, N0).
    """
    z = np.asarray(x, dtype=np.complex128)
    if z.size == 0:
        return z

    Es = _es_avg(z)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    N0 = Es / max(snr_lin, 1e-20)

    rng = np.random.default_rng(seed)
    nr = rng.normal(0.0, np.sqrt(N0 / 2.0), size=z.shape)
    ni = rng.normal(0.0, np.sqrt(N0 / 2.0), size=z.shape)
    noise = (nr + 1j * ni).astype(np.complex128)
    return (z + noise).astype(np.complex128)


# --------------------------- Rayleigh -------------------------------- #
def rayleigh_fading(x: ComplexND,
                    snr_db: float,
                    seed: Optional[int] = None,
                    doppler_hz: float = 30.0,
                    symbol_rate: float = 1e6,
                    block_fading: bool = False,
                    snr_reference: str = "rx") -> Tuple[ComplexND, np.ndarray]:
    """
    Flat Rayleigh fading followed by AWGN.

    * Fading is generated per *sample* to avoid striping:
      - block_fading=True  → one coefficient per frame (constant)
      - block_fading=False → Gauss‑Markov AR(1) (Jakes‑like) sequence
    * SNR is defined as Es/N0. With snr_reference="rx", the average
      channel power E[|h|^2] is taken into account for N0.

    Returns
    -------
    y : np.ndarray (complex128)  -- faded + noisy signal
    h : np.ndarray (complex128)  -- channel taps per sample (CPU numpy)
    """
    z = np.asarray(x, dtype=np.complex128)
    N = z.size
    if N == 0:
        return z, np.ones(0, dtype=np.complex128)

    rng = np.random.default_rng(seed)

    # --- Generate fading h[n] on CPU ---
    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))).astype(np.complex128)
        h = np.full(N, h0, dtype=np.complex128)
    else:
        # Gauss‑Markov AR(1) with correlation set by Doppler
        Ts = 1.0 / max(1.0, float(symbol_rate))
        fd = abs(float(doppler_hz))
        rho = np.exp(-0.5 * (2.0 * np.pi * fd * Ts) ** 2)  # small‑angle approx of Jakes ACF
        rho = float(np.clip(rho, 0.0, 0.9999))
        w = (rng.normal(0.0, 1.0, size=N) + 1j * rng.normal(0.0, 1.0, size=N)) / np.sqrt(2.0)
        h = np.empty(N, dtype=np.complex128)
        h[0] = w[0]
        a = np.sqrt(max(0.0, 1.0 - rho ** 2))
        for n in range(1, N):
            h[n] = rho * h[n - 1] + a * w[n]

    # --- Apply fading & add AWGN (per sample) ---
    y = (z * h).astype(np.complex128)

    Es = _es_avg(z)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    ch_gain = float(np.mean(np.abs(h) ** 2)) if str(snr_reference).lower() == "rx" else 1.0
    N0 = Es * ch_gain / max(snr_lin, 1e-20)

    # Use a *different* RNG stream for noise to avoid correlation with h
    rng2 = np.random.default_rng(None if seed is None else (seed + 1))
    nr = rng2.normal(0.0, np.sqrt(N0 / 2.0), size=N)
    ni = rng2.normal(0.0, np.sqrt(N0 / 2.0), size=N)
    noise = (nr + 1j * ni).astype(np.complex128)

    y = (y + noise).astype(np.complex128)
    return y, h


# --------------------------- Equalizer -------------------------------- #
def equalize(rx_symbols: ComplexND, tx_pilot: ComplexND, rx_pilot: ComplexND
             ) -> Tuple[ComplexND, complex]:
    """
    Single‑tap equalizer using pilot estimate:
        ĥ = mean(rx_pilot / tx_pilot)
        ŝ = rx_symbols / ĥ
    Returns equalized data symbols and the complex channel estimate.
    """
    rs = np.asarray(rx_symbols, dtype=np.complex128).reshape(-1)
    tp = np.asarray(tx_pilot, dtype=np.complex128).reshape(-1)
    rp = np.asarray(rx_pilot, dtype=np.complex128).reshape(-1)

    mask = np.abs(tp) > 1e-12
    if not np.any(mask):
        return rs, complex(1.0 + 0.0j)

    h_hat = np.mean(rp[mask] / tp[mask])
    eq = rs / (h_hat + 1e-12)
    return eq.astype(np.complex128), complex(h_hat)


# ------------------------ Convenience wrapper ------------------------- #
def apply_channel(kind: str,
                  x: ComplexND,
                  snr_db: float,
                  seed: Optional[int] = None,
                  doppler_hz: float = 30.0,
                  symbol_rate: float = 1e6,
                  block_fading: bool = False,
                  snr_reference: str = "rx") -> Tuple[ComplexND, Optional[np.ndarray]]:
    """
    Dispatch to AWGN or Rayleigh based on a string.
    Returns (rx, h_or_None). 'h_or_None' is None for AWGN.
    """
    k = str(kind).lower()
    if k in ("awgn", "none"):
        return awgn_channel(x, snr_db, seed=seed, snr_reference=snr_reference), None
    elif k in ("rayleigh", "rayleigh_flat", "flat"):
        return rayleigh_fading(
            x, snr_db,
            seed=seed,
            doppler_hz=doppler_hz,
            symbol_rate=symbol_rate,
            block_fading=block_fading,
            snr_reference=snr_reference,
        )
    raise ValueError(f"Unknown channel kind: {kind}")
