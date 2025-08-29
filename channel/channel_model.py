# channel/channel_model.py
"""
Channel models: AWGN and Rayleigh fading with optional Doppler correlation.
We operate on complex baseband symbols.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def awgn_channel(symbols: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Es/N0 -> noise variance per complex dimension: N0/2 = Es / (2*SNR_linear)
    Es = np.mean(np.abs(symbols)**2) if len(symbols) else 1.0
    snr_lin = 10**(snr_db/10.0)
    N0 = Es / snr_lin
    noise_var = N0
    noise = (rng.normal(0, np.sqrt(noise_var/2), size=symbols.shape) +
             1j * rng.normal(0, np.sqrt(noise_var/2), size=symbols.shape))
    return symbols + noise

def rayleigh_fading(symbols: np.ndarray, snr_db: float, seed: int | None = None,
                    doppler_hz: float = 30.0, symbol_rate: float = 1e6, block_fading: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Rayleigh fading h[n] and AWGN. Returns (rx_symbols, h) where h is the complex fade per symbol.
    If block_fading=True, h is constant over the whole array.
    Otherwise, AR(1) process is used: h[n] = rho * h[n-1] + sqrt(1-rho^2) * v[n], v ~ CN(0,1)
    with rho approx J0(2*pi*fd*Ts) ~ exp(- (2*pi*fd*Ts)^2 / 2) for small arguments.
    """
    rng = np.random.default_rng(seed)
    N = len(symbols)
    if N == 0:
        return symbols.copy(), np.ones(0, dtype=np.complex128)

    Ts = 1.0 / max(1.0, symbol_rate)
    fd = abs(float(doppler_hz))
    # Approximate rho for small fd*Ts: rho ~ exp(-(2*pi*fd*Ts)^2/2). Bound it between 0 and 0.9999
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = min(0.9999, max(0.0, rho))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.ones(N, dtype=np.complex128) * h0
    else:
        v_real = rng.normal(0, 1.0, size=N)
        v_imag = rng.normal(0, 1.0, size=N)
        v = (v_real + 1j*v_imag) / np.sqrt(2.0)  # CN(0,1)
        h = np.zeros(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    faded = symbols * h

    # AWGN on top
    Es = np.mean(np.abs(symbols)**2) if len(symbols) else 1.0
    snr_lin = 10**(snr_db/10.0)
    N0 = Es / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=N) + 1j*rng.normal(0, np.sqrt(N0/2), size=N))

    return faded + noise, h

def equalize(rx_symbols: np.ndarray, tx_pilot: np.ndarray, rx_pilot: np.ndarray) -> tuple[np.ndarray, complex]:
    """
    One-tap equalizer using average pilot-based channel estimate h_hat = mean(rx_pilot / tx_pilot).
    Returns (equalized_symbols, h_hat).
    """
    # Avoid division by zero
    mask = np.abs(tx_pilot) > 1e-12
    if not np.any(mask):
        return rx_symbols, 1+0j
    h_hat = np.mean(rx_pilot[mask] / tx_pilot[mask])
    eq = rx_symbols / (h_hat + 1e-12)
    return eq, h_hat
