# channel/channel_model.py
"""
Channel models: AWGN and Rayleigh fading with optional Doppler correlation.
We operate on complex baseband symbols.
"""

from __future__ import annotations
import numpy as np

def awgn_channel(symbols: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Es = np.mean(np.abs(symbols)**2) if len(symbols) else 1.0
    snr_lin = 10**(snr_db/10.0)
    N0 = Es / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=symbols.shape) +
             1j * rng.normal(0, np.sqrt(N0/2), size=symbols.shape))
    return symbols + noise

def rayleigh_fading(symbols: np.ndarray, snr_db: float, seed: int | None = None,
                    doppler_hz: float = 30.0, symbol_rate: float = 1e6,
                    block_fading: bool = False,
                    snr_reference: str = "rx") -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Rayleigh fading h[n] and AWGN. Returns (rx_symbols, h).

    snr_reference:
      - "tx": noise set w.r.t. transmit Es  -> N0 = Es / SNR
      - "rx": noise set w.r.t. avg receive power Es*E[|h|^2] -> N0 = Es*E[|h|^2] / SNR
    """
    rng = np.random.default_rng(seed)
    N = len(symbols)
    if N == 0:
        return symbols.copy(), np.ones(0, dtype=np.complex128)

    Ts = 1.0 / max(1.0, float(symbol_rate))
    fd = abs(float(doppler_hz))
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = min(0.9999, max(0.0, rho))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.ones(N, dtype=np.complex128) * h0
    else:
        v = (rng.normal(0, 1.0, size=N) + 1j*rng.normal(0, 1.0, size=N)) / np.sqrt(2.0)  # CN(0,1)
        h = np.zeros(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    faded = symbols * h

    # --- Average SNR calibration ---
    Es = np.mean(np.abs(symbols)**2) if len(symbols) else 1.0
    snr_lin = 10**(snr_db/10.0)
    ref = str(snr_reference).lower()
    if ref == "rx":
        gain = float(np.mean(np.abs(h)**2))  # ≈ 1.0 in theory, but use sample average
    else:
        gain = 1.0
    N0 = Es * gain / snr_lin

    noise = (rng.normal(0, np.sqrt(N0/2), size=N) + 1j*rng.normal(0, np.sqrt(N0/2), size=N))
    return faded + noise, h

def equalize(rx_symbols: np.ndarray, tx_pilot: np.ndarray, rx_pilot: np.ndarray) -> tuple[np.ndarray, complex]:
    """
    One-tap equalizer using average pilot-based channel estimate h_hat = mean(rx_pilot / tx_pilot).
    Returns (equalized_symbols, h_hat).
    """
    mask = np.abs(tx_pilot) > 1e-12
    if not np.any(mask):
        return rx_symbols, 1+0j
    h_hat = np.mean(rx_pilot[mask] / tx_pilot[mask])
    eq = rx_symbols / (h_hat + 1e-12)
    return eq, h_hat
