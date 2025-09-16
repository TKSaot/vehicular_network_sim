
# multimodal_ofdm/channel.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from .ofdm import grid_to_time, time_to_grid

def rayleigh_ofdm(X: np.ndarray, snr_db: float, seed: int = 12345,
                  n_fft: int = 512, cp_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Apply per-subcarrier flat Rayleigh fading + AWGN.
    We realize the waveform in time-domain with IFFT+CP, then add AWGN.
    Fading is modeled as one complex coefficient per subcarrier constant over time (block fading).
    Returns freq-domain Y and channel H per subcarrier."""
    rng = np.random.default_rng(seed)
    Nsc, T = X.shape
    # Channel per subcarrier
    H = (rng.normal(0, 1/np.sqrt(2), size=Nsc) + 1j*rng.normal(0, 1/np.sqrt(2), size=Nsc)).astype(np.complex128)
    Yf = (X.T * H).T

    # Time-domain signal with CP and AWGN
    tx_td = grid_to_time(Yf, n_fft=n_fft, cp_len=cp_len)
    # Es measured on TD excluding CP
    sym_energy = np.mean(np.abs(tx_td)**2)
    snr_lin = 10.0**(snr_db/10.0)
    N0 = sym_energy / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=tx_td.shape) + 1j*rng.normal(0, np.sqrt(N0/2), size=tx_td.shape)).astype(np.complex128)
    rx_td = tx_td + noise
    # Back to frequency domain (used carriers only)
    Y = time_to_grid(rx_td, n_fft=n_fft, cp_len=cp_len, Nsc=Nsc, T=T)
    return Y, H

def awgn_ofdm(X: np.ndarray, snr_db: float, seed: int = 12345,
              n_fft: int = 512, cp_len: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Nsc, T = X.shape
    tx_td = grid_to_time(X, n_fft=n_fft, cp_len=cp_len)
    sym_energy = np.mean(np.abs(tx_td)**2)
    snr_lin = 10.0**(snr_db/10.0)
    N0 = sym_energy / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=tx_td.shape) + 1j*rng.normal(0, np.sqrt(N0/2), size=tx_td.shape)).astype(np.complex128)
    rx_td = tx_td + noise
    return time_to_grid(rx_td, n_fft=n_fft, cp_len=cp_len, Nsc=Nsc, T=T)
