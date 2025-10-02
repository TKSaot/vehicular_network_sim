from __future__ import annotations
import numpy as np
from typing import Tuple
from .ofdm import grid_to_time, time_to_grid

def rayleigh_ofdm(X: np.ndarray, snr_db: float, seed: int = 12345,
                  n_fft: int = 512, cp_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Nsc, T = X.shape
    H = (rng.normal(0, 1/np.sqrt(2), size=Nsc) + 1j*rng.normal(0, 1/np.sqrt(2), size=Nsc)).astype(np.complex128)
    Yf = (X.T * H).T
    
    tx_td = grid_to_time(Yf, n_fft=n_fft, cp_len=cp_len)
    
    # --- FINAL FIX: CORRECTED NOISE VARIANCE CALCULATION ---
    # The noise variance in the time domain must be scaled by 1/n_fft
    # to achieve the target Es/N0 in the frequency domain after the FFT.
    payload_symbol_energy = 1.0
    snr_lin = 10.0**(snr_db / 10.0)
    noise_variance = (payload_symbol_energy / snr_lin) / n_fft
    # ---------------------------------------------------------

    noise = (rng.normal(0, np.sqrt(noise_variance/2), size=tx_td.shape) + 1j*rng.normal(0, np.sqrt(noise_variance/2), size=tx_td.shape)).astype(np.complex128)
    rx_td = tx_td + noise
    Y = time_to_grid(rx_td, n_fft=n_fft, cp_len=cp_len, Nsc=Nsc, T=T)
    return Y, H

def awgn_ofdm(X: np.ndarray, snr_db: float, seed: int = 12345,
              n_fft: int = 512, cp_len: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Nsc, T = X.shape
    tx_td = grid_to_time(X, n_fft=n_fft, cp_len=cp_len)
    
    # --- FINAL FIX: CORRECTED NOISE VARIANCE CALCULATION ---
    # The noise variance in the time domain must be scaled by 1/n_fft
    # to achieve the target Es/N0 in the frequency domain after the FFT.
    payload_symbol_energy = 1.0
    snr_lin = 10.0**(snr_db / 10.0)
    noise_variance = (payload_symbol_energy / snr_lin) / n_fft
    # ---------------------------------------------------------

    noise = (rng.normal(0, np.sqrt(noise_variance/2), size=tx_td.shape) + 1j*rng.normal(0, np.sqrt(noise_variance/2), size=tx_td.shape)).astype(np.complex128)
    rx_td = tx_td + noise
    return time_to_grid(rx_td, n_fft=n_fft, cp_len=cp_len, Nsc=Nsc, T=T)