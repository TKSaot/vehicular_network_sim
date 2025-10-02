from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from .config import OfdmConfig, LinkConfig
from .utils import bytes_to_bits, bits_to_bytes, block_interleave, block_deinterleave, repeat_bits
from . import hamming74 as ham

def bpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return (2.0*b - 1.0).astype(np.complex128)

def bpsk_demod(symbols: np.ndarray) -> np.ndarray:
    z = np.asarray(symbols, dtype=np.complex128).real
    return (z >= 0).astype(np.uint8)

def make_subcarrier_slices(cfg: OfdmConfig, modalities: List[str]) -> Dict[str, slice]:
    N = cfg.used_subcarriers
    idx = 0
    sl = {}
    for m in modalities:
        w = cfg.subcarrier_split[m]
        n = int(round(N * w))
        if m == modalities[-1]:
            n = N - idx
        sl[m] = slice(idx, idx+n)
        idx += n
    return sl

def assemble_grid(payload_per_mod: Dict[str, bytes],
                  header_per_mod: Dict[str, bytes],
                  cfg_ofdm: OfdmConfig,
                  cfg_link: LinkConfig,
                  power_linear_per_mod: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, slice], Dict[str,int]]:
    mods = list(payload_per_mod.keys())
    sc_slices = make_subcarrier_slices(cfg_ofdm, mods)

    def _enc(b: bytes, rep_k: int) -> np.ndarray:
        bt = bytes_to_bits(b)
        if cfg_link.fec_enabled:
            enc = ham.encode(bt)
        else:
            enc = bt
        if int(rep_k) > 1:
            enc = repeat_bits(enc, int(rep_k))
        inter = block_interleave(enc, cfg_link.interleaver_depth)
        return inter

    hdr_bits = {m: np.tile(_enc(header_per_mod[m], 1), cfg_link.header_rep_k) for m in mods}
    pay_bits = {}
    for m in mods:
        rep_k = getattr(cfg_link, 'payload_rep_k', {}).get(m, 1)
        pay_bits[m] = _enc(payload_per_mod[m], rep_k)

    Nsc = cfg_ofdm.used_subcarriers
    
    def n_syms(bits_len, n_sc_used):
        return int(np.ceil(bits_len / n_sc_used)) if n_sc_used > 0 else 0

    syms_per_mod = {}
    data_grids = {}
    for m in mods:
        hdr_bpsk = bpsk_mod(hdr_bits[m])
        data_bpsk = bpsk_mod(pay_bits[m])

        n_sc_mod = sc_slices[m].stop - sc_slices[m].start
        n_hdr_sym = n_syms(len(hdr_bpsk), n_sc_mod)
        n_dat_sym = n_syms(len(data_bpsk), n_sc_mod)
        T = 1 + n_hdr_sym + n_dat_sym
        syms_per_mod[m] = n_dat_sym

        X = np.zeros((Nsc, T), dtype=np.complex128)
        X[:,0] = 1.0 + 0.0j
        
        sl = sc_slices[m]
        t = 1
        
        if len(hdr_bpsk) > 0:
            hdr_flat = np.pad(hdr_bpsk, (0, n_hdr_sym * n_sc_mod - len(hdr_bpsk)))
            X[sl, t:t+n_hdr_sym] = hdr_flat.reshape(n_hdr_sym, n_sc_mod).T
            t += n_hdr_sym

        if len(data_bpsk) > 0:
            data_flat = np.pad(data_bpsk, (0, n_dat_sym * n_sc_mod - len(data_bpsk)))
            X[sl, t:t+n_dat_sym] = data_flat.reshape(n_dat_sym, n_sc_mod).T
            t += n_dat_sym
        
        # NOTE: power_linear_per_mod (UEP-P) is now applied directly without complex normalization
        scale = np.sqrt(max(1e-12, power_linear_per_mod.get(m, 1.0)))
        if scale != 1.0 and X.shape[1] > 1:
            X[:, 1:] *= scale # Apply power scaling to all non-pilot symbols
        
        data_grids[m] = X

    T_max = max((G.shape[1] for G in data_grids.values()), default=1)
    for m in mods:
        G = data_grids[m]
        if G.shape[1] < T_max:
            pad = np.zeros((Nsc, T_max - G.shape[1]), dtype=np.complex128)
            data_grids[m] = np.concatenate([G, pad], axis=1)

    Xsum = sum(data_grids.values(), np.zeros((Nsc, T_max), dtype=np.complex128))
    # Pilot is not summed, ensure it exists if grid is not empty
    if Xsum.shape[1] > 0:
        Xsum[:, 0] = 1.0 + 0.0j

    # --- MODIFIED: REMOVED ALL POWER BOOSTING AND NORMALIZATION LOGIC ---
    # The system is now simpler and more robust against configuration errors.
    # Header protection relies on repetition + soft decoding.
    
    return Xsum, sc_slices, syms_per_mod

def grid_to_time(X: np.ndarray, n_fft: int, cp_len: int) -> np.ndarray:
    Nsc, T = X.shape
    if T == 0: return np.array([], dtype=np.complex128)
    if Nsc > n_fft:
        raise ValueError("used_subcarriers must be <= n_fft")
    pad_hi = n_fft - Nsc
    td = []
    for t in range(T):
        spec = np.concatenate([X[:,t], np.zeros(pad_hi, dtype=np.complex128)], axis=0)
        x = np.fft.ifft(spec, n=n_fft)
        cp = x[-cp_len:]
        td.append(np.concatenate([cp, x]))
    return np.concatenate(td)

def time_to_grid(y: np.ndarray, n_fft: int, cp_len: int, Nsc: int, T: int) -> np.ndarray:
    sym_len = n_fft + cp_len
    if T == 0: return np.zeros((Nsc, 0), dtype=np.complex128)
    if len(y) < sym_len * T:
        # Pad with zeros if receiver buffer is shorter than expected
        y_padded = np.pad(y, (0, sym_len * T - len(y)))
    else:
        y_padded = y
        
    X = np.zeros((Nsc, T), dtype=np.complex128)
    for t in range(T):
        seg = y_padded[t*sym_len:(t+1)*sym_len]
        seg = seg[cp_len:]
        spec = np.fft.fft(seg, n=n_fft)
        X[:,t] = spec[:Nsc]
    return X