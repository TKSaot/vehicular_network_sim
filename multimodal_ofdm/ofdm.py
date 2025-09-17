
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from .config import OfdmConfig
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
    # center DC removed: we just index [0..N-1] for simplicity and ignore actual RF mapping
    starts = {}
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
                  cfg_link,  # has interleaver_depth, header_rep_k
                  power_linear_per_mod: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, slice], Dict[str,int]]:
    """Return frequency-domain resource grid X[f, t] including one all-ones pilot at t=0.
    Also return modality->slice mapping and symbol counts per modality (data only)."""
    mods = list(payload_per_mod.keys())
    sc_slices = make_subcarrier_slices(cfg_ofdm, mods)

    # 1) Convert bytes -> bits -> Hamming -> interleave
    def _enc(b: bytes, rep_k: int) -> np.ndarray:
        bt = bytes_to_bits(b)
        enc = ham.encode(bt)
        if int(rep_k) > 1:
            enc = repeat_bits(enc, int(rep_k))
        inter = block_interleave(enc, cfg_link.interleaver_depth)
        return inter

    # Repeat & boost headers to be robust (constant across EEP/EEP)
    hdr_bits = {m: np.tile(_enc(header_per_mod[m], 1), cfg_link.header_rep_k) for m in mods}
    pay_bits = {}
    for m in mods:
        rep_k = getattr(cfg_link, 'payload_rep_k', {}).get(m, 1)
        pay_bits[m] = _enc(payload_per_mod[m], rep_k)

    # 2) Map to BPSK with power scaling; first OFDM symbol is pilot=1+0j
    Nsc = cfg_ofdm.used_subcarriers
    # Determine required number of OFDM DATA symbols
    def n_syms(bits_len, n_sc_used):
        return int(np.ceil(bits_len / n_sc_used))

    # Header first (one block), then data
    syms_per_mod = {}
    data_grids = {}
    for m in mods:
        # BPSK streams
        hdr_bpsk = bpsk_mod(hdr_bits[m])
        data_bpsk = bpsk_mod(pay_bits[m])

        # number of symbols needed
        n_hdr_sym = n_syms(len(hdr_bpsk), sc_slices[m].stop - sc_slices[m].start)
        n_dat_sym = n_syms(len(data_bpsk), sc_slices[m].stop - sc_slices[m].start)
        T = 1 + n_hdr_sym + n_dat_sym  # +1 for pilot
        syms_per_mod[m] = n_dat_sym  # for reporting

        # build grid
        X = np.zeros((Nsc, T), dtype=np.complex128)
        # pilot
        X[:,0] = 1.0 + 0.0j
        # place header then data into slice
        sl = sc_slices[m]
        t = 1
        # header
        H = len(hdr_bpsk)
        per = sl.stop - sl.start
        if H > 0:
            n = n_syms(H, per)
            for i in range(n):
                seg = hdr_bpsk[i*per:(i+1)*per]
                col = np.zeros(Nsc, dtype=np.complex128)
                col[sl][0:len(seg)] = seg
                X[:, t] = col; t += 1
        # data
        D = len(data_bpsk)
        if D > 0:
            n = n_syms(D, per)
            for i in range(n):
                seg = data_bpsk[i*per:(i+1)*per]
                col = np.zeros(Nsc, dtype=np.complex128)
                col[sl][0:len(seg)] = seg
                X[:, t] = col; t += 1

        # Apply modality power scaling (normalized later by total average power)
        # Scale data columns only (pilot left at 1), headers get the same modality scale; external header boost is applied outside.
        scale = np.sqrt(max(1e-12, power_linear_per_mod[m]))
        X[:,1:] *= scale

        data_grids[m] = X

    # Align time dimension across modalities by padding columns (zero)
    T_max = max(G.shape[1] for G in data_grids.values())
    for m in mods:
        G = data_grids[m]
        if G.shape[1] < T_max:
            pad = np.zeros((Nsc, T_max - G.shape[1]), dtype=np.complex128)
            data_grids[m] = np.concatenate([G, pad], axis=1)

    # Sum grids from all modalities (frequency-domain superposition)
    Xsum = sum(data_grids[m] for m in mods)

    # Header boost (constant, not part of UEP/EEP comparison)
    if cfg_link.header_boost_db != 0.0:
        boost = 10.0**(cfg_link.header_boost_db/20.0)
        Xsum[:, 1] *= boost  # only the first header column (they all align at t=1)

    # Normalize total average power to 1.0
    Es = np.mean(np.abs(Xsum[:,1:])**2) if Xsum.shape[1] > 1 else 1.0
    if Es > 0: Xsum[:,1:] /= np.sqrt(Es)

    return Xsum, sc_slices, syms_per_mod

def extract_per_mod_bits(Yeq: np.ndarray, sc_slices: Dict[str, slice], cfg_ofdm: OfdmConfig,
                         cfg_link, total_cols: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Equalized Yeq (after dividing by channel, pilot at t=0 removed by caller).
    Returns dict m -> (header_bits_decoded, payload_bits_decoded)."""
    out = {}
    Nsc = cfg_ofdm.used_subcarriers
    for m, sl in sc_slices.items():
        # Collect columns 1.. for the slice
        sub = Yeq[sl, 1:total_cols]  # exclude pilot
        # Read row-wise then flatten per column
        syms = sub.reshape(-1)
        bits = bpsk_demod(syms)
        # Split header vs payload using the known header length (we can't know it here cleanly),
        # so we expect the caller to provide the exact number of header bits emitted at TX.
        # For simplicity we assume the first column after pilot (=col 1) is entirely header,
        # which is true with our construction because we repeat/pad header to fill that column.
        per = sl.stop - sl.start
        hdr_len = per  # equals one column for all mods in this simplified demux
        hdr_bits = bits[:hdr_len]
        pay_bits = bits[hdr_len:]
        out[m] = (hdr_bits, pay_bits)
    return out

def decode_stream(hdr_bits: np.ndarray, pay_bits: np.ndarray, cfg_link, original_bit_len_hdr: int, original_bit_len_pay: int) -> tuple[bytes, bytes]:
    # Deinterleave -> Hamming decode -> de-pad to original bit lengths
    from .utils import bits_to_bytes
    def _dec(b: np.ndarray, L: int) -> bytes:
        de = block_deinterleave(b, cfg_link.interleaver_depth)
        d  = ham.decode(de)
        if len(d) > L: d = d[:L]
        return bits_to_bytes(d)
    return _dec(hdr_bits, original_bit_len_hdr), _dec(pay_bits, original_bit_len_pay)


# ---- IFFT/FFT helpers (optional) ----
def grid_to_time(X: np.ndarray, n_fft: int, cp_len: int) -> np.ndarray:
    """X: [Nsc, T] with zeros already in unused carriers.
    Return time-domain signal with CP concatenated for all OFDM symbols."""
    Nsc, T = X.shape
    # We'll embed the Nsc used carriers into n_fft by zero-padding edges evenly.
    # For simplicity, map to lowest bins [0..Nsc-1]. In a practical system we'd center around DC.
    if Nsc > n_fft:
        raise ValueError("used_subcarriers must be <= n_fft")
    pad_lo = 0
    pad_hi = n_fft - Nsc
    td = []
    for t in range(T):
        spec = np.concatenate([X[:,t], np.zeros(pad_hi, dtype=np.complex128)], axis=0)
        x = np.fft.ifft(spec, n=n_fft)
        cp = x[-cp_len:]
        td.append(np.concatenate([cp, x]))
    return np.concatenate(td)

def time_to_grid(y: np.ndarray, n_fft: int, cp_len: int, Nsc: int, T: int) -> np.ndarray:
    """Inverse of grid_to_time. Extract per-symbol FFT and return used carriers only."""
    sym_len = n_fft + cp_len
    if len(y) < sym_len * T:
        raise ValueError("RX buffer shorter than expected")
    X = np.zeros((Nsc, T), dtype=np.complex128)
    for t in range(T):
        seg = y[t*sym_len:(t+1)*sym_len]
        seg = seg[cp_len:]  # remove CP
        spec = np.fft.fft(seg, n=n_fft)
        X[:,t] = spec[:Nsc]
    return X
