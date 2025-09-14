# channel/channel_model.py
from __future__ import annotations
from typing import Tuple
from common.backend import xp, np, asnumpy, to_xp, default_rng, is_cupy

__all__ = ["awgn_channel", "rayleigh_fading", "equalize"]

def awgn_channel(symbols, snr_db: float, seed: int | None = None):
    """
    Complex AWGN on GPU if available. SNR is Es/N0 (per complex symbol).
    """
    z = to_xp(symbols).astype(xp.complex128, copy=False)
    if z.size == 0:
        return z
    Es = float(xp.mean(xp.abs(z)**2).item())
    snr_lin = 10.0**(snr_db/10.0)
    N0 = Es / snr_lin

    if is_cupy:
        rs = default_rng(seed)
        nr = rs.normal(0, xp.sqrt(N0/2.0), size=z.shape)
        ni = rs.normal(0, xp.sqrt(N0/2.0), size=z.shape)
        noise = (nr + 1j*ni).astype(xp.complex128)
    else:
        rng = default_rng(seed)
        nr = rng.normal(0, np.sqrt(N0/2.0), size=z.shape)
        ni = rng.normal(0, np.sqrt(N0/2.0), size=z.shape)
        noise = to_xp((nr + 1j*ni).astype(np.complex128))

    return (z + noise).astype(xp.complex128)

def rayleigh_fading(symbols,
                    snr_db: float,
                    seed: int | None = None,
                    doppler_hz: float = 30.0,
                    symbol_rate: float = 1e6,
                    block_fading: bool = False,
                    snr_reference: str = "rx") -> Tuple["xp.ndarray", "np.ndarray"]:
    """
    Rayleigh flat fading + AWGN.
    * AR(1) fading process h[n] is generated on CPU (NumPy) for stability.
    * Fading application (z * h) and AWGN addition are done on GPU if available.
    Returns: (rx_symbols [xp], h[n] [numpy]).
    """
    z = to_xp(symbols).astype(xp.complex128, copy=False)
    N = int(z.size)
    if N == 0:
        return z, np.ones(0, dtype=np.complex128)

    # --- CPU: generate h[n] (sequential AR(1)) ---
    rng = np.random.default_rng(seed)
    Ts = 1.0 / max(1.0, float(symbol_rate))
    fd = abs(float(doppler_hz))
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = float(min(0.9999, max(0.0, rho)))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.full(N, h0, dtype=np.complex128)
    else:
        v = (rng.normal(0, 1.0, size=N) + 1j*rng.normal(0, 1.0, size=N)) / np.sqrt(2.0)
        h = np.empty(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    # --- GPU: apply fading & AWGN ---
    h_gpu = to_xp(h, dtype=xp.complex128)
    y = (z * h_gpu).astype(xp.complex128)

    # Es measured on GPU; average channel power on CPU
    Es = float(xp.mean(xp.abs(z)**2).item())
    snr_lin = 10.0**(snr_db/10.0)
    gain = float(np.mean(np.abs(h)**2)) if str(snr_reference).lower() == "rx" else 1.0
    N0 = Es * gain / snr_lin

    if is_cupy:
        rs = default_rng(None if seed is None else seed + 1)
        nr = rs.normal(0, xp.sqrt(N0/2.0), size=N)
        ni = rs.normal(0, xp.sqrt(N0/2.0), size=N)
        noise = (nr + 1j*ni).astype(xp.complex128)
    else:
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        nr = rng2.normal(0, np.sqrt(N0/2.0), size=N)
        ni = rng2.normal(0, np.sqrt(N0/2.0), size=N)
        noise = to_xp((nr + 1j*ni).astype(np.complex128))

    y = (y + noise).astype(xp.complex128)
    return y, h

def equalize(rx_symbols, tx_pilot, rx_pilot):
    """
    Single-tap equalizer using pilot: ĥ = mean(rx_pilot / tx_pilot).
    """
    rs = to_xp(rx_symbols).astype(xp.complex128, copy=False)
    tp = to_xp(tx_pilot).astype(xp.complex128, copy=False)
    rp = to_xp(rx_pilot).astype(xp.complex128, copy=False)

    mask = xp.abs(tp) > 1e-12
    # CuPy-safe check (avoid bool(cp.ndarray))
    if int(xp.count_nonzero(mask)) == 0:
        return rs, complex(1.0 + 0.0j)

    h_hat = xp.mean(rp[mask] / tp[mask])
    eq = rs / (h_hat + 1e-12)

    try:
        h_hat_py = complex(asnumpy(h_hat).item())
    except Exception:
        h_hat_py = complex(float(xp.real(h_hat)), float(xp.imag(h_hat)))

    return eq.astype(xp.complex128), h_hat_py
