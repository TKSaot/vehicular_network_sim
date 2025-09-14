# channel/channel_model.py
from __future__ import annotations
from typing import Tuple
from common.backend import xp, np, asnumpy, to_xp, default_rng, is_cupy

__all__ = ["awgn_channel", "rayleigh_fading", "equalize"]

def awgn_channel(symbols, snr_db: float, seed: int | None = None):
    """
    Complex AWGN: SNR is Es/N0．noise variance per complex dimension is N0/2．
    GPU path (CuPy) if available, else CPU (NumPy).
    """
    z = to_xp(symbols).astype(xp.complex128, copy=False)
    Es = float(xp.mean(xp.abs(z)**2)) if z.size else 1.0
    snr_lin = 10.0**(snr_db/10.0)
    N0 = Es / snr_lin
    if is_cupy:
        rs = default_rng(seed)  # CuPy RandomState
        nr = rs.normal(0, xp.sqrt(N0/2.0), size=z.shape)
        ni = rs.normal(0, xp.sqrt(N0/2.0), size=z.shape)
        noise = (nr + 1j*ni).astype(xp.complex128)
    else:
        rng = default_rng(seed)
        nr = rng.normal(0, np.sqrt(N0/2.0), size=z.shape)
        ni = rng.normal(0, np.sqrt(N0/2.0), size=z.shape)
        noise = (nr + 1j*ni).astype(np.complex128)
        noise = to_xp(noise)
    return (z + noise).astype(xp.complex128)

def rayleigh_fading(symbols,
                    snr_db: float,
                    seed: int | None = None,
                    doppler_hz: float = 30.0,
                    symbol_rate: float = 1e6,
                    block_fading: bool = False,
                    snr_reference: str = "rx") -> Tuple["xp.ndarray", "np.ndarray"]:
    """
    Rayleigh flat fading + AWGN．戻り値は (受信シンボル[xp], フェージング h[n][np])．
    AR(1) fading is generated on CPU (sequential); the output symbols are returned on GPU if enabled.
    """
    x_np = asnumpy(symbols).astype(np.complex128, copy=False)
    N = x_np.size
    rng = default_rng(seed) if not is_cupy else np.random.default_rng(seed)  # ensure NumPy on CPU

    if N == 0:
        return to_xp(x_np), np.ones(0, dtype=np.complex128)

    # AR(1) correlation (CPU)
    Ts = 1.0 / max(1.0, float(symbol_rate))
    fd = abs(float(doppler_hz))
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = min(0.9999, max(0.0, float(rho)))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.ones(N, dtype=np.complex128) * h0
    else:
        v = (rng.normal(0, 1.0, size=N) + 1j*rng.normal(0, 1.0, size=N)) / np.sqrt(2.0)
        h = np.zeros(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    faded = x_np * h

    # Average SNR calibration (CPU)
    Es = float(np.mean(np.abs(x_np)**2)) if N else 1.0
    snr_lin = 10.0**(snr_db/10.0)
    gain = float(np.mean(np.abs(h)**2)) if str(snr_reference).lower() == "rx" else 1.0
    N0 = Es * gain / snr_lin
    nr = rng.normal(0, np.sqrt(N0/2.0), size=N)
    ni = rng.normal(0, np.sqrt(N0/2.0), size=N)
    noise = (nr + 1j*ni).astype(np.complex128)

    y = (faded + noise).astype(np.complex128)
    return to_xp(y), h

def equalize(rx_symbols, tx_pilot, rx_pilot):
    """
    1 タップ等化．h_hat = mean(rx_pilot / tx_pilot)．
    Runs on GPU if available．Returns (eq_symbols[xp], h_hat[python complex]).
    """
    rs = to_xp(rx_symbols).astype(xp.complex128, copy=False)
    tp = to_xp(tx_pilot).astype(xp.complex128, copy=False)
    rp = to_xp(rx_pilot).astype(xp.complex128, copy=False)

    mask = xp.abs(tp) > 1e-12
    # CuPy-friendly check (avoid bool(cp.ndarray)):
    if int(xp.count_nonzero(mask)) == 0:
        return rs, complex(1.0 + 0.0j)

    h_hat = xp.mean(rp[mask] / tp[mask])
    eq = rs / (h_hat + 1e-12)
    # return python complex for logging
    try:
        h_hat_py = complex(asnumpy(h_hat).item())
    except Exception:
        h_hat_py = complex(float(xp.real(h_hat)), float(xp.imag(h_hat)))
    return eq.astype(xp.complex128), h_hat_py
