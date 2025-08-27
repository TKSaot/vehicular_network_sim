import numpy as np
from common.config import ChannelConfig

class ChannelModel:
    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg

    def _fading(self, n: int) -> np.ndarray:
        c = self.cfg
        ct = c.channel_type.lower()
        if ct == "awgn":
            return np.ones(n, dtype=np.complex128)

        if ct == "rayleigh":
            if not c.time_selective:
                h = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                return np.full(n, h, dtype=np.complex128)
            # time-selective
            if c.fading_model == "block":
                L = max(1, int(c.coherence_symbols))
                blocks = (n+L-1)//L
                hh = (np.random.randn(blocks)+1j*np.random.randn(blocks))/np.sqrt(2)
                h = np.repeat(hh, L)[:n]
                return h.astype(np.complex128)
            else:  # gauss_markov
                rho = float(c.rho)
                w = (np.random.randn(n)+1j*np.random.randn(n))/np.sqrt(2)
                h = np.zeros(n, dtype=np.complex128)
                h[0] = w[0]
                for i in range(1,n):
                    h[i] = rho*h[i-1] + np.sqrt(1-rho**2)*w[i]
                return h

        if ct == "rician":
            K = float(self.cfg.rician_K)
            mu = np.sqrt(K/(K+1))
            sig = 1/np.sqrt(2*(K+1))
            if not c.time_selective:
                scat = (np.random.randn()+1j*np.random.randn())*sig
                h = mu + scat
                return np.full(n, h, dtype=np.complex128)
            if c.fading_model == "block":
                L = max(1, int(c.coherence_symbols))
                blocks = (n+L-1)//L
                scat = (np.random.randn(blocks)+1j*np.random.randn(blocks))*sig
                hh = mu + scat
                h = np.repeat(hh, L)[:n]
                return h.astype(np.complex128)
            else:
                rho = float(c.rho)
                w = (np.random.randn(n)+1j*np.random.randn(n))*sig*np.sqrt(2)
                h = np.zeros(n, dtype=np.complex128)
                h[0] = mu + w[0]
                for i in range(1,n):
                    h[i] = mu + rho*(h[i-1]-mu) + np.sqrt(1-rho**2)*w[i]
                return h

        raise ValueError("unknown channel type " + ct)

    def transmit(self, s: np.ndarray, esn0_db: float):
        n = len(s)
        h = self._fading(n)
        esn0_lin = 10.0**(esn0_db/10.0)
        # noise variance per complex sample = 1/esn0
        sigma = np.sqrt(1.0/(2.0*esn0_lin))
        w = (np.random.randn(n)+1j*np.random.randn(n))*sigma
        y = h * s + w
        return y, h
