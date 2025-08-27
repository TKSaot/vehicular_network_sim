import numpy as np
from typing import Tuple

def bits_per_symbol(mod: str) -> int:
    m = mod.upper()
    if m == "BPSK": return 1
    if m == "QPSK": return 2
    if m == "16QAM": return 4
    raise ValueError("unknown modulation " + mod)

# 平均Es=1に正規化
def modulate(bits: np.ndarray, mod: str) -> np.ndarray:
    mod = mod.upper()
    if mod == "BPSK":
        b = bits.astype(np.float64)
        # 0 -> +1, 1 -> -1
        s = 1.0 - 2.0*b
        return s.astype(np.complex128)

    if mod == "QPSK":
        b = bits.astype(np.uint8)
        pad = (-len(b)) % 2
        if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        b = b.reshape(-1,2)
        # Gray: 00->(1+1j)/√2, 01->(-1+1j)/√2, 11->(-1-1j)/√2, 10->(1-1j)/√2
        m = (1-2*b[:,0]) + 1j*(1-2*b[:,1])
        return (m/np.sqrt(2)).astype(np.complex128)

    if mod == "16QAM":
        b = bits.astype(np.uint8)
        pad = (-len(b)) % 4
        if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        b = b.reshape(-1,4)
        # 2bit -> Gray level {-3,-1,+1,+3}
        def gray2pam(u,v):
            g = (u<<1) ^ v
            return np.array([-3,-1,1,3], dtype=np.int32)[g]
        I = gray2pam(b[:,0], b[:,1])
        Q = gray2pam(b[:,2], b[:,3])
        s = (I + 1j*Q) / np.sqrt(10)  # Es=1
        return s.astype(np.complex128)

    raise ValueError("unknown modulation " + mod)

def demod_hard(x: np.ndarray, mod: str) -> np.ndarray:
    mod = mod.upper()
    if mod == "BPSK":
        bits = (np.real(x) < 0).astype(np.uint8)
        return bits

    if mod == "QPSK":
        b0 = (np.real(x) < 0).astype(np.uint8)
        b1 = (np.imag(x) < 0).astype(np.uint8)
        return np.vstack([b0,b1]).T.reshape(-1)

    if mod == "16QAM":
        # 4レベル判定
        def slicer(z):
            # レベル境界: -2,0,2 （正規化済み: /√10）
            r = np.real(z) * np.sqrt(10)
            i = np.imag(z) * np.sqrt(10)
            def to_bits(v):
                # -inf..-2 -> 00, -2..0 -> 01, 0..2 -> 11, 2..inf -> 10 （Gray逆写像）
                if v < -2:   return (0,0)
                elif v < 0:  return (0,1)
                elif v < 2:  return (1,1)
                else:        return (1,0)
            bI0,bI1 = np.vectorize(lambda val: to_bits(val))(r)
            bQ0,bQ1 = np.vectorize(lambda val: to_bits(val))(i)
            return np.vstack([bI0,bI1,bQ0,bQ1]).T.reshape(-1,4)
        B = slicer(x)
        return B.reshape(-1)

    raise ValueError("unknown modulation " + mod)
