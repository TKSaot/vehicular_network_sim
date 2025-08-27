import numpy as np
from typing import Protocol

class Code(Protocol):
    def encode(self, u: np.ndarray) -> np.ndarray: ...
    def decode(self, v: np.ndarray) -> np.ndarray: ...

# ---- NONE ----
class CodeNone:
    def encode(self, u: np.ndarray) -> np.ndarray:
        return u.astype(np.uint8)
    def decode(self, v: np.ndarray) -> np.ndarray:
        return v.astype(np.uint8)

# ---- repetition(3,1) ----
class CodeRepetition3:
    def encode(self, u: np.ndarray) -> np.ndarray:
        return np.repeat(u.astype(np.uint8), 3)
    def decode(self, v: np.ndarray) -> np.ndarray:
        if len(v) == 0:
            return v.astype(np.uint8)
        m = len(v) // 3
        vv = v[:m*3].reshape(-1,3)
        s = vv.sum(axis=1)
        return (s >= 2).astype(np.uint8)

# ---- Hamming(7,4) ----
# G = [1 0 0 0 | 1 1 0
#      0 1 0 0 | 1 0 1
#      0 0 1 0 | 1 0 0
#      0 0 0 1 | 0 1 1]
# H = [1 1 1 0 1 0 0
#      1 0 0 1 0 1 0
#      0 1 0 1 0 0 1]
G = np.array([
    [1,0,0,0, 1,1,0],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 1,0,0],
    [0,0,0,1, 0,1,1],
], dtype=np.uint8)
H = np.array([
    [1,1,1,0,1,0,0],
    [1,0,0,1,0,1,0],
    [0,1,0,1,0,0,1],
], dtype=np.uint8)
SYN_TO_ERR = {
    (1,0,0): 0,
    (0,1,0): 1,
    (0,0,1): 2,
    (1,1,0): 3,
    (1,0,1): 4,
    (0,1,1): 5,
    (1,1,1): 6,
}

class CodeHamming74:
    def encode(self, u: np.ndarray) -> np.ndarray:
        if len(u) == 0:
            return u.astype(np.uint8)
        pad = (-len(u)) % 4
        if pad:
            u = np.concatenate([u.astype(np.uint8), np.zeros(pad, dtype=np.uint8)])
        U = u.reshape(-1,4)
        V = (U @ G) % 2
        return V.reshape(-1).astype(np.uint8)

    def decode(self, v: np.ndarray) -> np.ndarray:
        if len(v) == 0:
            return v.astype(np.uint8)
        pad = (-len(v)) % 7
        if pad:
            v = np.concatenate([v.astype(np.uint8), np.zeros(pad, dtype=np.uint8)])
        V = v.reshape(-1,7)
        S = (V @ H.T) % 2
        out = V.copy()
        for i in range(len(V)):
            syn = tuple(int(x) for x in S[i])
            if syn != (0,0,0):
                pos = SYN_TO_ERR.get(syn, None)
                if pos is not None:
                    out[i, pos] ^= 1
        # 先頭4bitが情報
        Uhat = out[:,:4].reshape(-1)
        return Uhat.astype(np.uint8)

def get_code(name: str) -> Code:
    n = name.lower()
    if n == "none": return CodeNone()
    if n == "repetition3": return CodeRepetition3()
    if n == "hamming74": return CodeHamming74()
    raise ValueError("Unknown code: " + name)
