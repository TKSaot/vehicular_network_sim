
# multimodal_ofdm/hamming74.py
from __future__ import annotations
import numpy as np

def encode(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(b)) % 4
    if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    D = b.reshape(-1,4)
    d1,d2,d3,d4 = D[:,0],D[:,1],D[:,2],D[:,3]
    p1 = (d1 ^ d2 ^ d4).astype(np.uint8)
    p2 = (d1 ^ d3 ^ d4).astype(np.uint8)
    p3 = (d2 ^ d3 ^ d4).astype(np.uint8)
    C = np.stack([d1,d2,d3,d4,p1,p2,p3], axis=1)
    return C.reshape(-1)

def decode(bits: np.ndarray) -> np.ndarray:
    c = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = (len(c)//7)*7
    c = c[:L]
    if L == 0: return np.zeros(0, dtype=np.uint8)
    C = c.reshape(-1,7)
    d1,d2,d3,d4,p1,p2,p3 = [C[:,i] for i in range(7)]
    s1 = (d1 ^ d2 ^ d4 ^ p1).astype(np.uint8)
    s2 = (d1 ^ d3 ^ d4 ^ p2).astype(np.uint8)
    s3 = (d2 ^ d3 ^ d4 ^ p3).astype(np.uint8)
    synd = (s1 + (s2<<1) + (s3<<2)).astype(np.uint8)
    pos_map = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)  # matches your existing mapping
    err_pos = pos_map[synd]
    for i in range(C.shape[0]):
        p = int(err_pos[i])
        if p != 0:
            C[i, p-1] ^= 1
    return C[:,:4].reshape(-1)
