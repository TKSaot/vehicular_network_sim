from __future__ import annotations
import numpy as np

def encode(bits: np.ndarray) -> np.ndarray:
    # TYPO FIX: -l changed to -1
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

# ==============================================================================
# NEW: Soft-decision decoding implementation
# ==============================================================================

# Pre-compute all 16 valid 4-bit data words and their corresponding 7-bit codewords
_DATAWORDS = np.array([[i>>3&1, i>>2&1, i>>1&1, i&1] for i in range(16)], dtype=np.uint8)
_CODEBOOK = encode(_DATAWORDS.flatten())
_CODEBOOK = _CODEBOOK.reshape(16, 7)
# BPSK representation (-1 for 0, +1 for 1) of the codebook for distance calculation
_CODEBOOK_BPSK = (2.0 * _CODEBOOK - 1.0).astype(np.float32)

def decode_soft(soft_values: np.ndarray) -> np.ndarray:
    """
    Performs soft-decision decoding of Hamming(7,4) coded data using
    Maximum Likelihood decoding. It finds the closest valid codeword in
    Euclidean distance for each received block.

    Args:
        soft_values: A 1D NumPy array of floats representing the soft values
                     (e.g., the real part of BPSK symbols). Length must be a
                     multiple of 7.

    Returns:
        A 1D NumPy array of decoded data bits (0s and 1s).
    """
    # Reshape the input soft values into blocks of 7
    num_blocks = len(soft_values) // 7
    if num_blocks == 0:
        return np.zeros(0, dtype=np.uint8)
    
    rx_blocks = soft_values[:num_blocks * 7].reshape(num_blocks, 7)

    # Use NumPy broadcasting to efficiently calculate the squared Euclidean distance
    # between each received block and every codeword in the BPSK codebook.
    # Shape of rx_blocks: (num_blocks, 1, 7)
    # Shape of _CODEBOOK_BPSK: (1, 16, 7)
    # Shape of diff: (num_blocks, 16, 7)
    diff = rx_blocks[:, np.newaxis, :] - _CODEBOOK_BPSK[np.newaxis, :, :]
    distances = np.sum(diff**2, axis=2) # Shape: (num_blocks, 16)

    # For each block, find the index of the codeword with the minimum distance
    best_codeword_indices = np.argmin(distances, axis=1) # Shape: (num_blocks,)

    # Use the indices to look up the corresponding 4-bit data words
    decoded_blocks = _DATAWORDS[best_codeword_indices] # Shape: (num_blocks, 4)

    # Flatten the decoded blocks back into a 1D bit stream
    return decoded_blocks.reshape(-1)