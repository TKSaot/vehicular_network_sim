# app_layer/application.py
"""
Application-layer serializers for modalities.

Segmentation is transmitted as an **ID map** with a **pre-shared palette**
stored inside this Python process (out-of-band). We guarantee:
- No out-of-palette IDs are introduced (IDs are always in 0..K-1).
- No white pixels appear as an artifact in the RGB output:
    * White boundary strokes in input PNGs are removed before palette building.
    * The palette is forced to contain no white-like colors.
    * If palette is missing, a deterministic "no-white" LUT is used.

Environment knobs (optional; sensible defaults if unset):
- SEG_STRIP_WHITE: "1" (default) to remove white boundary strokes from input RGB.
- SEG_WHITE_THRESH: int threshold in [0..255], default 250; RGB >= thresh on all channels is treated as white-like.
- SEG_ID_NOISE_P: float in [0,1], default 0.0; optional semantic ID substitution prob (uniform to a different valid ID).
- SEG_ID_NOISE_SEED: int seed for deterministic substitution (optional).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import os
import numpy as np
from PIL import Image

# ----------------- App Header -----------------
@dataclass
class AppHeader:
    version: int
    modality: Literal["text", "edge", "depth", "segmentation"]
    height: int = 0
    width: int = 0
    channels: int = 0
    bits_per_sample: int = 8
    payload_len_bytes: int = 0

    def to_bytes(self) -> bytes:
        mod_code = {"text":0, "edge":1, "depth":2, "segmentation":3}[self.modality]
        b = bytearray(16)
        b[0] = self.version & 0xFF
        b[1] = mod_code & 0xFF
        b[2:4] = int(self.height).to_bytes(2, 'big')
        b[4:6] = int(self.width).to_bytes(2, 'big')
        b[6] = self.channels & 0xFF
        b[7] = self.bits_per_sample & 0xFF
        b[8:12] = int(self.payload_len_bytes).to_bytes(4, 'big')
        # [12:16) zeros (reserved) to keep a stable 16-byte header
        return bytes(b)

    @staticmethod
    def from_bytes(b: bytes) -> 'AppHeader':
        if len(b) < 16:
            raise ValueError(f"AppHeader bytes too short: {len(b)} < 16")
        version = b[0]
        code = b[1]
        mapping = {0:"text",1:"edge",2:"depth",3:"segmentation"}
        if code not in mapping:
            raise ValueError(f"Invalid modality code: {code}")
        modality = mapping[code]
        height = int.from_bytes(b[2:4], 'big')
        width  = int.from_bytes(b[4:6], 'big')
        channels = b[6]
        bps = b[7]
        payload_len = int.from_bytes(b[8:12], 'big')
        return AppHeader(version=version, modality=modality, height=height, width=width,
                         channels=channels, bits_per_sample=bps, payload_len_bytes=payload_len)

# ----------------- Pre-shared palette (per run) -----------------
# Stored out-of-band in-process; not sent over the channel.
_SEG_PALETTE_CURRENT: np.ndarray | None = None  # shape (K,3) uint8

def _set_seg_palette(palette: np.ndarray) -> None:
    global _SEG_PALETTE_CURRENT
    _SEG_PALETTE_CURRENT = palette.astype(np.uint8, copy=False)

def _get_seg_palette() -> np.ndarray | None:
    return _SEG_PALETTE_CURRENT

# ----------------- Utilities -----------------
def load_text_as_bytes(path: str, encoding: str="utf-8") -> bytes:
    with open(path, "r", encoding=encoding) as f:
        txt = f.read()
    return txt.encode(encoding)

def text_bytes_to_string(b: bytes, encoding: str="utf-8", errors: str="replace") -> str:
    return b.decode(encoding, errors=errors)

def load_image_to_array(path: str, modality: str, validate_mode: bool = True) -> np.ndarray:
    """
    Load image for edge/depth/segmentation.
    - edge: single-channel binary (0/255) -> uint8 HxW
    - depth: grayscale 8-bit -> uint8 HxW
    - segmentation: RGB (converted to IDs later) -> uint8 HxWx3
    """
    im = Image.open(path)
    if modality == "edge":
        if validate_mode and im.mode not in ("L","1"):
            im = im.convert("L")
        arr = np.array(im)
        if arr.ndim == 3:
            arr = np.array(im.convert("L"))
        arr = (arr >= 128).astype(np.uint8) * 255
        return arr

    elif modality == "depth":
        if validate_mode and im.mode != "L":
            im = im.convert("L")
        arr = np.array(im)
        return arr.astype(np.uint8)

    elif modality == "segmentation":
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.array(im).astype(np.uint8)
        return arr

    else:
        raise ValueError("Unsupported modality for image")

def _reshape_bytes_safe(payload_bytes: bytes, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Robust reshape: truncate or zero-pad to the target element count."""
    n_expected = 1
    for s in shape: n_expected *= s
    elem = np.dtype(dtype).itemsize
    cnt = min(len(payload_bytes)//elem, n_expected)
    arr = np.frombuffer(payload_bytes, dtype=dtype, count=cnt)
    if arr.size < n_expected:
        arr = np.pad(arr, (0, n_expected - arr.size), mode="constant", constant_values=0)
    else:
        arr = arr[:n_expected]
    return arr.reshape(shape)

# ----------------- White handling & palette building -----------------
def _white_mask(rgb: np.ndarray, thresh: int) -> np.ndarray:
    """Return boolean mask where pixels are white-like: all channels >= thresh."""
    return (rgb[..., 0] >= thresh) & (rgb[..., 1] >= thresh) & (rgb[..., 2] >= thresh)

def _suppress_white_boundaries(rgb: np.ndarray, white_thresh: int = 250, iters: int = 2) -> np.ndarray:
    """
    Remove white-like boundary strokes by propagating neighboring non-white colors.
    - No SciPy; uses iterative 8-neighbor copying.
    - Guarantees: output contains no white-like pixels (per threshold), unless the
      entire image is white (in which case we map to a fixed dark color later).
    """
    H, W, _ = rgb.shape
    out = rgb.copy()
    mask = _white_mask(out, white_thresh)
    if not mask.any():
        return out

    for _ in range(max(1, iters)):
        if not mask.any():
            break
        # Build 8 shifted neighbor stacks
        neighs = []
        neigh_nonwhite = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                pad_y0, pad_y1 = (max(dy, 0), max(-dy, 0))
                pad_x0, pad_x1 = (max(dx, 0), max(-dx, 0))
                shifted = np.pad(out, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode="edge")
                shifted = shifted[pad_y1:pad_y1+H, pad_x1:pad_x1+W, :]
                neighs.append(shifted)
                neigh_nonwhite.append(~_white_mask(shifted, white_thresh))
        stack = np.stack(neighs, axis=0)                 # [8,H,W,3]
        stack_mask = np.stack(neigh_nonwhite, axis=0)    # [8,H,W]

        # For white pixels, copy the first available non-white neighbor
        remaining = mask.copy()
        for n in range(stack.shape[0]):
            m = stack_mask[n] & remaining
            if not m.any():
                continue
            out[m] = stack[n][m]
            remaining[m] = False
        mask = _white_mask(out, white_thresh)

    # If still white remains (isolated), set to a dark gray (non-white, non-black)
    mask = _white_mask(out, white_thresh)
    if mask.any():
        out[mask] = np.array([32, 32, 32], dtype=np.uint8)
    return out

def _build_seg_ids_and_palette(rgb_arr: np.ndarray, white_thresh: int) -> tuple[np.ndarray, np.ndarray, str]:
    """
    From cleaned HxWx3 RGB, produce:
      - id_map: HxW uint8/uint16 (IDs 0..K-1) with no white class.
      - palette: Kx3 uint8 (id -> RGB), guaranteed to contain no white-like colors.
      - enc: "id8" if K<=256 else "id16".
    """
    H, W, _ = rgb_arr.shape
    flat = rgb_arr.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)  # colors sorted lexicographically
    # Sanity: remove any residual white-like colors (should be none after suppression)
    white_like = (uniq[:, 0] >= white_thresh) & (uniq[:, 1] >= white_thresh) & (uniq[:, 2] >= white_thresh)
    if white_like.any():
        # Map white-like indices to the nearest non-white color index (by L2 in RGB)
        non_white_idx = np.where(~white_like)[0]
        if non_white_idx.size == 0:
            # Degenerate: whole image was white; force single non-white color
            palette = np.array([[32, 32, 32]], dtype=np.uint8)
            id_map = np.zeros((H, W), dtype=np.uint8)
            return id_map, palette, "id8"
        # Build mapping table
        idx_map = np.arange(uniq.shape[0])
        for wi in np.where(white_like)[0]:
            # nearest non-white by Euclidean distance in RGB
            d = np.sum((uniq[non_white_idx].astype(np.int32) - uniq[wi].astype(np.int32))**2, axis=1)
            nearest = non_white_idx[np.argmin(d)]
            idx_map[idx_map == wi] = nearest
        inv = idx_map[inv]
        uniq = uniq[~white_like]

    K = uniq.shape[0]
    if K <= 256:
        id_map = inv.astype(np.uint8).reshape(H, W)
        enc = "id8"
    else:
        id_map = inv.astype(np.uint16).reshape(H, W)
        enc = "id16"
    palette = uniq.astype(np.uint8)
    return id_map, palette, enc

def _safe_color_lut(K: int) -> np.ndarray:
    """
    Deterministic "no-white" LUT for fallback visualization.
    Uses HSV-like sampling; avoids (255,255,255) and very bright values.
    """
    if K <= 0:
        return np.zeros((1, 3), dtype=np.uint8)
    hues = np.linspace(0.0, 1.0, K, endpoint=False)
    sat = 0.85
    val = 0.72  # < 1.0 so we avoid white
    rgb = []
    for h in hues:
        # Simple HSV->RGB
        i = int(h * 6) % 6
        f = (h * 6) - int(h * 6)
        p = val * (1 - sat)
        q = val * (1 - f * sat)
        t = val * (1 - (1 - f) * sat)
        if i == 0: r, g, b = val, t, p
        elif i == 1: r, g, b = q, val, p
        elif i == 2: r, g, b = p, val, t
        elif i == 3: r, g, b = p, q, val
        elif i == 4: r, g, b = t, p, val
        else:        r, g, b = val, p, q
        rgb.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(rgb, dtype=np.uint8)

# ----------------- Semantic ID corruption -----------------
def _apply_id_noise_uniform(ids: np.ndarray, K: int, p: float, seed: int | None) -> np.ndarray:
    """
    Uniformly substitute an ID with a *different* valid ID with probability p.
    Never introduces out-of-palette values.
    """
    if p <= 0.0 or K <= 1:
        return ids
    rng = np.random.default_rng(seed if seed is not None else (0xC0FFEE ^ (K * 2654435761 & 0xFFFFFFFF)))
    mask = rng.random(ids.shape, dtype=np.float32) < float(p)
    if not mask.any():
        return ids
    out = ids.copy()
    orig = out[mask].astype(np.int64)
    draw = rng.integers(0, K - 1, size=orig.size, dtype=np.int64)  # 0..K-2
    new_ids = draw + (draw >= orig)  # skip original
    out_dtype = out.dtype
    if out_dtype == np.uint8:
        new_ids = new_ids.astype(np.uint8, copy=False)
    else:
        new_ids = new_ids.astype(np.uint16, copy=False)
    out[mask] = new_ids
    return out

# ----------------- Serialize content -----------------
def serialize_content(modality: str, content_path: str, text_encoding: str="utf-8",
                      validate_image_mode: bool=True) -> tuple[AppHeader, bytes]:
    """
    Returns (AppHeader, payload_bytes).
    - text: UTF-8 bytes
    - edge/depth: L8 bytes
    - segmentation: ID bytes (id8/id16) with pre-shared palette; white boundary strokes removed.
    """
    if modality == "text":
        data = load_text_as_bytes(content_path, encoding=text_encoding)
        hdr = AppHeader(version=1, modality="text", height=0, width=0, channels=0,
                        bits_per_sample=8, payload_len_bytes=len(data))
        return hdr, data

    if modality in ("edge","depth"):
        arr = load_image_to_array(content_path, modality=modality, validate_mode=validate_image_mode)
        h, w = arr.shape
        payload = arr.reshape(-1).tobytes()
        hdr = AppHeader(version=1, modality=modality, height=h, width=w, channels=1,
                        bits_per_sample=8, payload_len_bytes=len(payload))
        return hdr, payload

    if modality == "segmentation":
        # 0) Settings
        strip_white = (os.getenv("SEG_STRIP_WHITE", "1").strip() != "0")
        white_thresh = int(os.getenv("SEG_WHITE_THRESH", "250"))

        # 1) Load + (optionally) remove white boundary strokes
        rgb = load_image_to_array(content_path, modality="segmentation", validate_mode=True)
        if strip_white:
            rgb = _suppress_white_boundaries(rgb, white_thresh=white_thresh, iters=2)

        h, w, _ = rgb.shape

        # 2) Build IDs and palette (force no white-like colors)
        id_map, palette, enc = _build_seg_ids_and_palette(rgb, white_thresh=white_thresh)
        _set_seg_palette(palette)  # pre-shared in-process, not transmitted

        # 3) Optional semantic ID corruption (uniform to a different valid ID)
        p_env = float(os.getenv("SEG_ID_NOISE_P", "0.0") or "0.0")
        seed_env = os.getenv("SEG_ID_NOISE_SEED", "").strip()
        seed = int(seed_env) if seed_env != "" else None
        if p_env > 0.0:
            id_map = _apply_id_noise_uniform(id_map, K=int(palette.shape[0]), p=p_env, seed=seed)

        # 4) Payload = IDs (big-endian for 16-bit)
        if enc == "id8":
            id_bytes = id_map.reshape(-1).tobytes()
            bits = 8
        else:
            id_bytes = id_map.astype('>u2').reshape(-1).tobytes()
            bits = 16

        hdr = AppHeader(version=1, modality="segmentation", height=h, width=w, channels=1,
                        bits_per_sample=bits, payload_len_bytes=len(id_bytes))
        return hdr, id_bytes

    raise ValueError("Unknown modality")

# ----------------- Deserialize content -----------------
def deserialize_content(hdr: AppHeader, payload_bytes: bytes, text_encoding: str="utf-8",
                        text_errors: str="replace") -> tuple[str, np.ndarray]:
    """
    Convert bytes back to text or image.
    - segmentation: reconstruct ID map and colorize with the pre-shared palette;
      if missing, use a deterministic "no-white" LUT (never produces white).
    """
    if hdr.modality == "text":
        s = text_bytes_to_string(payload_bytes, encoding=text_encoding, errors=text_errors)
        return s, np.array([], dtype=np.uint8)

    if hdr.modality in ("edge","depth"):
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        if hdr.modality == "edge":
            arr = (arr >= 128).astype(np.uint8) * 255
        return "", arr

    if hdr.modality == "segmentation":
        # 1) Rebuild IDs
        if hdr.bits_per_sample <= 8:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        else:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.dtype('>u2')).astype(np.uint16, copy=False)

        # 2) Colorize with pre-shared palette (preferred), else safe LUT
        pal = _get_seg_palette()
        if pal is None or pal.size == 0:
            K = int(ids.max()) + 1 if ids.size > 0 else 1
            pal = _safe_color_lut(K)  # guaranteed no white-like entries

        K = int(pal.shape[0])
        ids_clamped = np.minimum(ids, K - 1)  # stay in-palette even under bit errors
        rgb = pal[ids_clamped]  # HxWx3
        return "", rgb.astype(np.uint8)

    raise ValueError("Unknown modality in header")

# ----------------- Save output -----------------
def save_output(hdr: AppHeader, text_str: str, img_arr: np.ndarray, out_path: str) -> None:
    """
    Save reconstructed content to disk:
    - text -> .txt UTF-8
    - images -> PNG (edge/depth as L, segmentation as RGB)
    """
    if hdr.modality == "text":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_str)
        return

    if hdr.modality in ("edge","depth"):
        im = Image.fromarray(img_arr.astype(np.uint8), mode="L")
        im.save(out_path, format="PNG")
        return

    if hdr.modality == "segmentation":
        im = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
        im.save(out_path, format="PNG")
        return

    raise ValueError("Unsupported modality")

# channel/channel_model.py
"""
Channel models: AWGN and Rayleigh fading with optional Doppler correlation.
We operate on complex baseband symbols.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def awgn_channel(symbols: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Es/N0 -> noise variance per complex dimension: N0/2 = Es / (2*SNR_linear)
    Es = np.mean(np.abs(symbols)**2) if len(symbols) else 1.0
    snr_lin = 10**(snr_db/10.0)
    N0 = Es / snr_lin
    noise_var = N0
    noise = (rng.normal(0, np.sqrt(noise_var/2), size=symbols.shape) +
             1j * rng.normal(0, np.sqrt(noise_var/2), size=symbols.shape))
    return symbols + noise

def rayleigh_fading(symbols: np.ndarray, snr_db: float, seed: int | None = None,
                    doppler_hz: float = 30.0, symbol_rate: float = 1e6, block_fading: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Rayleigh fading h[n] and AWGN. Returns (rx_symbols, h) where h is the complex fade per symbol.
    If block_fading=True, h is constant over the whole array.
    Otherwise, AR(1) process is used: h[n] = rho * h[n-1] + sqrt(1-rho^2) * v[n], v ~ CN(0,1)
    with rho approx J0(2*pi*fd*Ts) ~ exp(- (2*pi*fd*Ts)^2 / 2) for small arguments.
    """
    rng = np.random.default_rng(seed)
    N = len(symbols)
    if N == 0:
        return symbols.copy(), np.ones(0, dtype=np.complex128)

    Ts = 1.0 / max(1.0, symbol_rate)
    fd = abs(float(doppler_hz))
    # Approximate rho for small fd*Ts: rho ~ exp(-(2*pi*fd*Ts)^2/2). Bound it between 0 and 0.9999
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = min(0.9999, max(0.0, rho))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.ones(N, dtype=np.complex128) * h0
    else:
        v_real = rng.normal(0, 1.0, size=N)
        v_imag = rng.normal(0, 1.0, size=N)
        v = (v_real + 1j*v_imag) / np.sqrt(2.0)  # CN(0,1)
        h = np.zeros(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    faded = symbols * h

    # AWGN on top
    Es = np.mean(np.abs(symbols)**2) if len(symbols) else 1.0
    snr_lin = 10**(snr_db/10.0)
    N0 = Es / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=N) + 1j*rng.normal(0, np.sqrt(N0/2), size=N))

    return faded + noise, h

def equalize(rx_symbols: np.ndarray, tx_pilot: np.ndarray, rx_pilot: np.ndarray) -> tuple[np.ndarray, complex]:
    """
    One-tap equalizer using average pilot-based channel estimate h_hat = mean(rx_pilot / tx_pilot).
    Returns (equalized_symbols, h_hat).
    """
    # Avoid division by zero
    mask = np.abs(tx_pilot) > 1e-12
    if not np.any(mask):
        return rx_symbols, 1+0j
    h_hat = np.mean(rx_pilot[mask] / tx_pilot[mask])
    eq = rx_symbols / (h_hat + 1e-12)
    return eq, h_hat
# common/byte_mapping.py
from __future__ import annotations
import numpy as np

def _ensure_len(b: np.ndarray, N: int) -> np.ndarray:
    """Trim or zero-pad to exactly N bytes (uint8)."""
    if b.size < N:
        return np.pad(b, (0, N - b.size), mode="constant", constant_values=0)
    if b.size > N:
        return b[:N]
    return b

def _perm_indices(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) ^ 0xA5C3_19D3)
    return rng.permutation(N)

def map_bytes(data: bytes, mtu_bytes: int, scheme: str = "none", seed: int | None = None) -> bytes:
    """
    Apply a byte mapping across the whole payload BEFORE segmentation.
    - "none": identity
    - "permute": global permutation by RNG(seed)
    - "frame_block": treat payload as matrix [rows=ceil(N/mtu), cols=mtu] and read column-wise
    """
    if scheme == "none":
        return data
    arr = np.frombuffer(data, dtype=np.uint8)
    N = arr.size
    if N == 0:
        return data

    if scheme == "permute":
        if seed is None:
            raise ValueError("permute mapping requires a seed")
        perm = _perm_indices(N, seed)
        return arr[perm].tobytes()

    if scheme == "frame_block":
        M = (N + mtu_bytes - 1) // mtu_bytes  # number of DATA frames (no headers)
        order = []
        for c in range(mtu_bytes):
            for r in range(M):
                j = r * mtu_bytes + c
                if j < N:
                    order.append(j)
        order = np.asarray(order, dtype=np.int64)
        return arr[order].tobytes()

    raise ValueError(f"Unknown byte mapping scheme: {scheme}")

def unmap_bytes(data_rx: bytes, mtu_bytes: int, scheme: str = "none",
                seed: int | None = None, original_len: int | None = None) -> bytes:
    """
    Invert the mapping after reassembly. We assume drop_bad_frames=False so length is preserved.
    If lengths mismatch, we trim/pad to original_len (if provided).
    """
    arr = np.frombuffer(data_rx, dtype=np.uint8)
    N = arr.size if original_len is None else int(original_len)
    arr = _ensure_len(arr, N)

    if scheme == "none":
        return arr.tobytes()

    if scheme == "permute":
        if seed is None:
            raise ValueError("permute mapping requires a seed")
        perm = _perm_indices(N, seed)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(N, dtype=perm.dtype)
        return arr[inv].tobytes()

    if scheme == "frame_block":
        M = (N + mtu_bytes - 1) // mtu_bytes
        order = []
        for c in range(mtu_bytes):
            for r in range(M):
                j = r * mtu_bytes + c
                if j < N:
                    order.append(j)
        order = np.asarray(order, dtype=np.int64)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=order.dtype)
        return arr[inv].tobytes()

    raise ValueError(f"Unknown byte mapping scheme: {scheme}")

# common/config.py
"""
Configuration dataclasses for the vehicular network simulation.
These allow you to customize parameters across the stack.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class AppConfig:
    # modality: "text" | "edge" | "depth" | "segmentation"
    modality: Literal["text", "edge", "depth", "segmentation"] = "text"
    # For images: if True, enforce the expected channel layout per modality
    validate_image_mode: bool = True
    # For text decoding robustness: replace undecodable bytes
    text_encoding: str = "utf-8"
    text_errors: str = "replace"  # "strict", "ignore", or "replace"

@dataclass
class LinkConfig:
    mtu_bytes: int = 1024
    interleaver_depth: int = 8
    fec_scheme: Literal["none", "repeat", "hamming74", "rs255_223"] = "hamming74"
    repeat_k: int = 3
    drop_bad_frames: bool = False

    # Header robustness (existing)
    strong_header_protection: bool = True
    header_copies: int = 7
    header_rep_k: int = 5
    force_output_on_hdr_fail: bool = True
    verbose: bool = False

    # NEW: cross-frame mapping / randomized mapping
    byte_mapping_scheme: Literal["none", "permute", "frame_block"] = "none"
    byte_mapping_seed: Optional[int] = None  # if None, fall back to chan.seed

@dataclass
class ModulationConfig:
    scheme: Literal["bpsk", "qpsk", "16qam"] = "qpsk"

@dataclass
class ChannelConfig:
    channel_type: Literal["awgn", "rayleigh"] = "rayleigh"
    snr_db: float = 10.0                 # Es/N0 in dB
    seed: Optional[int] = 12345          # RNG seed for reproducibility
    # Rayleigh-specific (simple time-varying AR(1) fading)
    doppler_hz: float = 30.0             # nominal Doppler (affects fading correlation)
    symbol_rate: float = 1e6             # symbols per second (for Doppler correlation)
    block_fading: bool = False           # if True, h is constant over a frame

@dataclass
class PilotConfig:
    preamble_len: int = 32               # BPSK preamble bits per frame (not used for sync in this sim)
    pilot_len: int = 16                  # known pilot symbols per frame for channel estimation (QPSK pilots)
    pilot_every_n_symbols: int = 0       # not used in this simplified sim (set 0 to disable mid-frame pilots)

@dataclass
class SimulationConfig:
    app: AppConfig = field(default_factory=AppConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    mod: ModulationConfig = field(default_factory=ModulationConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)

# common/run_utils.py
from __future__ import annotations
import os, json, re, datetime
from dataclasses import asdict

def _sanitize(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z_.-]+', '-', str(s)).strip('-')

def make_output_dir(cfg, modality: str, input_path: str, output_root: str = "outputs") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ch = cfg.chan.channel_type
    doppler_tag = f"_fd{int(round(cfg.chan.doppler_hz))}" if ch == "rayleigh" else ""
    snr_tag = f"snr{int(round(cfg.chan.snr_db))}"
    fec = cfg.link.fec_scheme
    if fec == "repeat":
        fec = f"repeat{cfg.link.repeat_k}"
    hdr_tag = f"_hdr{cfg.link.header_copies}xR{cfg.link.header_rep_k}"
    # NEW
    map_tag = {"none":"mapNone", "permute":"mapPerm", "frame_block":"mapFB"}.get(cfg.link.byte_mapping_scheme, "map?")
    if cfg.link.byte_mapping_scheme != "none":
        seed_tag = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
        map_tag += f"{seed_tag}"

    folder = (
        f"{ts}__{modality}__{ch}{doppler_tag}_{snr_tag}"
        f"__{cfg.mod.scheme}__{fec}{hdr_tag}_{map_tag}"
        f"_ilv{cfg.link.interleaver_depth}"
        f"_mtu{cfg.link.mtu_bytes}"
        f"_seed{cfg.chan.seed}"
    )
    base = os.path.join(output_root, _sanitize(folder))
    os.makedirs(base, exist_ok=True)

    meta = asdict(cfg)
    meta.update({
        "modality": modality,
        "input_file": input_path,
        "output_dir": base,
    })
    with open(os.path.join(base, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return base

def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# common/utils.py
"""
General helpers: bit/byte conversions, CRC, interleaving, padding.
"""

from __future__ import annotations
import numpy as np
import binascii

def set_seed(seed: int | None):
    if seed is not None:
        np.random.seed(seed)

# ----------------- Bits/Bytes -----------------
def bytes_to_bits(b: bytes) -> np.ndarray:
    """Convert bytes to a 1-D np.uint8 array of 0/1 bits."""
    if isinstance(b, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(b, dtype=np.uint8)
    elif isinstance(b, np.ndarray) and b.dtype == np.uint8:
        arr = b
    else:
        arr = np.array(list(b), dtype=np.uint8)
    return np.unpackbits(arr)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a 1-D bits array (0/1) to bytes. Pads to a multiple of 8 with zeros."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

def pad_bits_to_multiple(bits: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    """Pad with zeros so len(bits) is a multiple of m. Return (padded_bits, pad_len)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(bits)) % m
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bits, pad

def remove_tail_bits(bits: np.ndarray, n_remove: int) -> np.ndarray:
    if n_remove <= 0:
        return bits
    return bits[:-n_remove] if n_remove <= len(bits) else np.array([], dtype=np.uint8)

# ----------------- CRC -----------------
def crc32_bytes(data: bytes) -> int:
    """CRC-32 for a bytes payload (unsigned)."""
    return binascii.crc32(data) & 0xFFFFFFFF

def append_crc32(payload: bytes) -> bytes:
    c = crc32_bytes(payload)
    return payload + c.to_bytes(4, 'big')

def verify_and_strip_crc32(data_with_crc: bytes) -> tuple[bool, bytes]:
    if len(data_with_crc) < 4:
        return False, b""
    payload, crc = data_with_crc[:-4], data_with_crc[-4:]
    ok = (crc32_bytes(payload) == int.from_bytes(crc, 'big'))
    return ok, payload

# ----------------- Interleaving -----------------
def block_interleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """Simple block interleaver: write row-wise, read column-wise."""
    depth = max(1, int(depth))
    if depth == 1:
        return np.asarray(bits, dtype=np.uint8)
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = len(bits)
    cols = int(np.ceil(L / depth))
    pad = (depth * cols) - L
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    mat = bits.reshape(depth, cols)
    out = mat.T.reshape(-1)
    return out

def block_deinterleave(bits: np.ndarray, depth: int, original_len: int | None = None) -> np.ndarray:
    """Inverse of block_interleave. Optionally trim to original_len."""
    depth = max(1, int(depth))
    if depth == 1:
        out = np.asarray(bits, dtype=np.uint8).reshape(-1)
        return out[:original_len] if original_len is not None else out
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = len(bits)
    cols = int(np.ceil(L / depth))
    mat = bits.reshape(cols, depth).T
    out = mat.reshape(-1)
    if original_len is not None:
        out = out[:original_len]
    return out

# ----------------- PSI Metrics -----------------
def psnr(a: np.ndarray, b: np.ndarray, data_range: int = 255) -> float:
    """Peak Signal-to-Noise Ratio between two images (uint8 arrays)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

# configs/image_config.py
from __future__ import annotations
from typing import Literal
from common.config import (
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig
)

INPUTS = {
    "edge": "examples/edge_00001_.png",
    "depth": "examples/depth_00001_.png",
    "segmentation": "examples/segmentation_00001_.png",
}
OUTPUT_ROOT = "outputs"
DEFAULT_MODALITY: Literal["edge","depth","segmentation"] = "depth"

def build_config(modality: Literal["edge","depth","segmentation"] = DEFAULT_MODALITY) -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(modality=modality, validate_image_mode=True),
        link=LinkConfig(
            mtu_bytes=1024,
            interleaver_depth=16,
            fec_scheme="hamming74",
            repeat_k=3,
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            # NEW:
            byte_mapping_scheme="permute",   # try "permute" to fully randomize
            byte_mapping_seed=None,              # None -> use chan.seed
        ),
        mod=ModulationConfig(scheme="qpsk"),
        chan=ChannelConfig(
            channel_type="rayleigh",  # vehicular 既定
            snr_db=10.0,
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=16),
    )

# configs/text_config.py
from __future__ import annotations
from common.config import (
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig
)

INPUT = "examples/sample.txt"
OUTPUT_ROOT = "outputs"

def build_config() -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(modality="text", validate_image_mode=False),
        link=LinkConfig(
            mtu_bytes=1024,
            interleaver_depth=16,
            fec_scheme="hamming74",
            repeat_k=3,
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            # NEW (keep off for text):
            byte_mapping_scheme="none",
            byte_mapping_seed=None,
        ),
        mod=ModulationConfig(scheme="qpsk"),
        chan=ChannelConfig(
            channel_type="awgn",         # 初回安定実行
            snr_db=12.0,
            seed=12345,
            doppler_hz=50.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=16),
    )

# data_link_layer/encoding.py
"""
Link-layer framing: segmentation, headers, CRC, interleaving, and FEC application.

Frame types:
- 0x01: APP_HEADER  (this may be repeated 'header_copies' times up front)
- 0x02: DATA

Key robustness:
- Data reassembly ignores header.payload_len (which may be corrupted) and instead
  treats the whole tail (after 8B link header) as [payload || CRC32].
- APP header: multiple copies + FEC + Repetition(k), with bitwise-majority
  fallback to reconstruct a CRC-valid header at very low SNR.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from common.utils import bytes_to_bits, bits_to_bytes, append_crc32, verify_and_strip_crc32,\
                         block_interleave, block_deinterleave
from data_link_layer.error_correction import make_fec, RepetitionFEC

FRAME_TYPE_APP_HDR = 0x01
FRAME_TYPE_DATA    = 0x02

HEADER_LEN = 8  # bytes

@dataclass
class LinkHeader:
    version: int = 1
    frame_type: int = FRAME_TYPE_DATA
    seq_no: int = 0
    total_frames: int = 1
    payload_len: int = 0  # in bytes, before CRC

    def to_bytes(self) -> bytes:
        return bytes([self.version & 0xFF, self.frame_type & 0xFF]) + \
               int(self.seq_no).to_bytes(2, 'big') + \
               int(self.total_frames).to_bytes(2, 'big') + \
               int(self.payload_len).to_bytes(2, 'big')

    @staticmethod
    def from_bytes(b: bytes) -> 'LinkHeader':
        if len(b) < HEADER_LEN:
            raise ValueError("Header too short")
        ver = b[0]
        ftype = b[1]
        seq = int.from_bytes(b[2:4], 'big')
        tot = int.from_bytes(b[4:6], 'big')
        plen = int.from_bytes(b[6:8], 'big')
        return LinkHeader(version=ver, frame_type=ftype, seq_no=seq, total_frames=tot, payload_len=plen)

@dataclass
class LinkFrame:
    header: LinkHeader
    payload_with_crc: bytes  # payload || CRC32

    def to_bytes(self) -> bytes:
        return self.header.to_bytes() + self.payload_with_crc

    @staticmethod
    def from_bytes_unsafe(b: bytes) -> 'LinkFrame':
        """
        Parse using header.payload_len (unsafe if header corrupted).
        """
        hdr = LinkHeader.from_bytes(b[:HEADER_LEN])
        payload_with_crc = b[HEADER_LEN:HEADER_LEN + hdr.payload_len + 4]
        return LinkFrame(header=hdr, payload_with_crc=payload_with_crc)

    @staticmethod
    def from_bytes_safe(b: bytes) -> 'LinkFrame':
        """
        Safe parse that DOES NOT trust header.payload_len.
        It takes the entire tail after 8B header as [payload || CRC32].
        """
        hdr = LinkHeader.from_bytes(b[:HEADER_LEN])
        payload_with_crc = b[HEADER_LEN:]
        return LinkFrame(header=hdr, payload_with_crc=payload_with_crc)

# ------------- Packetization -------------
def segment_message(app_header: bytes, data: bytes, mtu_bytes: int, header_copies: int = 1) -> List[LinkFrame]:
    """
    First 'header_copies' frames carry the APP_HEADER (frame_type=APP_HDR).
    Following frames carry data. Each frame carries CRC over its payload (not including link header).
    """
    frames: List[LinkFrame] = []

    # Header frames
    for c in range(max(1, int(header_copies))):
        hdr0 = LinkHeader(frame_type=FRAME_TYPE_APP_HDR, seq_no=c, total_frames=0, payload_len=len(app_header))
        frames.append(LinkFrame(header=hdr0, payload_with_crc=append_crc32(app_header)))

    # Data frames
    total_data_frames = (len(data) + mtu_bytes - 1) // mtu_bytes
    for i in range(total_data_frames):
        chunk = data[i*mtu_bytes:(i+1)*mtu_bytes]
        # seq_no starts after header copies
        hdr = LinkHeader(frame_type=FRAME_TYPE_DATA, seq_no=header_copies + i, total_frames=0, payload_len=len(chunk))
        frames.append(LinkFrame(header=hdr, payload_with_crc=append_crc32(chunk)))

    # total_frames (for completeness; not relied on in RX)
    total_frames = header_copies + total_data_frames
    for fr in frames[:header_copies]:
        fr.header.total_frames = total_frames
    for fr in frames[header_copies:]:
        fr.header.total_frames = total_frames
    return frames

def apply_fec_and_interleave(frames: List[LinkFrame], fec_scheme: str, repeat_k: int,
                             interleaver_depth: int, strong_header: bool,
                             header_copies: int, header_rep_k: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert frames to bit arrays, apply FEC (+ header repetition if enabled) and interleaving per frame.
    Returns (list_of_bits, list_of_original_bit_lengths).
    """
    encoded_frames: List[np.ndarray] = []
    original_bit_lengths: List[int] = []

    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1, int(header_rep_k))) if strong_header else None

    for idx, fr in enumerate(frames):
        b = fr.to_bytes()
        bits = bytes_to_bits(b)
        original_bit_lengths.append(len(bits))

        enc = fec.encode(bits)
        if strong_header and header_rep is not None and idx < header_copies:
            enc = header_rep.encode(enc)

        inter = block_interleave(enc, interleaver_depth)
        encoded_frames.append(inter)

    return encoded_frames, original_bit_lengths

def reverse_fec_and_deinterleave(encoded_frames: List[np.ndarray], original_bit_lengths: List[int],
                                 fec_scheme: str, repeat_k: int, interleaver_depth: int,
                                 strong_header: bool, header_copies: int, header_rep_k: int) -> List[bytes]:
    """
    Reverse interleaving and FEC to recover raw frame bytes (without CRC verification).
    Header frames (first header_copies) are decoded with Repetition first.
    Returns list of frame bytes (may be corrupted).
    """
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1, int(header_rep_k))) if strong_header else None

    out_bytes: List[bytes] = []
    for idx, enc in enumerate(encoded_frames):
        deinter = block_deinterleave(enc, interleaver_depth)
        dec = deinter
        if strong_header and header_rep is not None and idx < header_copies:
            dec = header_rep.decode(dec)
        dec = fec.decode(dec)

        Lbits = original_bit_lengths[idx]
        if len(dec) > Lbits:
            dec = dec[:Lbits]
        b = bits_to_bytes(dec)
        out_bytes.append(b[: (Lbits + 7)//8])
    return out_bytes

def _majority_bytes(blobs: List[bytes]) -> bytes:
    """
    Bitwise majority vote across same-length byte sequences.
    Returns the majority-voted bytes (length = min length of inputs).
    """
    if not blobs:
        return b""
    min_len = min(len(x) for x in blobs)
    if min_len == 0:
        return b""
    arr_bits = [bytes_to_bits(x[:min_len]) for x in blobs]
    M = np.stack(arr_bits, axis=0)
    s = np.sum(M, axis=0)
    out_bits = (s >= (M.shape[0]//2 + 1)).astype(np.uint8)
    return bits_to_bytes(out_bits)[:min_len]

def reassemble_and_check(frames_bytes: List[bytes], header_copies: int = 1,
                         drop_bad: bool = False, verbose: bool = False) -> Tuple[bytes, bytes, dict]:
    """
    Parse frames bytes, verify CRC per frame (payload CRC only).
    - First 'header_copies' frames are APP_HEADER. We try CRC individually; if all fail,
      we do bitwise-majority across their payload||CRC and verify again.
    - Data frames: ignore header.payload_len and slice tail safely as [payload||CRC32].

    Returns (app_header_bytes, data_bytes, stats_dict)
    stats:
      {
        "all_crc_ok": bool,
        "app_header_crc_ok": bool,
        "app_header_recovered_via_majority": bool,
        "n_bad_frames": int,  # DATAのみカウント（ヘッダ失敗は app_header_crc_ok に反映）
        "n_frames": int
      }
    """
    n_frames = len(frames_bytes)
    H = max(1, int(header_copies))
    H = min(H, n_frames)

    # --- APP HEADER(s) ---
    hdr_ok = False
    hdr_majority_used = False
    hdr_payload = b""

    header_payload_crc_blobs: List[bytes] = []
    for i in range(H):
        b = frames_bytes[i]
        # Safe: use full tail as payload||CRC
        payload_crc = b[HEADER_LEN:]
        header_payload_crc_blobs.append(payload_crc)
        ok, payload = verify_and_strip_crc32(payload_crc)
        if ok and not hdr_ok:
            hdr_ok = True
            hdr_payload = payload  # 16B

    if not hdr_ok:
        # majority across header copies
        voted = _majority_bytes(header_payload_crc_blobs)
        ok, payload = verify_and_strip_crc32(voted)
        if ok:
            hdr_ok = True
            hdr_majority_used = True
            hdr_payload = payload

    # --- DATA frames ---
    data = bytearray()
    n_bad = 0
    for i in range(H, n_frames):
        b = frames_bytes[i]
        # Safe tail slicing
        payload_crc = b[HEADER_LEN:]
        ok, payload = verify_and_strip_crc32(payload_crc)
        if not ok:
            n_bad += 1
            if drop_bad:
                payload = b""
            if verbose:
                print(f"[WARN] DATA frame {i-H}/{n_frames-H} CRC failed")
        data.extend(payload)

    stats = {
        "all_crc_ok": bool(hdr_ok and (n_bad == 0)),
        "app_header_crc_ok": bool(hdr_ok),
        "app_header_recovered_via_majority": bool(hdr_majority_used),
        "n_bad_frames": int(n_bad),
        "n_frames": int(n_frames),
    }
    return hdr_payload, bytes(data), stats

# data_link_layer/error_correction.py
"""
Error correction schemes:
- None
- Repetition (k)
- Hamming(7,4)
- Reed-Solomon(255,223)  [optional, requires 'reedsolo' package]

All operate on bit arrays (0/1) and return bit arrays.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# ----------------- Base -----------------
class FECBase:
    name = "base"
    code_rate = 1.0
    def encode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.uint8)
    def decode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.uint8)

# ----------------- None -----------------
class NoFEC(FECBase):
    name = "none"
    code_rate = 1.0

# ----------------- Repetition -----------------
class RepetitionFEC(FECBase):
    def __init__(self, k: int = 3):
        assert k >= 1 and isinstance(k, int)
        self.k = k
        self.name = f"repeat{self.k}"
        self.code_rate = 1.0 / k
    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1, 1)
        return np.repeat(bits, self.k, axis=1).reshape(-1)
    def decode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) % self.k != 0:
            bits = bits[: (len(bits) // self.k) * self.k]
        grp = bits.reshape(-1, self.k)
        s = np.sum(grp, axis=1)
        return (s >= (self.k // 2 + 1)).astype(np.uint8)

# ----------------- Hamming(7,4) -----------------
class Hamming74FEC(FECBase):
    """
    Systematic Hamming(7,4) with codeword [d1 d2 d3 d4 p1 p2 p3]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4

    Parity-check H columns (for positions 1..7):
      pos1(d1): [1,1,0]
      pos2(d2): [1,0,1]
      pos3(d3): [0,1,1]
      pos4(d4): [1,1,1]
      pos5(p1): [1,0,0]
      pos6(p2): [0,1,0]
      pos7(p3): [0,0,1]
    """
    name = "hamming74"
    code_rate = 4/7

    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        pad = (-len(b)) % 4
        if pad:
            b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
        if len(b) == 0:
            return b
        D = b.reshape(-1, 4)
        d1, d2, d3, d4 = D[:,0], D[:,1], D[:,2], D[:,3]
        p1 = (d1 ^ d2 ^ d4).astype(np.uint8)
        p2 = (d1 ^ d3 ^ d4).astype(np.uint8)
        p3 = (d2 ^ d3 ^ d4).astype(np.uint8)
        C = np.stack([d1, d2, d3, d4, p1, p2, p3], axis=1)
        return C.reshape(-1)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        c = np.asarray(bits, dtype=np.uint8).reshape(-1)
        L = (len(c) // 7) * 7
        c = c[:L]
        if L == 0:
            return np.zeros(0, dtype=np.uint8)
        C = c.reshape(-1, 7)
        d1, d2, d3, d4, p1, p2, p3 = [C[:,i] for i in range(7)]
        s1 = (d1 ^ d2 ^ d4 ^ p1).astype(np.uint8)
        s2 = (d1 ^ d3 ^ d4 ^ p2).astype(np.uint8)
        s3 = (d2 ^ d3 ^ d4 ^ p3).astype(np.uint8)

        # syndrome value (s1 + 2*s2 + 4*s3) → error position
        # mapping derived from H columns shown in class docstring
        synd_val = (s1 + (s2 << 1) + (s3 << 2)).astype(np.uint8)
        # map: 0->0(no flip), 1->5, 2->6, 3->1, 4->7, 5->2, 6->3, 7->4
        map_arr = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)
        err_pos = map_arr[synd_val]  # 0..7

        # flip indicated bit (1-based position)
        for i in range(C.shape[0]):
            pos = int(err_pos[i])
            if pos != 0:
                C[i, pos-1] ^= 1

        # extract data bits
        data = C[:, :4].reshape(-1)
        return data

# ----------------- Reed-Solomon(255,223) optional -----------------
class RS255223FEC(FECBase):
    """
    Byte-oriented RS(255,223) over GF(256). Requires 'reedsolo' package.
    Encodes/decodes bytes, converts to/from bits at the boundary.
    Code rate 223/255 ~ 0.8745. Corrects up to 16 byte errors per 255-byte block.
    """
    name = "rs255_223"
    code_rate = 223/255

    def __init__(self):
        try:
            import reedsolo  # type: ignore
        except Exception as e:
            raise ImportError("reedsolo package is required for RS255_223 FEC. Install via: pip install reedsolo") from e
        self.rs = reedsolo.RSCodec(32)  # n-k = 32 parity bytes

    def encode(self, bits: np.ndarray) -> np.ndarray:
        from common.utils import bits_to_bytes, bytes_to_bits
        data_bytes = bits_to_bytes(bits)
        if len(data_bytes) % 223 != 0:
            pad = 223 - (len(data_bytes) % 223)
            data_bytes += bytes([0])*pad
        out = bytearray()
        for i in range(0, len(data_bytes), 223):
            out.extend(self.rs.encode(data_bytes[i:i+223]))
        return bytes_to_bits(bytes(out))

    def decode(self, bits: np.ndarray) -> np.ndarray:
        from common.utils import bits_to_bytes, bytes_to_bits
        data_bytes = bits_to_bytes(bits)
        L = (len(data_bytes) // 255) * 255
        data_bytes = data_bytes[:L]
        out = bytearray()
        for i in range(0, len(data_bytes), 255):
            block = data_bytes[i:i+255]
            try:
                dec = self.rs.decode(block)  # -> 223 bytes
            except Exception:
                dec = bytes([0])*223
            out.extend(dec)
        return bytes_to_bits(bytes(out))

def make_fec(scheme: str, repeat_k: int = 3) -> FECBase:
    s = scheme.lower()
    if s == "none":
        return NoFEC()
    if s == "repeat":
        return RepetitionFEC(k=repeat_k)
    if s == "hamming74":
        return Hamming74FEC()
    if s == "rs255_223":
        try:
            return RS255223FEC()
        except ImportError:
            print("[WARN] RS255_223 selected but 'reedsolo' not installed. Falling back to NoFEC.")
            return NoFEC()
    raise ValueError(f"Unknown FEC scheme: {scheme}")

# examples/simulate_image_transmission.py
"""
設定は configs/image_config.py に集約。
最小コマンド例:
  python examples/simulate_image_transmission.py --modality depth
  python examples/simulate_image_transmission.py --modality edge
  python examples/simulate_image_transmission.py --modality segmentation
必要なら --snr_db 等だけ上書き。
"""

import os, sys, argparse, numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.utils import set_seed, psnr
from common.run_utils import make_output_dir, write_json
from common.config import SimulationConfig
from common.byte_mapping import unmap_bytes
from app_layer.application import serialize_content, AppHeader, deserialize_content, save_output
from transmitter.send import build_transmission
from channel.channel_model import awgn_channel, rayleigh_fading
from receiver.receive import recover_from_symbols

from configs import image_config as CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", type=str, choices=["edge","depth","segmentation"], default=None)
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default=None)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--output_root", type=str, default=None)
    args = ap.parse_args()

    modality = args.modality if args.modality is not None else CFG.DEFAULT_MODALITY
    cfg: SimulationConfig = CFG.build_config(modality=modality)

    input_path = args.input if args.input is not None else CFG.INPUTS[modality]
    output_root = args.output_root if args.output_root is not None else CFG.OUTPUT_ROOT

    if args.snr_db is not None:
        cfg.chan.snr_db = float(args.snr_db)
    if args.channel is not None:
        cfg.chan.channel_type = args.channel

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input PNG not found: {input_path}")

    set_seed(cfg.chan.seed)

    # TX (keep tx_hdr for last-resort)
    tx_hdr, payload = serialize_content(modality, input_path, text_encoding="utf-8", validate_image_mode=True)
    app_hdr_bytes = tx_hdr.to_bytes()

    tx_syms, tx_meta = build_transmission(app_hdr_bytes, payload, cfg)

    if cfg.chan.channel_type == "awgn":
        rx_syms = awgn_channel(tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading
        )

    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    # Header selection
    hdr_used_mode = "valid"
    if not stats.get("app_header_crc_ok", False):
        if stats.get("app_header_recovered_via_majority", False):
            hdr_used_mode = "majority"
        elif cfg.link.force_output_on_hdr_fail:
            rx_app_hdr_b = tx_hdr.to_bytes()
            hdr_used_mode = "forced"
        else:
            hdr_used_mode = "invalid"
    # Parse header now (may be forced/majority)
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = tx_hdr
        hdr_used_mode = "forced-parse-failed"

    # NEW: invert byte mapping using hdr.payload_len_bytes (original length)
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes
    )

    # Robust image decoding (safe reshape inside, as before)
    _, img_arr = deserialize_content(rx_hdr, rx_payload_b, text_encoding="utf-8")

    out_dir = make_output_dir(cfg, modality=modality, input_path=input_path, output_root=output_root)
    out_png = os.path.join(out_dir, "received.png")
    save_output(rx_hdr, "", img_arr, out_png)

    # PSNR (with coercion to same shape already ensured)
    im_orig = Image.open(input_path)
    if modality in ("edge","depth"): im_orig = im_orig.convert("L")
    elif modality == "segmentation": im_orig = im_orig.convert("RGB")
    arr_orig = np.array(im_orig)
    if arr_orig.shape != img_arr.shape:
        arr_orig = arr_orig[:img_arr.shape[0], :img_arr.shape[1], ...]
    val_psnr = psnr(arr_orig.astype(np.uint8), img_arr.astype(np.uint8), data_range=255)

    report = {
        "modality": modality,
        "frames": int(stats["n_frames"]),
        "bad_frames": int(stats["n_bad_frames"]),
        "all_crc_ok": bool(stats["all_crc_ok"]),
        "app_header_crc_ok": bool(stats["app_header_crc_ok"]),
        "app_header_recovered_via_majority": bool(stats.get("app_header_recovered_via_majority", False)),
        "app_header_used_mode": hdr_used_mode,
        "psnr_db": float(val_psnr),
        "output_png": out_png,
    }
    write_json(os.path.join(out_dir, "rx_stats.json"), report)

    print("=== IMAGE Transmission Report ===")
    print(f"Output dir: {out_dir}")
    print(f"Modality: {modality}")
    print(f"SNR(dB): {cfg.chan.snr_db}, Channel: {cfg.chan.channel_type}, Mod: {cfg.mod.scheme}, FEC: {cfg.link.fec_scheme}")
    print(f"Frames: {report['frames']}, Bad: {report['bad_frames']}, All CRC OK: {report['all_crc_ok']}")
    print(f"Header mode: {hdr_used_mode}")
    print(f"PSNR(dB): {report['psnr_db']:.2f}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()


# examples/simulate_text_transmission.py
"""
設定は configs/text_config.py に集約。
最小コマンド:  python examples/simulate_text_transmission.py
必要なら --snr_db 等だけ上書き。
"""

import os, sys, argparse, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.utils import set_seed
from common.run_utils import make_output_dir, write_json
from common.config import SimulationConfig
from common.byte_mapping import unmap_bytes
from app_layer.application import serialize_content, AppHeader, deserialize_content, save_output
from transmitter.send import build_transmission
from channel.channel_model import awgn_channel, rayleigh_fading
from receiver.receive import recover_from_symbols

from configs import text_config as CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default=None)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--output_root", type=str, default=None)
    args = ap.parse_args()

    cfg: SimulationConfig = CFG.build_config()
    input_path = args.input if args.input is not None else CFG.INPUT
    output_root = args.output_root if args.output_root is not None else CFG.OUTPUT_ROOT

    if args.snr_db is not None:
        cfg.chan.snr_db = float(args.snr_db)
    if args.channel is not None:
        cfg.chan.channel_type = args.channel

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input text not found: {input_path}")

    set_seed(cfg.chan.seed)

    # TX-side (we keep hdr for last-resort fallback)
    tx_hdr, payload = serialize_content("text", input_path, text_encoding="utf-8", validate_image_mode=False)
    app_hdr_bytes = tx_hdr.to_bytes()

    tx_syms, tx_meta = build_transmission(app_hdr_bytes, payload, cfg)

    if cfg.chan.channel_type == "awgn":
        rx_syms = awgn_channel(tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading
        )

    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    # Choose header: valid → use; majority → use; else → optional force (for outputs at very low SNR)
    hdr_used_mode = "valid"
    if not stats.get("app_header_crc_ok", False):
        if stats.get("app_header_recovered_via_majority", False):
            hdr_used_mode = "majority"
        elif cfg.link.force_output_on_hdr_fail:
            rx_app_hdr_b = tx_hdr.to_bytes()  # forced oracle header to ensure output
            hdr_used_mode = "forced"
        else:
            # As a last resort, still try to parse (may throw)
            hdr_used_mode = "invalid"
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = tx_hdr
        hdr_used_mode = "forced-parse-failed"

    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes
    )

    text_str, _ = deserialize_content(rx_hdr, rx_payload_b, text_encoding="utf-8", text_errors="replace")

    out_dir = make_output_dir(cfg, modality="text", input_path=input_path, output_root=output_root)
    out_txt = os.path.join(out_dir, "received_text.txt")
    save_output(rx_hdr, text_str, np.array([]), out_txt)

    orig_text = open(input_path, "r", encoding="utf-8").read()
    recv_text = text_str
    minlen = min(len(orig_text), len(recv_text))
    mismatches = sum(1 for i in range(minlen) if orig_text[i] != recv_text[i]) + abs(len(orig_text)-len(recv_text))
    cer = mismatches / max(1, len(orig_text))

    report = {
        "frames": int(stats["n_frames"]),
        "bad_frames": int(stats["n_bad_frames"]),
        "all_crc_ok": bool(stats["all_crc_ok"]),
        "app_header_crc_ok": bool(stats["app_header_crc_ok"]),
        "app_header_recovered_via_majority": bool(stats.get("app_header_recovered_via_majority", False)),
        "app_header_used_mode": hdr_used_mode,
        "cer_approx": float(cer),
        "output_text": out_txt,
    }
    write_json(os.path.join(out_dir, "rx_stats.json"), report)

    print("=== TEXT Transmission Report ===")
    print(f"Output dir: {out_dir}")
    print(f"SNR(dB): {cfg.chan.snr_db}, Channel: {cfg.chan.channel_type}, Mod: {cfg.mod.scheme}, FEC: {cfg.link.fec_scheme}")
    print(f"Frames: {report['frames']}, Bad: {report['bad_frames']}, All CRC OK: {report['all_crc_ok']}")
    print(f"Header mode: {hdr_used_mode}")
    print(f"Approx. CER: {report['cer_approx']:.4f}")
    print(f"Saved: {out_txt}")

if __name__ == "__main__":
    main()


# physical_layer/modulation.py
"""
Mapping bits to complex baseband symbols (BPSK, QPSK, 16QAM) and back (hard decisions).
Also builds per-frame symbol sequences by adding a preamble and pilots.

⚠️ 重要: 整数 (uint8) 演算のまま I/Q を計算するとアンダーフローします。
必ず float にキャストしてから 1 - 2*b などの演算を行ってください。
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# -------------------------
# BPSK
# -------------------------
def _bpsk_mod(bits: np.ndarray) -> np.ndarray:
    # 0 -> -1, 1 -> +1 (real)
    return 2.0 * bits.astype(np.float64) - 1.0

def _bpsk_demod(symbols: np.ndarray) -> np.ndarray:
    return (symbols.real >= 0).astype(np.uint8)

# -------------------------
# QPSK (Gray)
# ビット [b0, b1] -> (I, Q) = (1-2*b0, 1-2*b1) / sqrt(2)
# -------------------------
def _qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(b) % 2 != 0:
        b = np.concatenate([b, np.zeros(1, dtype=np.uint8)])
    # ⚠️ float化してから演算（uint8のままだと 1-2*b でアンダーフロー）
    pairs = b.reshape(-1, 2).astype(np.float64)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]
    syms = (i + 1j * q) / np.sqrt(2.0)
    return syms

def _qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    i = (symbols.real < 0).astype(np.uint8)
    q = (symbols.imag < 0).astype(np.uint8)
    out = np.vstack([i, q]).T.reshape(-1)
    return out

# -------------------------
# 16QAM (Gray)
# 2ビット -> レベル in {-3, -1, +1, +3}
# (b1,b0) -> amp = (1 - 2*b1) * (3 - 2*b0)
# -------------------------
def _16qam_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(b)) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    quads = b.reshape(-1, 4).astype(np.float64)  # float化
    # I: (b1,b0) = (q[:,0], q[:,1]), Q: (q[:,2], q[:,3])
    I = (1.0 - 2.0 * quads[:, 0]) * (3.0 - 2.0 * quads[:, 1])
    Q = (1.0 - 2.0 * quads[:, 2]) * (3.0 - 2.0 * quads[:, 3])
    syms = (I + 1j * Q) / np.sqrt(10.0)
    return syms

def _16qam_demod(symbols: np.ndarray) -> np.ndarray:
    x = symbols * np.sqrt(10.0)
    I = x.real
    Q = x.imag
    # しきい値: -2, 0, +2 （レベル: -3, -1, +1, +3）
    def level_to_bits(vals):
        b1 = (vals < 0).astype(np.uint8)            # MSB
        b0 = (np.abs(vals) < 2).astype(np.uint8)    # 内側:1, 外側:0
        return b1, b0
    i1, i0 = level_to_bits(I)
    q1, q0 = level_to_bits(Q)
    bits = np.vstack([i1, i0, q1, q0]).T.reshape(-1)
    return bits.astype(np.uint8)

# -------------------------
# Modulator wrapper
# -------------------------
class Modulator:
    def __init__(self, scheme: str = "qpsk"):
        s = scheme.lower()
        if s not in ("bpsk", "qpsk", "16qam"):
            raise ValueError("Unsupported modulation")
        self.scheme = s

    @property
    def bits_per_symbol(self) -> int:
        return {"bpsk": 1, "qpsk": 2, "16qam": 4}[self.scheme]

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if self.scheme == "bpsk":
            return _bpsk_mod(bits)
        if self.scheme == "qpsk":
            return _qpsk_mod(bits)
        if self.scheme == "16qam":
            return _16qam_mod(bits)
        raise RuntimeError

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        if self.scheme == "bpsk":
            return _bpsk_demod(symbols)
        if self.scheme == "qpsk":
            return _qpsk_demod(symbols)
        if self.scheme == "16qam":
            return _16qam_demod(symbols)
        raise RuntimeError

# -------------------------
# PHY frame builder (preamble + pilots + data)
# -------------------------
def build_phy_frame(bits: np.ndarray, mod: Modulator,
                    preamble_len_bits: int = 32,
                    pilot_len_symbols: int = 16
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce a sequence of symbols for one link-layer frame:
    - preamble (BPSK on 1010...) for visibility (not used for sync here)
    - pilot (QPSK constant (1+j)/√2) for 1-tap channel estimation
    - data symbols (per selected modulation)
    Returns (symbols_concat, tx_pilot_symbols, data_symbols_only)
    """
    # Preamble: alternating 1010... (BPSK, real)
    pre_bits = np.tile(np.array([1, 0], dtype=np.uint8), preamble_len_bits // 2)
    pre_sym = _bpsk_mod(pre_bits).astype(np.complex128)  # 実軸 -> 複素

    # Pilots: constant QPSK @45°
    pilot = np.ones(pilot_len_symbols, dtype=np.complex128) * (1.0 + 1j) / np.sqrt(2.0)

    # Data
    data_syms = mod.modulate(bits).astype(np.complex128)

    all_syms = np.concatenate([pre_sym, pilot, data_syms])
    return all_syms, pilot, data_syms


# receiver/receive.py
"""
Receiver pipeline: channel equalization -> demod -> deinterleave -> FEC decode -> CRC check -> reassemble -> application decode.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from common.config import SimulationConfig
from data_link_layer.encoding import reverse_fec_and_deinterleave, reassemble_and_check
from physical_layer.modulation import Modulator
from channel.channel_model import equalize

def recover_from_symbols(rx_symbols: np.ndarray, tx_meta: Dict, cfg: SimulationConfig) -> Tuple[bytes, bytes, dict]:
    """
    Split concatenated rx_symbols back into frames using tx_meta, run equalization and demodulation,
    then reverse interleaving/FEC and reassemble.
    Returns (app_header_bytes, payload_bytes, stats_dict)
    """
    mod = Modulator(tx_meta["mod_scheme"])
    frames_bits: List[np.ndarray] = []
    est_channels: List[complex] = []

    for i, (start, end) in enumerate(tx_meta["frame_symbol_ranges"]):
        syms = rx_symbols[start:end]
        pilot_len = len(tx_meta["pilots_tx"][i])
        preamble_len = cfg.pilot.preamble_len
        rx_pilot = syms[preamble_len:preamble_len+pilot_len]
        rx_data = syms[preamble_len+pilot_len:]

        # Equalize using pilots
        eq_data, h_hat = equalize(rx_data, tx_meta["pilots_tx"][i], rx_pilot)
        est_channels.append(h_hat)

        # Demod to bits
        bits = mod.demodulate(eq_data)
        frames_bits.append(bits)

    # Reverse interleaving and FEC per frame
    raw_frame_bytes = reverse_fec_and_deinterleave(
        frames_bits,
        tx_meta["orig_bit_lengths"],
        fec_scheme=cfg.link.fec_scheme,
        repeat_k=cfg.link.repeat_k,
        interleaver_depth=cfg.link.interleaver_depth,
        strong_header=cfg.link.strong_header_protection,
        header_copies=cfg.link.header_copies,
        header_rep_k=cfg.link.header_rep_k
    )

    # Reassemble and CRC-check (header copies aware)
    app_hdr_bytes, payload_bytes, stats = reassemble_and_check(
        raw_frame_bytes,
        header_copies=cfg.link.header_copies,
        drop_bad=cfg.link.drop_bad_frames,
        verbose=cfg.link.verbose
    )

    # attach channel estimates (for logging)
    stats["h_estimates"] = est_channels
    return app_hdr_bytes, payload_bytes, stats


# transmitter/send.py
"""
Transmitter pipeline: App -> Link -> PHY assembly.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from common.config import SimulationConfig
from common.utils import bytes_to_bits
from common.byte_mapping import map_bytes
from data_link_layer.encoding import segment_message, apply_fec_and_interleave
from physical_layer.modulation import Modulator, build_phy_frame

def build_transmission(app_header_bytes: bytes, payload_bytes: bytes, cfg: SimulationConfig):
    # NEW: apply byte mapping before segmentation (affects DATA frames only)
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    mapped_payload = map_bytes(
        payload_bytes,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed
    )

    # 1) Link segmentation (with APP header copies) – pass mapped payload
    frames = segment_message(
        app_header_bytes,
        mapped_payload,
        mtu_bytes=cfg.link.mtu_bytes,
        header_copies=cfg.link.header_copies
    )

    # 2) Apply FEC + interleaving per frame (+ header repetition if enabled)
    enc_bits_list, orig_bit_lengths = apply_fec_and_interleave(
        frames,
        fec_scheme=cfg.link.fec_scheme,
        repeat_k=cfg.link.repeat_k,
        interleaver_depth=cfg.link.interleaver_depth,
        strong_header=cfg.link.strong_header_protection,
        header_copies=cfg.link.header_copies,
        header_rep_k=cfg.link.header_rep_k
    )

    # 3) PHY: preamble + pilots + data symbols per frame
    mod = Modulator(cfg.mod.scheme)
    frame_symbol_ranges: List[Tuple[int,int]] = []
    pilots_tx: List[np.ndarray] = []
    data_symbol_counts: List[int] = []
    all_syms: List[np.ndarray] = []

    cursor = 0
    for bits in enc_bits_list:
        syms, pilot, data_syms = build_phy_frame(bits, mod,
            preamble_len_bits=cfg.pilot.preamble_len,
            pilot_len_symbols=cfg.pilot.pilot_len)
        all_syms.append(syms)
        pilots_tx.append(pilot)
        data_symbol_counts.append(len(data_syms))
        frame_symbol_ranges.append((cursor, cursor+len(syms)))
        cursor += len(syms)

    tx_symbols = np.concatenate(all_syms) if all_syms else np.zeros(0, dtype=np.complex128)
    tx_meta = {
        "mod_scheme": cfg.mod.scheme,
        "frame_symbol_ranges": frame_symbol_ranges,
        "data_symbol_counts": data_symbol_counts,
        "pilots_tx": pilots_tx,
        "orig_bit_lengths": orig_bit_lengths,
    }
    return tx_symbols, tx_meta
