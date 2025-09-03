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

# NEW (decoder-side):
- SEG_DECODER_FALLBACK: "uniform" (default), "clamp", or "mod".
    * uniform: any out-of-range ID is remapped to Uniform{0..K-1} (eliminates color bias)
    * clamp:   old behavior (min(id, K-1))  ← 黄色に寄る現象の原因
    * mod:     id % K
- SEG_DECODER_SEED: RNG seed for "uniform" fallback (optional)
- SEG_DECODER_MAJ3x3: "1" to apply 3x3 majority smoothing after fallback (default "0")
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
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
    return (rgb[..., 0] >= thresh) & (rgb[..., 1] >= thresh) & (rgb[..., 2] >= thresh)

def _suppress_white_boundaries(rgb: np.ndarray, white_thresh: int = 250, iters: int = 2) -> np.ndarray:
    H, W, _ = rgb.shape
    out = rgb.copy()
    mask = _white_mask(out, white_thresh)
    if not mask.any():
        return out
    for _ in range(max(1, iters)):
        if not mask.any(): break
        neighs = []; neigh_nonwhite = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                py0, py1 = (max(dy, 0), max(-dy, 0))
                px0, px1 = (max(dx, 0), max(-dx, 0))
                shifted = np.pad(out, ((py0,py1),(px0,px1),(0,0)), mode="edge")
                shifted = shifted[py1:py1+H, px1:px1+W, :]
                neighs.append(shifted)
                neigh_nonwhite.append(~_white_mask(shifted, white_thresh))
        stack = np.stack(neighs, axis=0)
        stack_mask = np.stack(neigh_nonwhite, axis=0)
        remaining = mask.copy()
        for n in range(stack.shape[0]):
            m = stack_mask[n] & remaining
            if not m.any(): continue
            out[m] = stack[n][m]
            remaining[m] = False
        mask = _white_mask(out, white_thresh)
    if mask.any():
        out[mask] = np.array([32, 32, 32], dtype=np.uint8)
    return out

def _build_seg_ids_and_palette(rgb_arr: np.ndarray, white_thresh: int) -> tuple[np.ndarray, np.ndarray, str]:
    H, W, _ = rgb_arr.shape
    flat = rgb_arr.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    white_like = (uniq[:, 0] >= white_thresh) & (uniq[:, 1] >= white_thresh) & (uniq[:, 2] >= white_thresh)
    if white_like.any():
        non_white_idx = np.where(~white_like)[0]
        if non_white_idx.size == 0:
            palette = np.array([[32, 32, 32]], dtype=np.uint8)
            id_map = np.zeros((H, W), dtype=np.uint8)
            return id_map, palette, "id8"
        idx_map = np.arange(uniq.shape[0])
        for wi in np.where(white_like)[0]:
            d = np.sum((uniq[non_white_idx].astype(np.int32) - uniq[wi].astype(np.int32))**2, axis=1)
            nearest = non_white_idx[np.argmin(d)]
            idx_map[idx_map == wi] = nearest
        inv = idx_map[inv]
        uniq = uniq[~white_like]
    K = uniq.shape[0]
    if K <= 256:
        id_map = inv.astype(np.uint8).reshape(H, W); enc = "id8"
    else:
        id_map = inv.astype(np.uint16).reshape(H, W); enc = "id16"
    palette = uniq.astype(np.uint8)
    return id_map, palette, enc

def _safe_color_lut(K: int) -> np.ndarray:
    if K <= 0:
        return np.zeros((1, 3), dtype=np.uint8)
    hues = np.linspace(0.0, 1.0, K, endpoint=False)
    sat, val = 0.85, 0.72
    rgb = []
    for h in hues:
        i = int(h*6) % 6; f = (h*6) - int(h*6)
        p = val*(1-sat); q = val*(1-f*sat); t = val*(1-(1-f)*sat)
        if   i==0: r,g,b = val, t, p
        elif i==1: r,g,b = q, val, p
        elif i==2: r,g,b = p, val, t
        elif i==3: r,g,b = p, q, val
        elif i==4: r,g,b = t, p, val
        else:      r,g,b = val, p, q
        rgb.append([int(r*255), int(g*255), int(b*255)])
    return np.array(rgb, dtype=np.uint8)

# ----------------- Semantic ID corruption (TX-side) -----------------
def _apply_id_noise_uniform(ids: np.ndarray, K: int, p: float, seed: int | None) -> np.ndarray:
    if p <= 0.0 or K <= 1:
        return ids
    rng = np.random.default_rng(seed if seed is not None else (0xC0FFEE ^ (K * 2654435761 & 0xFFFFFFFF)))
    mask = rng.random(ids.shape, dtype=np.float32) < float(p)
    if not mask.any():
        return ids
    out = ids.copy()
    orig = out[mask].astype(np.int64)
    draw = rng.integers(0, K - 1, size=orig.size, dtype=np.int64)
    new_ids = draw + (draw >= orig)  # skip original
    out[mask] = (new_ids.astype(np.uint8) if out.dtype == np.uint8 else new_ids.astype(np.uint16))
    return out

# ----------------- Encode -----------------
def serialize_content(modality: str, content_path: str, text_encoding: str="utf-8",
                      validate_image_mode: bool=True) -> tuple[AppHeader, bytes]:
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
        strip_white = (os.getenv("SEG_STRIP_WHITE", "1").strip() != "0")
        white_thresh = int(os.getenv("SEG_WHITE_THRESH", "250"))

        rgb = load_image_to_array(content_path, modality="segmentation", validate_mode=True)
        if strip_white:
            rgb = _suppress_white_boundaries(rgb, white_thresh=white_thresh, iters=2)

        h, w, _ = rgb.shape
        id_map, palette, enc = _build_seg_ids_and_palette(rgb, white_thresh=white_thresh)
        _set_seg_palette(palette)

        p_env = float(os.getenv("SEG_ID_NOISE_P", "0.0") or "0.0")
        seed_env = os.getenv("SEG_ID_NOISE_SEED", "").strip()
        seed = int(seed_env) if seed_env != "" else None
        if p_env > 0.0:
            id_map = _apply_id_noise_uniform(id_map, K=int(palette.shape[0]), p=p_env, seed=seed)

        if enc == "id8":
            id_bytes = id_map.reshape(-1).tobytes(); bits = 8
        else:
            id_bytes = id_map.astype('>u2').reshape(-1).tobytes(); bits = 16

        hdr = AppHeader(version=1, modality="segmentation", height=h, width=w, channels=1,
                        bits_per_sample=bits, payload_len_bytes=len(id_bytes))
        return hdr, id_bytes

    raise ValueError("Unknown modality")

# ----------------- (optional) majority smoothing -----------------
def _majority3x3(ids: np.ndarray) -> np.ndarray:
    H, W = ids.shape
    out = ids.copy()
    pad = np.pad(ids, ((1,1),(1,1)), mode="edge")
    for r in range(H):
        for c in range(W):
            block = pad[r:r+3, c:c+3].reshape(-1)
            vals, cnt = np.unique(block, return_counts=True)
            out[r, c] = vals[np.argmax(cnt)]
    return out

# ----------------- Decode -----------------
def deserialize_content(hdr: AppHeader, payload_bytes: bytes, text_encoding: str="utf-8",
                        text_errors: str="replace") -> tuple[str, np.ndarray]:
    if hdr.modality == "text":
        s = text_bytes_to_string(payload_bytes, encoding=text_encoding, errors=text_errors)
        return s, np.array([], dtype=np.uint8)

    if hdr.modality in ("edge","depth"):
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        if hdr.modality == "edge":
            arr = (arr >= 128).astype(np.uint8) * 255
        return "", arr

    if hdr.modality == "segmentation":
        if hdr.bits_per_sample <= 8:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        else:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.dtype('>u2')).astype(np.uint16, copy=False)

        pal = _get_seg_palette()
        if pal is None or pal.size == 0:
            K = int(ids.max()) + 1 if ids.size > 0 else 1
            pal = _safe_color_lut(K)

        K = int(pal.shape[0])

        # ---- NEW: out-of-range ID fallback policy ----
        fallback = os.getenv("SEG_DECODER_FALLBACK", "uniform").strip().lower()
        if fallback == "uniform":
            invalid = (ids >= K)
            if invalid.any():
                seed_str = os.getenv("SEG_DECODER_SEED", "").strip()
                seed = int(seed_str) if seed_str != "" else None
                rng = np.random.default_rng(seed if seed is not None else (0xDEADBEEF ^ (K * 11400714819323198485 & 0xFFFFFFFFFFFF)))
                repl = rng.integers(0, K, size=int(invalid.sum()), dtype=np.int64)
                ids_fixed = ids.copy()
                ids_fixed[invalid] = (repl.astype(np.uint8) if ids.dtype == np.uint8 else repl.astype(np.uint16))
        elif fallback == "mod":
            ids_fixed = (ids % K).astype(ids.dtype, copy=False)
        else:  # "clamp" (legacy)
            ids_fixed = np.minimum(ids, K - 1)
        # ---------------------------------------------

        # optional 3x3 majority smoothing
        if os.getenv("SEG_DECODER_MAJ3x3", "0").strip() == "1":
            ids_fixed = _majority3x3(ids_fixed)

        rgb = pal[ids_fixed]
        return "", rgb.astype(np.uint8)

    raise ValueError("Unknown modality in header")

# ----------------- Save output -----------------
def save_output(hdr: AppHeader, text_str: str, img_arr: np.ndarray, out_path: str) -> None:
    if hdr.modality == "text":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_str); return
    if hdr.modality in ("edge","depth"):
        Image.fromarray(img_arr.astype(np.uint8), mode="L").save(out_path, format="PNG"); return
    if hdr.modality == "segmentation":
        Image.fromarray(img_arr.astype(np.uint8), mode="RGB").save(out_path, format="PNG"); return
    raise ValueError("Unsupported modality")
