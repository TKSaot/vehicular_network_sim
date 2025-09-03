# app_layer/application.py
"""
Application-layer serializers for modalities (Text, Edge, Depth, Segmentation).

Now decoder behavior is driven by AppConfig (passed from configs/*_config.py),
instead of environment variables. (Env vars are no longer needed.)

Key features:
- Segmentation: ID-map transmission, robust white suppression, contiguous IDs 0..K-1.
  Decoder supports: out-of-range fallback (uniform/clamp/mod), 3x3 majority (iters),
  and edge-assisted majority using an external edge image (PNG/L8).
- Edge: binary majority (3/5) and median (3/5) with an option to preserve thin lines.
- Depth: median (3/5) and lightweight bilateral(5x5).

NOTE: If app_cfg is None, reasonable defaults are used (backward compatible).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, Optional
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

# ----------------- I/O helpers -----------------
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
        return (arr >= 128).astype(np.uint8) * 255
    if modality == "depth":
        if validate_mode and im.mode != "L":
            im = im.convert("L")
        return np.array(im).astype(np.uint8)
    if modality == "segmentation":
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im).astype(np.uint8)
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

# ----------------- White suppression (segmentation preproc) -----------------
def _white_mask(rgb: np.ndarray, thresh: int) -> np.ndarray:
    return (rgb[...,0] >= thresh) & (rgb[...,1] >= thresh) & (rgb[...,2] >= thresh)
def _suppress_white_boundaries(rgb: np.ndarray, white_thresh: int = 250, iters: int = 2) -> np.ndarray:
    H, W, _ = rgb.shape
    out = rgb.copy()
    mask = _white_mask(out, white_thresh)
    if not mask.any(): return out
    for _ in range(max(1, iters)):
        if not mask.any(): break
        neighs, neigh_nonwhite = [], []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                py0, py1 = (max(dy,0), max(-dy,0))
                px0, px1 = (max(dx,0), max(-dx,0))
                shifted = np.pad(out, ((py0,py1),(px0,px1),(0,0)), mode="edge")
                shifted = shifted[py1:py1+H, px1:px1+W, :]
                neighs.append(shifted)
                neigh_nonwhite.append(~_white_mask(shifted, white_thresh))
        stack = np.stack(neighs, axis=0); stack_mask = np.stack(neigh_nonwhite, axis=0)
        remaining = mask.copy()
        for n in range(stack.shape[0]):
            m = stack_mask[n] & remaining
            if not m.any(): continue
            out[m] = stack[n][m]; remaining[m] = False
        mask = _white_mask(out, white_thresh)
    if mask.any():
        out[mask] = np.array([32,32,32], dtype=np.uint8)
    return out

# ----------------- Segmentation: build IDs & palette -----------------
def _build_seg_ids_and_palette(rgb_arr: np.ndarray, white_thresh: int) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Robust: remove white-like colors; map to contiguous 0..K-1 by explicit old->new remap.
    """
    H, W, _ = rgb_arr.shape
    flat = rgb_arr.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    white_like = (uniq[:,0] >= white_thresh) & (uniq[:,1] >= white_thresh) & (uniq[:,2] >= white_thresh)
    if white_like.all():
        palette = np.array([[32,32,32]], dtype=np.uint8)
        id_map = np.zeros((H, W), dtype=np.uint8)
        return id_map, palette, "id8"

    nonwhite_idx = np.where(~white_like)[0]
    nonwhite_colors = uniq[nonwhite_idx].astype(np.int32)

    old2new = np.full(uniq.shape[0], -1, dtype=np.int64)
    for new_i, old_i in enumerate(nonwhite_idx):
        old2new[old_i] = new_i
    if white_like.any():
        for wi in np.where(white_like)[0]:
            col = uniq[wi].astype(np.int32)
            d2 = np.sum((nonwhite_colors - col)**2, axis=1)
            old2new[wi] = int(np.argmin(d2))
    inv_new = old2new[inv]; inv_new = np.maximum(inv_new, 0)

    K = nonwhite_idx.size
    if K <= 256:
        id_map = inv_new.astype(np.uint8).reshape(H, W); enc = "id8"
    else:
        id_map = inv_new.astype(np.uint16).reshape(H, W); enc = "id16"
    palette = uniq[nonwhite_idx].astype(np.uint8)
    return id_map, palette, enc

def _safe_color_lut(K: int) -> np.ndarray:
    if K <= 0: return np.zeros((1,3), dtype=np.uint8)
    hues = np.linspace(0.0, 1.0, K, endpoint=False); sat, val = 0.85, 0.72
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

# ----------------- Optional TX-side semantic ID noise -----------------
def _apply_id_noise_uniform(ids: np.ndarray, K: int, p: float, seed: Optional[int]) -> np.ndarray:
    if p <= 0.0 or K <= 1: return ids
    rng = np.random.default_rng(seed if seed is not None else (0xC0FFEE ^ (K * 2654435761 & 0xFFFFFFFF)))
    mask = rng.random(ids.shape, dtype=np.float32) < float(p)
    if not mask.any(): return ids
    out = ids.copy()
    orig = out[mask].astype(np.int64)
    draw = rng.integers(0, K - 1, size=orig.size, dtype=np.int64)
    new_ids = draw + (draw >= orig)  # skip original
    out[mask] = (new_ids.astype(np.uint8) if out.dtype == np.uint8 else new_ids.astype(np.uint16))
    return out

# ----------------- Edge/Depth helpers -----------------
def _median_filter_uint8(img: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    assert k in (3,5)
    out = img.astype(np.uint8, copy=True)
    for _ in range(max(1, iters)):
        pad = k // 2
        padded = np.pad(out, ((pad,pad),(pad,pad)), mode="edge")
        stacks = []
        for dy in range(k):
            for dx in range(k):
                stacks.append(padded[dy:dy+out.shape[0], dx:dx+out.shape[1]])
        out = np.median(np.stack(stacks, axis=-1), axis=-1).astype(np.uint8)
    return out

def _edge_binary_majority(img01: np.ndarray, k: int = 3, thr: Optional[int] = None,
                          iters: int = 1, preserve_lines: bool = True) -> np.ndarray:
    """
    Majority on {0,1}. With preserve_lines=True, avoid erasing thin ridges:
    if center==1 and neighbor-count>=2, keep 1 even if below threshold.
    """
    assert k in (3,5)
    a = (img01 > 0).astype(np.uint8)
    if thr is None: thr = (k*k)//2 + 1
    for _ in range(max(1, iters)):
        pad = k // 2
        padded = np.pad(a, ((pad,pad),(pad,pad)), mode="edge")
        sums = np.zeros_like(a, dtype=np.uint16)
        for dy in range(k):
            for dx in range(k):
                sums += padded[dy:dy+a.shape[0], dx:dx+a.shape[1]]
        maj = (sums >= thr).astype(np.uint8)
        if preserve_lines:
            # keep center if it has >=2 active neighbors (8-neighborhood)
            neigh = sums - a  # exclude center
            keep = (a == 1) & (neigh >= 2) & (maj == 0)
            maj[keep] = 1
        a = maj
    return a

def _majority3x3_ids(ids: np.ndarray, iters: int = 1) -> np.ndarray:
    H, W = ids.shape
    out = ids.copy()
    for _ in range(max(1, iters)):
        pad = np.pad(out, ((1,1),(1,1)), mode="edge")
        nxt = out.copy()
        for r in range(H):
            for c in range(W):
                block = pad[r:r+3, c:c+3].reshape(-1)
                vals, counts = np.unique(block, return_counts=True)
                nxt[r,c] = vals[np.argmax(counts)]
        out = nxt
    return out

def _edge_guided_majority_ids(ids: np.ndarray, edge_l8: np.ndarray, iters: int = 1) -> np.ndarray:
    """Edge-guided 3x3 majority: prohibit crossing edges (edge=255)."""
    H, W = ids.shape
    edge = (edge_l8 >= 128)
    out = ids.copy()
    for _ in range(max(1, iters)):
        pad_ids  = np.pad(out,  ((1,1),(1,1)), mode="edge")
        pad_edge = np.pad(edge, ((1,1),(1,1)), mode="edge")
        nxt = out.copy()
        for r in range(H):
            for c in range(W):
                if pad_edge[r+1, c+1]:
                    nxt[r,c] = out[r,c]
                    continue
                votes = []
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dy==0 and dx==0: continue
                        if pad_edge[r+1+dy, c+1+dx]:  # don't cross edges
                            continue
                        votes.append(pad_ids[r+1+dy, c+1+dx])
                if votes:
                    vals, counts = np.unique(np.array(votes), return_counts=True)
                    nxt[r,c] = vals[np.argmax(counts)]
                else:
                    nxt[r,c] = out[r,c]
        out = nxt
    return out

def _bilateral5_uint8(img: np.ndarray, sigma_s: float = 1.6, sigma_r: float = 12.0, iters: int = 1) -> np.ndarray:
    """
    Lightweight 5x5 bilateral filter for uint8 images.
    sigma_s: spatial std (pixels), sigma_r: range std (intensity).
    """
    out = img.astype(np.float32, copy=True)
    # precompute spatial weights (5x5)
    k = 5; pad = 2
    yy, xx = np.mgrid[-pad:pad+1, -pad:pad+1]
    spatial = np.exp(-(xx**2 + yy**2) / (2.0 * (sigma_s**2))).astype(np.float32)

    for _ in range(max(1, iters)):
        padded = np.pad(out, ((pad,pad),(pad,pad)), mode="edge")
        num = np.zeros_like(out, dtype=np.float32)
        den = np.zeros_like(out, dtype=np.float32)
        for dy in range(-pad, pad+1):
            for dx in range(-pad, pad+1):
                w_s = spatial[dy+pad, dx+pad]
                neigh = padded[dy+pad:dy+pad+out.shape[0], dx+pad:dx+pad+out.shape[1]]
                diff = neigh - out
                w_r = np.exp(-(diff**2) / (2.0 * (sigma_r**2))).astype(np.float32)
                w = w_s * w_r
                num += w * neigh
                den += w
        out = num / np.maximum(den, 1e-8)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

# ----------------- Serialize content (TX) -----------------
def serialize_content(modality: str, content_path: str, app_cfg: Optional[Any] = None,
                      text_encoding: str="utf-8", validate_image_mode: bool=True) -> tuple[AppHeader, bytes]:
    if modality == "text":
        data = load_text_as_bytes(content_path, encoding=text_encoding)
        hdr = AppHeader(version=1, modality="text", payload_len_bytes=len(data))
        return hdr, data

    if modality in ("edge","depth"):
        arr = load_image_to_array(content_path, modality=modality, validate_mode=validate_image_mode)
        h, w = arr.shape
        payload = arr.reshape(-1).tobytes()
        hdr = AppHeader(version=1, modality=modality, height=h, width=w, channels=1,
                        bits_per_sample=8, payload_len_bytes=len(payload))
        return hdr, payload

    if modality == "segmentation":
        strip_white  = True if app_cfg is None else bool(getattr(app_cfg, "seg_strip_white", True))
        white_thresh = 250 if app_cfg is None else int(getattr(app_cfg, "seg_white_thresh", 250))

        rgb = load_image_to_array(content_path, modality="segmentation", validate_mode=True)
        if strip_white:
            rgb = _suppress_white_boundaries(rgb, white_thresh=white_thresh, iters=2)

        h, w, _ = rgb.shape
        id_map, palette, enc = _build_seg_ids_and_palette(rgb, white_thresh=white_thresh)
        _set_seg_palette(palette)

        # Optional semantic noise (TX-side) -- default OFF
        p = 0.0 if app_cfg is None else float(getattr(app_cfg, "seg_tx_noise_p", 0.0))
        seed = None if app_cfg is None else getattr(app_cfg, "seg_tx_noise_seed", None)
        if p > 0.0:
            id_map = _apply_id_noise_uniform(id_map, K=int(palette.shape[0]), p=p, seed=seed)

        if enc == "id8":
            id_bytes = id_map.reshape(-1).tobytes(); bits = 8
        else:
            id_bytes = id_map.astype('>u2').reshape(-1).tobytes(); bits = 16

        hdr = AppHeader(version=1, modality="segmentation", height=h, width=w, channels=1,
                        bits_per_sample=bits, payload_len_bytes=len(id_bytes))
        return hdr, id_bytes

    raise ValueError("Unknown modality")

# ----------------- Deserialize content (RX) -----------------
def deserialize_content(hdr: AppHeader, payload_bytes: bytes, app_cfg: Optional[Any] = None,
                        text_encoding: str="utf-8", text_errors: str="replace") -> tuple[str, np.ndarray]:
    if hdr.modality == "text":
        s = text_bytes_to_string(payload_bytes, encoding=text_encoding, errors=text_errors)
        return s, np.array([], dtype=np.uint8)

    if hdr.modality == "edge":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        arr = (arr >= 128).astype(np.uint8) * 255
        # Decoder post-filter (config-driven)
        edec = None if app_cfg is None else getattr(app_cfg, "edgedec", None)
        if edec is not None:
            mode = getattr(edec, "denoise", "none")
            iters = int(getattr(edec, "iters", 1))
            if mode.startswith("maj"):
                k = 3 if mode == "maj3" else 5
                thr = getattr(edec, "thresh", None)
                preserve = bool(getattr(edec, "preserve_lines", True))
                a01 = _edge_binary_majority(arr, k=k, thr=thr, iters=iters, preserve_lines=preserve)
                arr = (a01 * 255).astype(np.uint8)
            elif mode.startswith("median"):
                k = 3 if mode == "median3" else 5
                arr = _median_filter_uint8(arr, k=k, iters=iters)
                arr = (arr >= 128).astype(np.uint8) * 255
        return "", arr

    if hdr.modality == "depth":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        ddec = None if app_cfg is None else getattr(app_cfg, "depthdec", None)
        if ddec is not None:
            mode = getattr(ddec, "filt", "median3")
            iters = int(getattr(ddec, "iters", 1))
            if mode == "median3":
                arr = _median_filter_uint8(arr, k=3, iters=iters)
            elif mode == "median5":
                arr = _median_filter_uint8(arr, k=5, iters=iters)
            elif mode == "bilateral5":
                sig_s = float(getattr(ddec, "sigma_s", 1.6))
                sig_r = float(getattr(ddec, "sigma_r", 12.0))
                arr = _bilateral5_uint8(arr, sigma_s=sig_s, sigma_r=sig_r, iters=iters)
            # else: none
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

        # Out-of-range fallback (config-driven; default uniform)
        sdec = None if app_cfg is None else getattr(app_cfg, "segdec", None)
        fallback = "uniform" if sdec is None else getattr(sdec, "fallback", "uniform")
        if fallback == "uniform":
            invalid = (ids >= K)
            if invalid.any():
                seed = None if sdec is None else getattr(sdec, "seed", None)
                rng = np.random.default_rng(seed if seed is not None else (0xDEADBEEF ^ (K * 11400714819323198485 & 0xFFFFFFFFFFFF)))
                repl = rng.integers(0, K, size=int(invalid.sum()), dtype=np.int64)
                ids = ids.copy()
                ids[invalid] = (repl.astype(np.uint8) if ids.dtype == np.uint8 else repl.astype(np.uint16))
        elif fallback == "mod":
            ids = (ids % K).astype(ids.dtype, copy=False)
        else:  # "clamp" (legacy)  ← 旧実装ではこれのみだったため色偏りが出やすかった
            ids = np.minimum(ids, K - 1)  # :contentReference[oaicite:6]{index=6}

        # Majority smoothing (optional)
        if sdec is not None and bool(getattr(sdec, "maj3x3", False)):
            iters = int(getattr(sdec, "maj_iters", 1))
            use_edge = bool(getattr(sdec, "use_edge_guidance", False))
            if use_edge:
                path = getattr(sdec, "edge_guide_path", None)
                if path and os.path.isfile(path):
                    eg = Image.open(path).convert("L")
                    eg_arr = np.array(eg).astype(np.uint8)
                    H = min(ids.shape[0], eg_arr.shape[0]); W = min(ids.shape[1], eg_arr.shape[1])
                    ids = ids[:H,:W]; eg_arr = eg_arr[:H,:W]
                    ids = _edge_guided_majority_ids(ids, eg_arr, iters=iters)
                else:
                    ids = _majority3x3_ids(ids, iters=iters)
            else:
                ids = _majority3x3_ids(ids, iters=iters)

        rgb = pal[np.minimum(ids, K - 1)]
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
