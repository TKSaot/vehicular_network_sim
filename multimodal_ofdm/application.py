from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Any, Tuple
import numpy as np
from PIL import Image

# ---- App header ----
@dataclass
class AppHeader:
    version: int
    modality: Literal["text","edge","depth","segmentation"]
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
        b[2:4] = int(self.height).to_bytes(2, "big")
        b[4:6] = int(self.width).to_bytes(2, "big")
        b[6] = self.channels & 0xFF
        b[7] = self.bits_per_sample & 0xFF
        b[8:12] = int(self.payload_len_bytes).to_bytes(4, "big")
        return bytes(b)

    @staticmethod
    def from_bytes(b: bytes) -> "AppHeader":
        if len(b) < 16: raise ValueError("AppHeader too short")
        version = b[0]; code = b[1]
        mapping = {0:"text",1:"edge",2:"depth",3:"segmentation"}
        if code not in mapping: raise ValueError("Invalid modality code")
        modality = mapping[code]
        height = int.from_bytes(b[2:4], "big")
        width  = int.from_bytes(b[4:6], "big")
        channels = b[6]; bps = b[7]
        payload_len = int.from_bytes(b[8:12], "big")
        return AppHeader(version, modality, height, width, channels, bps, payload_len)

# ---- In-process segmentation palette ----
_SEG_PALETTE: np.ndarray | None = None
def _set_palette(p: np.ndarray) -> None:
    global _SEG_PALETTE; _SEG_PALETTE = p.astype(np.uint8, copy=False)
def _get_palette() -> np.ndarray | None:
    return _SEG_PALETTE

# ---- I/O helpers ----
def load_text_as_bytes(path: str, encoding: str="utf-8") -> bytes:
    with open(path, "r", encoding=encoding) as f:
        return f.read().encode(encoding)

def _load_image(path: str, mode: str) -> np.ndarray:
    im = Image.open(path)
    if mode == "L" and im.mode != "L": im = im.convert("L")
    if mode == "RGB" and im.mode != "RGB": im = im.convert("RGB")
    return np.array(im)

# ---- White suppression for segmentation (TX pre-clean) ----
def _white_mask(rgb: np.ndarray, t: int) -> np.ndarray:
    return (rgb[...,0] >= t) & (rgb[...,1] >= t) & (rgb[...,2] >= t)

def _suppress_white_boundaries(rgb: np.ndarray, white_thresh: int = 250, iters: int = 2) -> np.ndarray:
    H,W,_ = rgb.shape
    out = rgb.copy()
    mask = _white_mask(out, white_thresh)
    if not mask.any(): return out
    for _ in range(max(1, iters)):
        if not mask.any(): break
        neighs, nmask = [], []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                py0,py1 = (max(dy,0), max(-dy,0)); px0,px1 = (max(dx,0), max(-dx,0))
                sh = np.pad(out, ((py0,py1),(px0,px1),(0,0)), mode="edge")[py1:py1+H, px1:px1+W]
                neighs.append(sh)
                nmask.append(~_white_mask(sh, white_thresh))
        stack = np.stack(neighs,0); sm = np.stack(nmask,0)
        remaining = mask.copy()
        for n in range(stack.shape[0]):
            m = sm[n] & remaining
            if not m.any(): continue
            out[m] = stack[n][m]; remaining[m] = False
        mask = _white_mask(out, white_thresh)
    if mask.any(): out[mask] = np.array([32,32,32], dtype=np.uint8)
    return out

def _build_seg_ids_and_palette(rgb_arr: np.ndarray, white_thresh: int) -> tuple[np.ndarray, np.ndarray, str]:
    H,W,_ = rgb_arr.shape
    flat = rgb_arr.reshape(-1,3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    white_like = (uniq[:,0] >= white_thresh) & (uniq[:,1] >= white_thresh) & (uniq[:,2] >= white_thresh)
    if white_like.all():
        pal = np.array([[32,32,32]], dtype=np.uint8)
        ids = np.zeros((H,W), dtype=np.uint8)
        return ids, pal, "id8"
    nonwhite_idx = np.where(~white_like)[0]
    nonwhite_cols = uniq[nonwhite_idx].astype(np.int32)
    old2new = np.full(uniq.shape[0], -1, dtype=np.int64)
    for new_i, old_i in enumerate(nonwhite_idx): old2new[old_i] = new_i
    if white_like.any():
        for wi in np.where(white_like)[0]:
            col = uniq[wi].astype(np.int32)
            d2 = np.sum((nonwhite_cols - col)**2, axis=1)
            old2new[wi] = int(np.argmin(d2))
    inv_new = np.maximum(old2new[inv], 0)
    K = nonwhite_idx.size
    if K <= 256:
        ids = inv_new.astype(np.uint8).reshape(H,W); enc="id8"
    else:
        ids = inv_new.astype(np.uint16).reshape(H,W); enc="id16"
    palette = uniq[nonwhite_idx].astype(np.uint8)
    return ids, palette, enc

# ---------- RX post-processing helpers (new) ----------
def _shift2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Edge-padded shift used throughout; preserves shape."""
    H, W = arr.shape
    py0, py1 = (max(dy, 0), max(-dy, 0))
    px0, px1 = (max(dx, 0), max(-dx, 0))
    pad = np.pad(arr, ((py0, py1), (px0, px1)), mode="edge")
    return pad[py1:py1+H, px1:px1+W]

def _neighbors_stack2d(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """Stack of (2r+1)^2 shifted arrays including center. shape=(K,H,W)."""
    outs = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            outs.append(_shift2d(arr, dy, dx))
    return np.stack(outs, axis=0)  # (K,H,W)

def _boundary_mask_ids(ids: np.ndarray) -> np.ndarray:
    """4-neighborhood boundary pixels in an ID map."""
    up = _shift2d(ids, -1, 0)
    dn = _shift2d(ids,  1, 0)
    lf = _shift2d(ids,  0,-1)
    rt = _shift2d(ids,  0, 1)
    return (ids != up) | (ids != dn) | (ids != lf) | (ids != rt)

def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return mask
    acc = mask.copy()
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy == 0 and dx == 0: continue
            acc |= _shift2d(mask, dy, dx)
    return acc

def _majority3_ids(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    3x3 majority (mode) on integer IDs.
    Returns (mode_ids, mode_count_fraction). No SciPy; fully vectorized.
    """
    S = _neighbors_stack2d(ids, radius=1)          # (9,H,W)
    # counts[i] = how many neighbors equal S[i]
    eq = (S[:, None, :, :] == S[None, :, :, :])    # (9,9,H,W) boolean
    counts = eq.sum(axis=0)                         # (9,H,W)
    best_idx = counts.argmax(axis=0)                # (H,W)
    best_cnt = counts.max(axis=0).astype(np.float32)
    mode_ids = np.take_along_axis(S, best_idx[None, :, :], axis=0)[0]
    frac = best_cnt / 9.0
    return mode_ids.astype(ids.dtype, copy=False), frac

def _postprocess_seg_ids(ids: np.ndarray,
                         mode: str = "strong",
                         iters: int = 2,
                         tau: float = 0.6) -> np.ndarray:
    """
    Boundary-aware interior majority. 3x3 mode with confidence >= tau on interior only.
    `majority5` and `strong` widen the boundary belt (radius=2) and run more passes.
    """
    if mode == "none" or ids.size == 0:
        return ids
    belt_r = 1
    passes = max(1, int(iters))
    if mode in ("majority5", "strong"):
        belt_r = 2
        passes = max(passes, 2)
        if mode == "strong":
            tau = min(tau, 0.55)  # slightly easier to flip obvious interiors

    out = ids.copy()
    for _ in range(passes):
        B = _dilate_mask(_boundary_mask_ids(out), belt_r)
        interior = ~B
        mode_ids, frac = _majority3_ids(out)
        upd = interior & (frac >= float(tau)) & (mode_ids != out)
        out = np.where(upd, mode_ids, out)
    return out

def _denoise_edge_binary(edge: np.ndarray,
                         level: str = "gentle",
                         iters: int = 1) -> np.ndarray:
    """
    Edge-preserving salt&pepper cleanup for 0/255 binary maps.
    Removes isolated dots and fills tiny gaps without eating true thin lines.
    """
    if edge.size == 0 or level == "none":
        return edge
    e = (edge > 0).astype(np.uint8)
    # thresholds by level (neighbors in 8-NN)
    if level == "gentle":
        rm_th, add_th = 1, 7   # remove if <=1 white neighbor; add if >=7 white neighbors
    elif level == "medium":
        rm_th, add_th = 2, 6
    else:  # "strong"
        rm_th, add_th = 3, 5

    for _ in range(max(1, int(iters))):
        # 8-neighborhood count
        neigh = []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                neigh.append(_shift2d(e, dy, dx))
        cnt = np.sum(neigh, axis=0).astype(np.uint8)  # 0..8
        # rules
        remove = (e == 1) & (cnt <= rm_th)
        add    = (e == 0) & (cnt >= add_th)
        e = np.where(remove, 0, e)
        e = np.where(add,    1, e)
    return (e * 255).astype(np.uint8)

def _median_filter_uint8(img: np.ndarray, radius: int = 1, passes: int = 1) -> np.ndarray:
    """Pure-numpy sliding median using stack of shifted windows."""
    out = img.astype(np.uint8, copy=True)
    K = (2*radius + 1) ** 2
    for _ in range(max(1, int(passes))):
        stack = _neighbors_stack2d(out, radius=radius).astype(np.uint8)
        out = np.median(stack, axis=0).astype(np.uint8)
    return out

# ---- Serialize/deserialize ----
def serialize_content(modality: str, path: str, app_cfg: Optional[Any] = None) -> tuple[AppHeader, bytes]:
    if modality == "text":
        data = load_text_as_bytes(path)
        hdr = AppHeader(version=1, modality="text", payload_len_bytes=len(data))
        return hdr, data

    if modality == "edge":
        arr = _load_image(path, "L")
        arr = (arr >= 128).astype(np.uint8) * 255
        h,w = arr.shape
        payload = arr.reshape(-1).tobytes()
        hdr = AppHeader(version=1, modality="edge", height=h, width=w, channels=1,
                        bits_per_sample=8, payload_len_bytes=len(payload))
        return hdr, payload

    if modality == "depth":
        arr = _load_image(path, "L").astype(np.uint8)
        h,w = arr.shape
        payload = arr.reshape(-1).tobytes()
        hdr = AppHeader(version=1, modality="depth", height=h, width=w, channels=1,
                        bits_per_sample=8, payload_len_bytes=len(payload))
        return hdr, payload

    if modality == "segmentation":
        rgb = _load_image(path, "RGB").astype(np.uint8)
        white_thresh = getattr(app_cfg, "seg_white_thresh", 250) if app_cfg is not None else 250
        pre_iters    = getattr(app_cfg, "seg_iters", 2) if app_cfg is not None else 2
        rgb = _suppress_white_boundaries(rgb, white_thresh=white_thresh, iters=int(pre_iters))
        h,w,_ = rgb.shape
        ids, pal, enc = _build_seg_ids_and_palette(rgb, white_thresh=white_thresh)
        _set_palette(pal)
        if enc == "id8":
            id_bytes = ids.reshape(-1).tobytes(); bits=8
        else:
            id_bytes = ids.astype(">u2").reshape(-1).tobytes(); bits=16
        hdr = AppHeader(version=1, modality="segmentation", height=h, width=w, channels=1,
                        bits_per_sample=bits, payload_len_bytes=len(id_bytes))
        return hdr, id_bytes

    raise ValueError("Unknown modality")

def _reshape_bytes_safe(payload_bytes: bytes, shape: tuple[int,...], dtype) -> np.ndarray:
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

def deserialize_content(hdr: AppHeader, payload_bytes: bytes, app_cfg: Optional[Any] = None) -> tuple[str, np.ndarray]:
    if hdr.modality == "text":
        try:
            s = payload_bytes.decode("utf-8", errors="replace")
        except Exception:
            s = ""
        return s, np.array([], dtype=np.uint8)

    if hdr.modality == "edge":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        arr = (arr >= 128).astype(np.uint8) * 255
        # --- RX denoise (gentle by default)
        level = getattr(app_cfg, "edge_denoise", "gentle") if app_cfg is not None else "gentle"
        iters = getattr(app_cfg, "edge_iters", 1) if app_cfg is not None else 1
        if level != "none":
            arr = _denoise_edge_binary(arr, level=level, iters=int(iters))
        return "", arr

    if hdr.modality == "depth":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        # --- RX denoise
        dmode = getattr(app_cfg, "depth_denoise", "median3") if app_cfg is not None else "median3"
        diters = getattr(app_cfg, "depth_iters", 1) if app_cfg is not None else 1
        if dmode != "none":
            r = 1
            if dmode == "median5": r = 2
            arr = _median_filter_uint8(arr, radius=r, passes=int(diters))
        return "", arr

    if hdr.modality == "segmentation":
        if hdr.bits_per_sample <= 8:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        else:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.dtype(">u2")).astype(np.uint16)

        # --- RX denoise on IDs (boundary-aware)
        seg_mode = getattr(app_cfg, "seg_mode", "strong") if app_cfg is not None else "strong"
        seg_iters = getattr(app_cfg, "seg_iters", 2) if app_cfg is not None else 2
        seg_tau = getattr(app_cfg, "seg_consensus_min_frac", 0.6) if app_cfg is not None else 0.6
        if seg_mode != "none":
            ids = _postprocess_seg_ids(ids, mode=seg_mode, iters=int(seg_iters), tau=float(seg_tau))

        pal = _get_palette()
        if pal is None or pal.size == 0:
            K = int(ids.max()) + 1 if ids.size > 0 else 1
            pal = np.stack([np.linspace(0,255,K), np.roll(np.linspace(0,255,K),1), np.roll(np.linspace(0,255,K),2)], axis=1).astype(np.uint8)
        K = int(pal.shape[0])
        ids = np.minimum(ids, (K-1))
        rgb = pal[ids]
        return "", rgb.astype(np.uint8)

    raise ValueError("Unknown modality")
