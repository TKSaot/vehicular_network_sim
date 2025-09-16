
# multimodal_ofdm/application.py
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

# ---- White suppression for segmentation ----
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
        white_thresh = 250
        rgb = _suppress_white_boundaries(rgb, white_thresh=white_thresh, iters=2)
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
    import numpy as np
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
        return "", arr

    if hdr.modality == "depth":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        return "", arr

    if hdr.modality == "segmentation":
        if hdr.bits_per_sample <= 8:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        else:
            ids = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.dtype(">u2")).astype(np.uint16)
        pal = _get_palette()
        if pal is None or pal.size == 0:
            K = int(ids.max()) + 1 if ids.size > 0 else 1
            pal = np.stack([np.linspace(0,255,K), np.roll(np.linspace(0,255,K),1), np.roll(np.linspace(0,255,K),2)], axis=1).astype(np.uint8)
        K = int(pal.shape[0])
        ids = np.minimum(ids, (K-1))
        rgb = pal[ids]
        return "", rgb.astype(np.uint8)

    raise ValueError("Unknown modality")
