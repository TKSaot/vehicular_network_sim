# app_layer/application.py
"""
Application-layer serializers for modalities (Text, Edge, Depth, Segmentation).

Decoder behavior is driven by AppConfig (configs/image_config.py)，環境変数は不要です．
- Segmentation: ID マップ送受．受信側は out-of-range fallback（uniform/clamp/mod）と
  ラベルモードフィルタ（3×3/5×5）＋多数派合意（consensus）で内域を強く一様化．
- Edge: 二値画像に対し，多数決／メディアンに加え，形態学的 open/close（細線保護）を実装．
- Depth: 3×3/5×5 メディアンと軽量 5×5 バイラテラル．
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, Optional
import numpy as np
from PIL import Image

# -------- App header --------
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
        channels = b[6]; bps = b[7]
        payload_len = int.from_bytes(b[8:12], 'big')
        return AppHeader(version=version, modality=modality, height=height, width=width,
                         channels=channels, bits_per_sample=bps, payload_len_bytes=payload_len)

# -------- in-process palette for segmentation --------
_SEG_PALETTE_CURRENT: np.ndarray | None = None
def _set_seg_palette(palette: np.ndarray) -> None:
    global _SEG_PALETTE_CURRENT
    _SEG_PALETTE_CURRENT = palette.astype(np.uint8, copy=False)
def _get_seg_palette() -> np.ndarray | None:
    return _SEG_PALETTE_CURRENT

# -------- I/O helpers --------
def load_text_as_bytes(path: str, encoding: str="utf-8") -> bytes:
    with open(path, "r", encoding=encoding) as f:
        return f.read().encode(encoding)

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

# -------- White suppression for segmentation --------
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

# -------- Build segmentation IDs + palette --------
def _build_seg_ids_and_palette(rgb_arr: np.ndarray, white_thresh: int) -> tuple[np.ndarray, np.ndarray, str]:
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
    inv_new = np.maximum(old2new[inv], 0)
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

# -------- Image-domain filters (vectorized) --------
def _median_filter_uint8(img: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    assert k in (3,5)
    out = img.astype(np.uint8, copy=True)
    for _ in range(max(1, iters)):
        pad = k // 2
        padded = np.pad(out, ((pad,pad),(pad,pad)), mode="edge")
        stacks = [padded[dy:dy+out.shape[0], dx:dx+out.shape[1]]
                  for dy in range(k) for dx in range(k)]
        out = np.median(np.stack(stacks, axis=-1), axis=-1).astype(np.uint8)
    return out

def _binary_counts(a01: np.ndarray, k: int) -> np.ndarray:
    """3/5 近傍の1の個数（各画素）を計算．"""
    pad = k // 2
    padded = np.pad(a01, ((pad,pad),(pad,pad)), mode="edge")
    sums = np.zeros_like(a01, dtype=np.uint16)
    for dy in range(k):
        for dx in range(k):
            sums += padded[dy:dy+a01.shape[0], dx:dx+a01.shape[1]]
    return sums

def _binary_erosion(a01: np.ndarray, k: int) -> np.ndarray:
    sums = _binary_counts(a01, k)
    return (sums == (k*k)).astype(np.uint8)

def _binary_dilation(a01: np.ndarray, k: int) -> np.ndarray:
    sums = _binary_counts(a01, k)
    return (sums > 0).astype(np.uint8)

def _binary_open(a01: np.ndarray, k: int, iters: int = 1, preserve_lines: bool = True) -> np.ndarray:
    out = a01.copy()
    for _ in range(max(1, iters)):
        er = _binary_erosion(out, k)
        di = _binary_dilation(er, k)
        out = di
    if preserve_lines:
        # 細線保護：近傍に2つ以上の1がある元1画素は復活
        neigh = _binary_counts(a01, 3) - a01
        keep = (a01 == 1) & (out == 0) & (neigh >= 2)
        out[keep] = 1
    return out

def _binary_close(a01: np.ndarray, k: int, iters: int = 1) -> np.ndarray:
    out = a01.copy()
    for _ in range(max(1, iters)):
        di = _binary_dilation(out, k)
        er = _binary_erosion(di, k)
        out = er
    return out

def _edge_binary_majority(a01: np.ndarray, k: int = 3, thr: Optional[int] = None,
                          iters: int = 1, preserve_lines: bool = True) -> np.ndarray:
    assert k in (3,5)
    a = (a01 > 0).astype(np.uint8)
    if thr is None: thr = (k*k)//2 + 1
    for _ in range(max(1, iters)):
        sums = _binary_counts(a, k)
        maj = (sums >= thr).astype(np.uint8)
        if preserve_lines:
            neigh = _binary_counts(a, 3) - a
            keep = (a == 1) & (maj == 0) & (neigh >= 2)
            maj[keep] = 1
        a = maj
    return a

def _bilateral5_uint8(img: np.ndarray, sigma_s: float = 1.6, sigma_r: float = 12.0, iters: int = 1) -> np.ndarray:
    out = img.astype(np.float32, copy=True)
    pad = 2
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
                w = w_s * np.exp(-(diff**2) / (2.0 * (sigma_r**2))).astype(np.float32)
                num += w * neigh
                den += w
        out = num / np.maximum(den, 1e-8)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

# -------- Segmentation: label-mode & consensus --------
def _label_mode_filter(ids: np.ndarray, K: int, k: int = 5, iters: int = 1) -> np.ndarray:
    """one-hot → 近傍積算 → argmax（モード）で高速にラベル多数決．"""
    assert k in (3,5,7)
    out = ids.copy()
    H, W = out.shape
    for _ in range(max(1, iters)):
        one_hot = np.stack([(out == c).astype(np.uint8) for c in range(K)], axis=0)  # [K,H,W]
        pad = k // 2
        padded = np.pad(one_hot, ((0,0),(pad,pad),(pad,pad)), mode="edge")
        counts = np.zeros_like(one_hot, dtype=np.uint16)
        for dy in range(k):
            for dx in range(k):
                counts += padded[:, dy:dy+H, dx:dx+W]
        out = np.argmax(counts, axis=0).astype(out.dtype)
    return out

def _label_consensus(ids: np.ndarray, K: int, k: int = 5, min_frac: float = 0.6) -> np.ndarray:
    """現在のラベルが近傍内で十分優勢でない画素のみ，近傍モードに置換．"""
    H, W = ids.shape
    one_hot = np.stack([(ids == c).astype(np.uint8) for c in range(K)], axis=0)
    pad = k // 2
    padded = np.pad(one_hot, ((0,0),(pad,pad),(pad,pad)), mode="edge")
    counts = np.zeros_like(one_hot, dtype=np.uint16)
    for dy in range(k):
        for dx in range(k):
            counts += padded[:, dy:dy+H, dx:dx+W]
    mode_ids = np.argmax(counts, axis=0)
    mode_cnt = np.max(counts, axis=0)
    K2 = k * k
    # 現在ラベルの票数
    flat_counts = counts.transpose(1,2,0).reshape(-1, K)
    flat_ids = ids.reshape(-1).astype(np.int64)
    cur_cnt = flat_counts[np.arange(flat_ids.size), flat_ids].reshape(H, W)
    need_flip = (cur_cnt.astype(np.float32) < (min_frac * K2))
    out = ids.copy()
    out[need_flip] = mode_ids[need_flip].astype(out.dtype)
    return out

# -------- Serialize (TX) --------
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
        white_thresh = 250
        if app_cfg is not None and hasattr(app_cfg, "segdec"):
            # 白境界の事前処理は常時 ON（高速・副作用小）
            white_thresh = 250
        rgb = load_image_to_array(content_path, modality="segmentation", validate_mode=True)
        rgb = _suppress_white_boundaries(rgb, white_thresh=white_thresh, iters=2)

        h, w, _ = rgb.shape
        id_map, palette, enc = _build_seg_ids_and_palette(rgb, white_thresh=white_thresh)
        _set_seg_palette(palette)  # 送付せず，プロセス内共有

        # ★ご要望により，アプリ層での擬似ノイズ付与は廃止（以前の SEG_ID_NOISE_* は削除）

        if enc == "id8":
            id_bytes = id_map.reshape(-1).tobytes(); bits = 8
        else:
            id_bytes = id_map.astype('>u2').reshape(-1).tobytes(); bits = 16

        hdr = AppHeader(version=1, modality="segmentation", height=h, width=w, channels=1,
                        bits_per_sample=bits, payload_len_bytes=len(id_bytes))
        return hdr, id_bytes

    raise ValueError("Unknown modality")

# -------- Deserialize (RX) --------
def deserialize_content(hdr: AppHeader, payload_bytes: bytes, app_cfg: Optional[Any] = None,
                        text_encoding: str="utf-8", text_errors: str="replace") -> tuple[str, np.ndarray]:
    if hdr.modality == "text":
        s = text_bytes_to_string(payload_bytes, encoding=text_encoding, errors=text_errors)
        return s, np.array([], dtype=np.uint8)

    if hdr.modality == "edge":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        arr = (arr >= 128).astype(np.uint8) * 255
        if app_cfg is not None and hasattr(app_cfg, "edgedec"):
            ed = app_cfg.edgedec
            a01 = (arr > 0).astype(np.uint8)
            if ed.denoise == "maj3":
                a01 = _edge_binary_majority(a01, k=3, thr=ed.thresh, iters=ed.iters, preserve_lines=ed.preserve_lines)
            elif ed.denoise == "maj5":
                a01 = _edge_binary_majority(a01, k=5, thr=ed.thresh, iters=ed.iters, preserve_lines=ed.preserve_lines)
            elif ed.denoise == "median3":
                arr = _median_filter_uint8(arr, k=3, iters=ed.iters); a01 = (arr > 0).astype(np.uint8)
            elif ed.denoise == "median5":
                arr = _median_filter_uint8(arr, k=5, iters=ed.iters); a01 = (arr > 0).astype(np.uint8)
            elif ed.denoise == "open3":
                a01 = _binary_open(a01, k=3, iters=ed.iters, preserve_lines=ed.preserve_lines)
            elif ed.denoise == "open5":
                a01 = _binary_open(a01, k=5, iters=ed.iters, preserve_lines=ed.preserve_lines)
            elif ed.denoise == "open3close3":
                a01 = _binary_open(a01, k=3, iters=ed.iters, preserve_lines=ed.preserve_lines)
                a01 = _binary_close(a01, k=3, iters=1)
            elif ed.denoise == "open5close5":
                a01 = _binary_open(a01, k=5, iters=ed.iters, preserve_lines=ed.preserve_lines)
                a01 = _binary_close(a01, k=5, iters=1)
            arr = (a01 * 255).astype(np.uint8)
        return "", arr

    if hdr.modality == "depth":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width), np.uint8)
        if app_cfg is not None and hasattr(app_cfg, "depthdec"):
            dd = app_cfg.depthdec
            if dd.filt == "median3":
                arr = _median_filter_uint8(arr, k=3, iters=dd.iters)
            elif dd.filt == "median5":
                arr = _median_filter_uint8(arr, k=5, iters=dd.iters)
            elif dd.filt == "bilateral5":
                arr = _bilateral5_uint8(arr, sigma_s=dd.sigma_s, sigma_r=dd.sigma_r, iters=dd.iters)
            elif dd.filt == "median5_bilateral5":
                arr = _median_filter_uint8(arr, k=5, iters=dd.iters)
                arr = _bilateral5_uint8(arr, sigma_s=dd.sigma_s, sigma_r=dd.sigma_r, iters=1)
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

        # out-of-range fallback（既定 "uniform"）
        fallback = "uniform"; seed = 123
        if app_cfg is not None and hasattr(app_cfg, "segdec"):
            fallback = app_cfg.segdec.fallback
            seed = app_cfg.segdec.seed if app_cfg.segdec.seed is not None else 123
        if fallback == "uniform":
            invalid = (ids >= K)
            if invalid.any():
                rng = np.random.default_rng(seed)
                repl = rng.integers(0, K, size=int(invalid.sum()), dtype=np.int64)
                ids = ids.copy()
                ids[invalid] = (repl.astype(np.uint8) if ids.dtype == np.uint8 else repl.astype(np.uint16))
        elif fallback == "mod":
            ids = (ids % K).astype(ids.dtype, copy=False)
        else:  # clamp（旧実装）
            ids = np.minimum(ids, K - 1)

        # ラベル平滑化（edge ガイドは用いない）
        if app_cfg is not None and hasattr(app_cfg, "segdec"):
            sd = app_cfg.segdec
            if sd.mode != "none":
                if sd.mode == "majority3":
                    ids = _label_mode_filter(ids, K=K, k=3, iters=sd.iters)
                elif sd.mode == "majority5":
                    ids = _label_mode_filter(ids, K=K, k=5, iters=sd.iters)
                elif sd.mode == "strong":
                    ids = _label_mode_filter(ids, K=K, k=5, iters=max(1, sd.iters))
                    ids = _label_consensus(ids, K=K, k=5, min_frac=float(sd.consensus_min_frac))

        rgb = pal[np.minimum(ids, K - 1)]
        return "", rgb.astype(np.uint8)

    raise ValueError("Unknown modality in header")

# -------- Save --------
def save_output(hdr: AppHeader, text_str: str, img_arr: np.ndarray, out_path: str) -> None:
    if hdr.modality == "text":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_str); return
    if hdr.modality in ("edge","depth"):
        Image.fromarray(img_arr.astype(np.uint8), mode="L").save(out_path, format="PNG"); return
    if hdr.modality == "segmentation":
        Image.fromarray(img_arr.astype(np.uint8), mode="RGB").save(out_path, format="PNG"); return
    raise ValueError("Unsupported modality")
