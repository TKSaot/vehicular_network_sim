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
from common.backend import asnumpy

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
def save_output(hdr: AppHeader, text_str: str, img_arr, out_path: str) -> None:
    if hdr.modality == "text":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_str)
        return

    arr = asnumpy(img_arr)  # accept CuPy or NumPy and ensure NumPy for PIL
    if hdr.modality in ("edge","depth"):
        Image.fromarray(arr.astype(np.uint8), mode="L").save(out_path, format="PNG"); return
    if hdr.modality == "segmentation":
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(out_path, format="PNG"); return
    raise ValueError("Unsupported modality")


# channel/channel_model.py
from __future__ import annotations
from typing import Tuple
from common.backend import xp, np, asnumpy, to_xp, default_rng, is_cupy

__all__ = ["awgn_channel", "rayleigh_fading", "equalize"]

def awgn_channel(symbols, snr_db: float, seed: int | None = None):
    """
    Complex AWGN: SNR is Es/N0．noise variance per complex dimension is N0/2．
    GPU path (CuPy) if available, else CPU (NumPy).
    """
    z = to_xp(symbols).astype(xp.complex128, copy=False)
    Es = float(xp.mean(xp.abs(z)**2)) if z.size else 1.0
    snr_lin = 10.0**(snr_db/10.0)
    N0 = Es / snr_lin
    if is_cupy:
        rs = default_rng(seed)  # CuPy RandomState
        nr = rs.normal(0, xp.sqrt(N0/2.0), size=z.shape)
        ni = rs.normal(0, xp.sqrt(N0/2.0), size=z.shape)
        noise = (nr + 1j*ni).astype(xp.complex128)
    else:
        rng = default_rng(seed)
        nr = rng.normal(0, np.sqrt(N0/2.0), size=z.shape)
        ni = rng.normal(0, np.sqrt(N0/2.0), size=z.shape)
        noise = (nr + 1j*ni).astype(np.complex128)
        noise = to_xp(noise)
    return (z + noise).astype(xp.complex128)

def rayleigh_fading(symbols,
                    snr_db: float,
                    seed: int | None = None,
                    doppler_hz: float = 30.0,
                    symbol_rate: float = 1e6,
                    block_fading: bool = False,
                    snr_reference: str = "rx") -> Tuple["xp.ndarray", "np.ndarray"]:
    """
    Rayleigh flat fading + AWGN．戻り値は (受信シンボル[xp], フェージング h[n][np])．
    AR(1) fading is generated on CPU (sequential); the output symbols are returned on GPU if enabled.
    """
    x_np = asnumpy(symbols).astype(np.complex128, copy=False)
    N = x_np.size
    rng = default_rng(seed) if not is_cupy else np.random.default_rng(seed)  # ensure NumPy on CPU

    if N == 0:
        return to_xp(x_np), np.ones(0, dtype=np.complex128)

    # AR(1) correlation (CPU)
    Ts = 1.0 / max(1.0, float(symbol_rate))
    fd = abs(float(doppler_hz))
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = min(0.9999, max(0.0, float(rho)))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.ones(N, dtype=np.complex128) * h0
    else:
        v = (rng.normal(0, 1.0, size=N) + 1j*rng.normal(0, 1.0, size=N)) / np.sqrt(2.0)
        h = np.zeros(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    faded = x_np * h

    # Average SNR calibration (CPU)
    Es = float(np.mean(np.abs(x_np)**2)) if N else 1.0
    snr_lin = 10.0**(snr_db/10.0)
    gain = float(np.mean(np.abs(h)**2)) if str(snr_reference).lower() == "rx" else 1.0
    N0 = Es * gain / snr_lin
    nr = rng.normal(0, np.sqrt(N0/2.0), size=N)
    ni = rng.normal(0, np.sqrt(N0/2.0), size=N)
    noise = (nr + 1j*ni).astype(np.complex128)

    y = (faded + noise).astype(np.complex128)
    return to_xp(y), h

def equalize(rx_symbols, tx_pilot, rx_pilot):
    """
    1 タップ等化．h_hat = mean(rx_pilot / tx_pilot)．
    Runs on GPU if available．Returns (eq_symbols[xp], h_hat[python complex]).
    """
    rs = to_xp(rx_symbols).astype(xp.complex128, copy=False)
    tp = to_xp(tx_pilot).astype(xp.complex128, copy=False)
    rp = to_xp(rx_pilot).astype(xp.complex128, copy=False)

    mask = xp.abs(tp) > 1e-12
    if not bool(xp.any(mask)):
        return rs, complex(1.0 + 0.0j)

    h_hat = xp.mean(rp[mask] / tp[mask])
    eq = rs / (h_hat + 1e-12)
    # return python complex for logging
    try:
        h_hat_py = complex(asnumpy(h_hat).item())
    except Exception:
        h_hat_py = complex(float(xp.real(h_hat)), float(xp.imag(h_hat)))
    return eq.astype(xp.complex128), h_hat_py


# common/backend.py
from __future__ import annotations
import os

# Always have NumPy
import numpy as _np

# Try CuPy
try:
    import cupy as _cp  # type: ignore
except Exception:
    _cp = None

def _env_flag(name: str, default: str = "1") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

# Decide backend
_USE_CUDA = _env_flag("VC_USE_CUDA", "1") and (_cp is not None)

# Public aliases
is_cupy = bool(_USE_CUDA)
np = _np
cp = _cp
xp = _cp if is_cupy else _np

def asnumpy(a):
    """Return a NumPy array view/copy of 'a' (handles CuPy/NumPy)."""
    if is_cupy and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return _np.asarray(a)

def to_xp(a, dtype=None):
    """Convert arbitrary array-like to backend array."""
    if is_cupy:
        return _cp.asarray(a, dtype=dtype) if not isinstance(a, _cp.ndarray) else (a.astype(dtype) if dtype else a)
    return _np.asarray(a, dtype=dtype) if not isinstance(a, _np.ndarray) else (a.astype(dtype) if dtype else a)

def default_rng(seed=None):
    """
    Return a backend RNG object.
    - For NumPy: np.random.default_rng(seed)
    - For CuPy:  cp.random.RandomState(seed) (Generator is not guaranteed in all envs)
    """
    if is_cupy:
        # RandomState offers .normal/.randint etc. and is reproducible by seed.
        return _cp.random.RandomState(seed)  # type: ignore[attr-defined]
    return _np.random.default_rng(seed)


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


"""
Configuration dataclasses for the vehicular network simulation.
Now includes per-modality *decoder* options controlled from config.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

# ---------- Decoder options ----------
@dataclass
class SegDecoderConfig:
    # Out-of-range ID fallback at decoder
    fallback: Literal["uniform", "clamp", "mod"] = "uniform"
    # Post-filter strength
    # - "none": no smoothing
    # - "majority3": 3x3 モード（反復 iters）
    # - "majority5": 5x5 モード（反復 iters）
    # - "strong":   5x5 モード + 多数派合意（consensus，min_frac）
    mode: Literal["none", "majority3", "majority5", "strong"] = "strong"
    iters: int = 2
    # consensus：近傍内で同一ラベル率がこの閾値未満なら近傍モードに置換
    consensus_min_frac: float = 0.6
    # 乱数を使う処理の種（uniform 代替 ID など）
    seed: Optional[int] = 123

@dataclass
class EdgeDecoderConfig:
    # "none" | "maj3" | "maj5" | "median3" | "median5" |
    # "open3" | "open5" | "open3close3" | "open5close5"
    denoise: Literal["none","maj3","maj5","median3","median5","open3","open5","open3close3","open5close5"] = "open3close3"
    iters: int = 1
    # Majority のしきい値（未指定なら過半数）
    thresh: Optional[int] = None
    # 細線保護：1→0 で線が消えそうな画素は保持
    preserve_lines: bool = True

@dataclass
class DepthDecoderConfig:
    # "none" | "median3" | "median5" | "bilateral5" | "median5_bilateral5"
    filt: Literal["none","median3","median5","bilateral5","median5_bilateral5"] = "median5_bilateral5"
    iters: int = 1
    # bilateral(5x5)
    sigma_s: float = 1.6
    sigma_r: float = 12.0

# ---------- App / Link / PHY / Channel ----------
@dataclass
class AppConfig:
    modality: Literal["text", "edge", "depth", "segmentation"] = "text"
    validate_image_mode: bool = True
    text_encoding: str = "utf-8"
    text_errors: str = "replace"
    # 受信側デコーダ設定
    segdec: SegDecoderConfig = field(default_factory=SegDecoderConfig)
    edgedec: EdgeDecoderConfig = field(default_factory=EdgeDecoderConfig)
    depthdec: DepthDecoderConfig = field(default_factory=DepthDecoderConfig)

@dataclass
class LinkConfig:
    mtu_bytes: int = 1024
    interleaver_depth: int = 16
    # ★ 802.11p 風畳み込み符号のレート名も選択可能（既定は hamming74 のまま）
    fec_scheme: Literal[
        "none", "repeat", "hamming74", "rs255_223",
        "conv_k7_r12", "conv_k7_r23", "conv_k7_r34"
    ] = "hamming74"
    repeat_k: int = 3
    drop_bad_frames: bool = False

    strong_header_protection: bool = True
    header_copies: int = 7
    header_rep_k: int = 5
    force_output_on_hdr_fail: bool = True
    verbose: bool = False

    # 送信前のバイトマッピング（フレーム化前の拡散）
    byte_mapping_scheme: Literal["none", "permute", "frame_block"] = "none"
    byte_mapping_seed: Optional[int] = None  # None → chan.seed を使用

@dataclass
class ModulationConfig:
    scheme: Literal["bpsk", "qpsk", "16qam"] = "qpsk"

@dataclass
class ChannelConfig:
    channel_type: Literal["awgn", "rayleigh"] = "rayleigh"
    snr_db: float = 10.0
    seed: Optional[int] = 12345
    doppler_hz: float = 30.0
    symbol_rate: float = 1e6
    block_fading: bool = False
    # （必要なら）snr_reference を channel 側でオプション引数に渡す

@dataclass
class PilotConfig:
    preamble_len: int = 32
    pilot_len: int = 16
    pilot_every_n_symbols: int = 0

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
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig,
    SegDecoderConfig, EdgeDecoderConfig, DepthDecoderConfig
)

INPUTS = {
    "edge": "examples/edge_00001_.png",
    "depth": "examples/depth_00001_.png",
    "segmentation": "examples/segmentation_00001_.png",
    "text": "examples/sample.txt",   # ★ 追加（存在しない場合はランナー側で /mnt/data を自動フォールバック）
}
OUTPUT_ROOT = "outputs"
DEFAULT_MODALITY: Literal["edge","depth","segmentation"] = "segmentation"

def build_config(modality: Literal["edge","depth","segmentation"] = DEFAULT_MODALITY) -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(
            modality=modality,
            validate_image_mode=True,
            segdec=SegDecoderConfig(
                fallback="uniform",
                mode="strong",
                iters=2,
                consensus_min_frac=0.6,
                seed=123
            ),
            edgedec=EdgeDecoderConfig(
                denoise="open3close3",
                iters=1,
                thresh=None,
                preserve_lines=True
            ),
            depthdec=DepthDecoderConfig(
                filt="median5_bilateral5",
                iters=1,
                sigma_s=1.6,
                sigma_r=12.0
            ),
        ),
        link=LinkConfig(
            mtu_bytes=512,
            interleaver_depth=16,
            fec_scheme="hamming74",
            repeat_k=3,
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            byte_mapping_scheme="permute",
            byte_mapping_seed=None,
        ),
        mod=ModulationConfig(scheme="qpsk"),
        chan=ChannelConfig(
            channel_type="rayleigh",
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
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from common.utils import (
    bytes_to_bits, bits_to_bytes, append_crc32, verify_and_strip_crc32,
    block_interleave, block_deinterleave
)
from data_link_layer.error_correction import make_fec, RepetitionFEC

FRAME_TYPE_APP_HDR = 0x01
FRAME_TYPE_DATA    = 0x02
HEADER_LEN = 8  # bytes

# ---------- Headers ----------
@dataclass
class LinkHeader:
    version: int = 1
    frame_type: int = FRAME_TYPE_DATA
    seq_no: int = 0
    total_frames: int = 1
    payload_len: int = 0
    def to_bytes(self) -> bytes:
        return bytes([self.version & 0xFF, self.frame_type & 0xFF]) + \
               int(self.seq_no).to_bytes(2,'big') + \
               int(self.total_frames).to_bytes(2,'big') + \
               int(self.payload_len).to_bytes(2,'big')
    @staticmethod
    def from_bytes(b: bytes) -> "LinkHeader":
        if len(b) < HEADER_LEN:
            raise ValueError("Header too short")
        ver = b[0]; ftype = b[1]
        seq = int.from_bytes(b[2:4],'big')
        tot = int.from_bytes(b[4:6],'big')
        plen = int.from_bytes(b[6:8],'big')
        return LinkHeader(ver, ftype, seq, tot, plen)

@dataclass
class LinkFrame:
    header: LinkHeader
    payload_with_crc: bytes
    def to_bytes(self) -> bytes:
        return self.header.to_bytes() + self.payload_with_crc
    @staticmethod
    def from_bytes_safe(b: bytes) -> "LinkFrame":
        hdr = LinkHeader.from_bytes(b[:HEADER_LEN])
        return LinkFrame(hdr, b[HEADER_LEN:])

# ---------- Packetization ----------
def segment_message(app_header: bytes, data: bytes, mtu_bytes: int, header_copies: int = 1) -> List[LinkFrame]:
    frames: List[LinkFrame] = []
    for c in range(max(1, int(header_copies))):
        hdr0 = LinkHeader(frame_type=FRAME_TYPE_APP_HDR, seq_no=c, total_frames=0, payload_len=len(app_header))
        frames.append(LinkFrame(header=hdr0, payload_with_crc=append_crc32(app_header)))
    total_data_frames = (len(data) + mtu_bytes - 1) // mtu_bytes
    for i in range(total_data_frames):
        chunk = data[i*mtu_bytes:(i+1)*mtu_bytes]
        hdr = LinkHeader(frame_type=FRAME_TYPE_DATA, seq_no=header_copies + i, total_frames=0, payload_len=len(chunk))
        frames.append(LinkFrame(header=hdr, payload_with_crc=append_crc32(chunk)))
    total_frames = header_copies + total_data_frames
    for fr in frames:
        fr.header.total_frames = total_frames
    return frames

# ---------- FEC apply / reverse (ハード版：既存) ----------
def apply_fec_and_interleave(frames: List[LinkFrame], fec_scheme: str, repeat_k: int,
                             interleaver_depth: int, strong_header: bool,
                             header_copies: int, header_rep_k: int) -> Tuple[List[np.ndarray], List[int]]:
    enc_frames: List[np.ndarray] = []
    orig_bit_lengths: List[int] = []
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1,int(header_rep_k))) if strong_header else None
    for idx, fr in enumerate(frames):
        bits = bytes_to_bits(fr.to_bytes())
        orig_bit_lengths.append(len(bits))
        enc = fec.encode(bits)
        if strong_header and header_rep is not None and idx < header_copies:
            enc = header_rep.encode(enc)
        inter = block_interleave(enc, interleaver_depth)
        enc_frames.append(inter)
    return enc_frames, orig_bit_lengths

def reverse_fec_and_deinterleave(encoded_frames: List[np.ndarray], original_bit_lengths: List[int],
                                 fec_scheme: str, repeat_k: int, interleaver_depth: int,
                                 strong_header: bool, header_copies: int, header_rep_k: int) -> List[bytes]:
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1,int(header_rep_k))) if strong_header else None
    out: List[bytes] = []
    for idx, enc in enumerate(encoded_frames):
        deinter = block_deinterleave(enc, interleaver_depth)
        dec = deinter
        if strong_header and header_rep is not None and idx < header_copies:
            dec = header_rep.decode(dec)
        dec = fec.decode(dec)
        Lbits = original_bit_lengths[idx]
        if len(dec) > Lbits: dec = dec[:Lbits]
        b = bits_to_bytes(dec)
        out.append(b[: (Lbits + 7)//8])
    return out

# ---------- Hamming(7,4) 用 Chase 風ソフト復号 ----------
def _block_deinterleave_generic(arr: np.ndarray, depth: int, original_len: Optional[int] = None) -> np.ndarray:
    depth = max(1, int(depth))
    arr = np.asarray(arr).reshape(-1)
    if depth == 1:
        return arr[:original_len] if original_len is not None else arr
    L = len(arr); cols = int(np.ceil(L / depth))
    pad = depth*cols - L
    if pad:
        padv = 0.0 if np.issubdtype(arr.dtype, np.floating) else 0
        arr = np.concatenate([arr, np.full(pad, padv, dtype=arr.dtype)])
    mat = arr.reshape(cols, depth).T
    out = mat.reshape(-1)
    return out[:original_len] if original_len is not None else out

def _ham74_encode(d4: np.ndarray) -> np.ndarray:
    d1,d2,d3,d4b = [d4[i] & 1 for i in range(4)]
    p1 = d1 ^ d2 ^ d4b
    p2 = d1 ^ d3 ^ d4b
    p3 = d2 ^ d3 ^ d4b
    return np.array([d1,d2,d3,d4b,p1,p2,p3], dtype=np.uint8)

def _ham74_correct_one(c7: np.ndarray) -> np.ndarray:
    d1,d2,d3,d4,p1,p2,p3 = [c7[i] & 1 for i in range(7)]
    s1 = d1 ^ d2 ^ d4 ^ p1
    s2 = d1 ^ d3 ^ d4 ^ p2
    s3 = d2 ^ d3 ^ d4 ^ p3
    synd = (s1 + (s2<<1) + (s3<<2)) & 0x7
    pos_map = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)  # 既存実装に整合
    pos = int(pos_map[synd])
    if pos != 0: c7[pos-1] ^= 1
    return c7

def _ham74_decode_chase(bits: np.ndarray, rel: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    r = np.asarray(rel, dtype=np.float32).reshape(-1)
    L = (len(b)//7)*7
    b = b[:L]; r = r[:L]
    out = []
    for i in range(0, L, 7):
        c = b[i:i+7].copy(); w = r[i:i+7].copy()
        idx = np.argsort(w)  # 低信頼→高信頼
        cand_sets = [[], [idx[0]]]
        if len(idx) > 1: cand_sets += [[idx[1]], [idx[0], idx[1]]]
        if len(idx) > 2: cand_sets += [[idx[2]]]

        best_metric = 1e18; best_d4 = c[:4]
        for flips in cand_sets:
            c_try = c.copy()
            for f in flips: c_try[f] ^= 1
            c_fix = _ham74_correct_one(c_try.copy())
            d4 = c_fix[:4].copy()
            cref = _ham74_encode(d4)
            mism = (cref ^ c_try).astype(np.float32)
            metric = float(np.sum(mism * (1.0 + (1.0/(w+1e-6)))))
            if metric < best_metric:
                best_metric = metric; best_d4 = d4
        out.append(best_d4)
    return np.concatenate(out, axis=0).astype(np.uint8)

def reverse_fec_and_deinterleave_soft(encoded_frames_bits: List[np.ndarray],
                                      encoded_frames_rel: List[np.ndarray],
                                      original_bit_lengths: List[int],
                                      fec_scheme: str, repeat_k: int, interleaver_depth: int,
                                      strong_header: bool, header_copies: int, header_rep_k: int
                                      ) -> List[bytes]:
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1,int(header_rep_k))) if strong_header else None
    out: List[bytes] = []
    for idx, (enc_b, enc_r) in enumerate(zip(encoded_frames_bits, encoded_frames_rel)):
        de_b = block_deinterleave(enc_b, interleaver_depth)
        de_r = _block_deinterleave_generic(enc_r, interleaver_depth)
        if strong_header and header_rep is not None and idx < header_copies:
            de_b = header_rep.decode(de_b)  # 信頼度は未使用（ハード多数決）
        if fec_scheme.lower() == "hamming74":
            dec = _ham74_decode_chase(de_b, de_r)
        else:
            dec = fec.decode(de_b)
        Lbits = original_bit_lengths[idx]
        if len(dec) > Lbits: dec = dec[:Lbits]
        b = bits_to_bytes(dec)
        out.append(b[: (Lbits + 7)//8])
    return out

# ---------- Reassembly ----------
def _majority_bytes(blobs: List[bytes]) -> bytes:
    if not blobs: return b""
    min_len = min(len(x) for x in blobs)
    if min_len == 0: return b""
    arr_bits = [bytes_to_bits(x[:min_len]) for x in blobs]
    M = np.stack(arr_bits, axis=0); s = np.sum(M, axis=0)
    out_bits = (s >= (M.shape[0]//2 + 1)).astype(np.uint8)
    from common.utils import bits_to_bytes
    return bits_to_bytes(out_bits)[:min_len]

def reassemble_and_check(frames_bytes: List[bytes], header_copies: int = 1,
                         drop_bad: bool = False, verbose: bool = False):
    n_frames = len(frames_bytes)
    H = max(1,int(header_copies)); H = min(H, n_frames)
    hdr_ok = False; hdr_major = False; hdr_payload = b""
    header_payload_crc_blobs: List[bytes] = []
    for i in range(H):
        b = frames_bytes[i]; payload_crc = b[HEADER_LEN:]; header_payload_crc_blobs.append(payload_crc)
        ok, payload = verify_and_strip_crc32(payload_crc)
        if ok and not hdr_ok: hdr_ok = True; hdr_payload = payload
    if not hdr_ok:
        voted = _majority_bytes(header_payload_crc_blobs)
        ok, payload = verify_and_strip_crc32(voted)
        if ok: hdr_ok = True; hdr_major = True; hdr_payload = payload
    data = bytearray(); n_bad = 0
    for i in range(H, n_frames):
        b = frames_bytes[i]; payload_crc = b[HEADER_LEN:]
        ok, payload = verify_and_strip_crc32(payload_crc)
        if not ok:
            n_bad += 1
            if drop_bad: payload = b""
            if verbose: print(f"[WARN] DATA frame {i-H}/{n_frames-H} CRC failed")
        data.extend(payload)
    stats = {
        "all_crc_ok": bool(hdr_ok and (n_bad == 0)),
        "app_header_crc_ok": bool(hdr_ok),
        "app_header_recovered_via_majority": bool(hdr_major),
        "n_bad_frames": int(n_bad),
        "n_frames": int(n_frames),
    }
    return hdr_payload, bytes(data), stats


"""
Error correction schemes:
- None
- Repetition (k)
- Hamming(7,4)
- Reed-Solomon(255,223)  [optional]
- NEW: Convolutional K=7 (g0=133_o, g1=171_o) with puncturing: R=1/2, 2/3, 3/4

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
        synd_val = (s1 + (s2 << 1) + (s3 << 2)).astype(np.uint8)
        map_arr = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)  # pos map (1-based)
        err_pos = map_arr[synd_val]
        for i in range(C.shape[0]):
            pos = int(err_pos[i])
            if pos != 0:
                C[i, pos-1] ^= 1
        data = C[:, :4].reshape(-1)
        return data

# ----------------- Reed-Solomon(255,223) optional -----------------
class RS255223FEC(FECBase):
    """
    Byte-oriented RS(255,223) over GF(256). Requires 'reedsolo'.
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

# ----------------- NEW: Convolutional (K=7) with puncturing -----------------
class ConvK7FEC(FECBase):
    """
    Rate-1/2 mother code with K=7 (g0=133_o, g1=171_o) + puncturing:
      R=1/2: p0=[1],        p1=[1]
      R=2/3: p0=[1,1,0],    p1=[1,0,1]
      R=3/4: p0=[1,1,0,1],  p1=[1,0,1,1]
    Hard-decision Viterbi (64 states). Zero termination (m=6).
    """
    def __init__(self, rate: str = "1/2"):
        rate = str(rate).lower().replace("r","").replace("_","/")
        if rate in ("1/2", "12"):
            p0, p1 = [1], [1]; R = 1/2; self.name = "conv_k7_r12"
        elif rate in ("2/3", "23"):
            p0, p1 = [1,1,0], [1,0,1]; R = 2/3; self.name = "conv_k7_r23"
        elif rate in ("3/4", "34"):
            p0, p1 = [1,1,0,1], [1,0,1,1]; R = 3/4; self.name = "conv_k7_r34"
        else:
            raise ValueError(f"Unsupported conv. rate: {rate}")
        self.p0 = np.array(p0, dtype=np.uint8)
        self.p1 = np.array(p1, dtype=np.uint8)
        self.period = len(p0)
        self.code_rate = float(R)

        self.m = 6
        self.K = 7
        self.g0 = 0o133
        self.g1 = 0o171

        S = 1 << self.m
        self.next_state = np.zeros((S, 2), dtype=np.int32)
        self.out0 = np.zeros((S, 2), dtype=np.uint8)
        self.out1 = np.zeros((S, 2), dtype=np.uint8)
        for s in range(S):
            for u in (0, 1):
                reg = (u << self.m) | s
                y0 = bin(reg & self.g0).count("1") & 1
                y1 = bin(reg & self.g1).count("1") & 1
                ns = (s >> 1) | (u << (self.m - 1))
                self.next_state[s, u] = ns
                self.out0[s, u] = y0
                self.out1[s, u] = y1

    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        s = 0
        y0_list, y1_list = [], []
        for u in b:
            y0 = int(self.out0[s, int(u)]); y1 = int(self.out1[s, int(u)])
            y0_list.append(y0); y1_list.append(y1)
            s = int(self.next_state[s, int(u)])
        for _ in range(self.m):  # zero-termination
            y0 = int(self.out0[s, 0]); y1 = int(self.out1[s, 0])
            y0_list.append(y0); y1_list.append(y1)
            s = int(self.next_state[s, 0])

        out = []
        P = self.period
        for t in range(len(y0_list)):
            if self.p0[t % P]: out.append(y0_list[t])
            if self.p1[t % P]: out.append(y1_list[t])
        return np.array(out, dtype=np.uint8)

    def _depuncture_to_pairs(self, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(bits, dtype=np.uint8).reshape(-1)
        obs0, obs1 = [], []
        i = 0; P = self.period
        while i < len(r):
            if self.p0[len(obs0) % P]:
                b0 = int(r[i]) if i < len(r) else -1; i += 1
            else:
                b0 = -1
            if self.p1[len(obs1) % P]:
                b1 = int(r[i]) if i < len(r) else -1; i += 1
            else:
                b1 = -1
            obs0.append(b0); obs1.append(b1)
        return np.array(obs0, dtype=np.int16), np.array(obs1, dtype=np.int16)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        obs0, obs1 = self._depuncture_to_pairs(bits)
        T = int(len(obs0)); S = 1 << self.m; INF = 10**9
        pm = np.full(S, INF, dtype=np.int64); pm[0] = 0
        prev_state = np.full((T, S), -1, dtype=np.int16)
        prev_bit   = np.zeros((T, S), dtype=np.uint8)

        for t in range(T):
            new_pm = np.full(S, INF, dtype=np.int64)
            o0, o1 = int(obs0[t]), int(obs1[t])
            for ps in range(S):
                m0 = pm[ps]
                if m0 >= INF: continue
                for u in (0,1):
                    ns = int(self.next_state[ps, u])
                    y0 = int(self.out0[ps, u]); y1 = int(self.out1[ps, u])
                    dist = 0
                    if o0 != -1: dist += (o0 ^ y0)
                    if o1 != -1: dist += (o1 ^ y1)
                    cost = m0 + dist
                    if cost < new_pm[ns]:
                        new_pm[ns] = cost
                        prev_state[t, ns] = ps
                        prev_bit[t, ns] = u
            pm = new_pm

        end_state = 0 if pm[0] < INF else int(np.argmin(pm))
        s = end_state
        bits_rev = []
        for t in range(T - 1, -1, -1):
            u = int(prev_bit[t, s])
            bits_rev.append(u)
            s = int(prev_state[t, s])
            if s < 0: s = 0
        seq = bits_rev[::-1]
        if len(seq) >= self.m:
            seq = seq[:len(seq) - self.m]
        return np.array(seq, dtype=np.uint8)

# ----------------- Factory -----------------
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
    if s in ("conv_k7_r12", "convk7_r12", "conv_k7_1_2"):
        return ConvK7FEC(rate="1/2")
    if s in ("conv_k7_r23", "convk7_r23", "conv_k7_2_3"):
        return ConvK7FEC(rate="2/3")
    if s in ("conv_k7_r34", "convk7_r34", "conv_k7_3_4"):
        return ConvK7FEC(rate="3/4")
    raise ValueError(f"Unknown FEC scheme: {scheme}")


# examples/simulate_image_transmission.py
"""
設定は configs/image_config.py に集約．
例:
  python examples/simulate_image_transmission.py --modality segmentation --channel rayleigh --snr_db 10
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

    # --- TX ---
    tx_hdr, payload = serialize_content(
        modality, input_path,
        app_cfg=cfg.app,
        text_encoding="utf-8",
        validate_image_mode=True
    )
    app_hdr_bytes = tx_hdr.to_bytes()

    tx_syms, tx_meta = build_transmission(app_hdr_bytes, payload, cfg)

    # --- Channel ---
    if cfg.chan.channel_type == "awgn":
        rx_syms = awgn_channel(tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading
        )

    # --- RX ---
    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    hdr_used_mode = "valid"
    if not stats.get("app_header_crc_ok", False):
        if stats.get("app_header_recovered_via_majority", False):
            hdr_used_mode = "majority"
        elif cfg.link.force_output_on_hdr_fail:
            rx_app_hdr_b = tx_hdr.to_bytes()
            hdr_used_mode = "forced"
        else:
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

    _, img_arr = deserialize_content(rx_hdr, rx_payload_b, app_cfg=cfg.app, text_encoding="utf-8")

    out_dir = make_output_dir(cfg, modality=modality, input_path=input_path, output_root=output_root)
    out_png = os.path.join(out_dir, "received.png")
    save_output(rx_hdr, "", img_arr, out_png)

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

    tx_hdr, payload = serialize_content("text", input_path, app_cfg=cfg.app, text_encoding="utf-8", validate_image_mode=False)
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

    hdr_used_mode = "valid"
    if not stats.get("app_header_crc_ok", False):
        if stats.get("app_header_recovered_via_majority", False):
            hdr_used_mode = "majority"
        elif cfg.link.force_output_on_hdr_fail:
            rx_app_hdr_b = tx_hdr.to_bytes()
            hdr_used_mode = "forced"
        else:
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

    text_str, _ = deserialize_content(rx_hdr, rx_payload_b, app_cfg=cfg.app, text_encoding="utf-8", text_errors="replace")

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
# GPU/CPU backend 両対応版：LLR（信頼度）を返すデモジュレーションを追加
from __future__ import annotations
from typing import Tuple
from common.backend import xp, np, to_xp, asnumpy

# ---------- BPSK ----------
def _bpsk_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    return (2.0 * b.astype(xp.float64) - 1.0).astype(xp.complex128)

def _bpsk_demod_with_rel(symbols) -> Tuple["np.ndarray", "np.ndarray"]:
    z = to_xp(symbols).astype(xp.complex128, copy=False).real
    bits = (z >= 0).astype(xp.uint8)
    rel = xp.abs(z).astype(xp.float32)
    return asnumpy(bits), asnumpy(rel)

def _bpsk_demod(symbols) -> "np.ndarray":
    bits, _ = _bpsk_demod_with_rel(symbols)
    return bits

# ---------- QPSK(Gray) ----------
def _qpsk_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    if (b.size % 2) != 0:
        b = xp.concatenate([b, xp.zeros(1, dtype=xp.uint8)])
    pairs = b.reshape(-1, 2).astype(xp.float64)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]
    return ((i + 1j*q) / xp.sqrt(2.0)).astype(xp.complex128)

def _qpsk_demod_with_rel(symbols) -> Tuple["np.ndarray", "np.ndarray"]:
    z = to_xp(symbols).astype(xp.complex128, copy=False)
    re = z.real.astype(xp.float64, copy=False)
    im = z.imag.astype(xp.float64, copy=False)
    b0 = (re < 0).astype(xp.uint8)  # I
    b1 = (im < 0).astype(xp.uint8)  # Q
    rel0 = xp.abs(re)
    rel1 = xp.abs(im)
    bits = xp.vstack([b0, b1]).T.reshape(-1)
    rel  = xp.vstack([rel0, rel1]).T.reshape(-1).astype(xp.float32, copy=False)
    return asnumpy(bits), asnumpy(rel)

def _qpsk_demod(symbols) -> "np.ndarray":
    bits, _ = _qpsk_demod_with_rel(symbols)
    return bits

# ---------- 16QAM(Gray) ----------
def _16qam_mod(bits) -> "xp.ndarray":
    b = to_xp(bits, dtype=xp.uint8).reshape(-1)
    pad = (-b.size) % 4
    if pad:
        b = xp.concatenate([b, xp.zeros(pad, dtype=xp.uint8)])
    quads = b.reshape(-1, 4).astype(xp.float64)
    I = (1.0 - 2.0 * quads[:, 0]) * (3.0 - 2.0 * quads[:, 1])
    Q = (1.0 - 2.0 * quads[:, 2]) * (3.0 - 2.0 * quads[:, 3])
    return ((I + 1j*Q) / xp.sqrt(10.0)).astype(xp.complex128)

def _16qam_demod_with_rel(symbols) -> Tuple["np.ndarray", "np.ndarray"]:
    z = to_xp(symbols).astype(xp.complex128, copy=False) * xp.sqrt(10.0)
    I = z.real.astype(xp.float64, copy=False)
    Q = z.imag.astype(xp.float64, copy=False)

    b_i1 = (I < 0).astype(xp.uint8)
    b_i0 = (xp.abs(I) < 2.0).astype(xp.uint8)
    b_q1 = (Q < 0).astype(xp.uint8)
    b_q0 = (xp.abs(Q) < 2.0).astype(xp.uint8)

    r_i1 = xp.abs(I)
    r_i0 = xp.abs(xp.abs(I) - 2.0)
    r_q1 = xp.abs(Q)
    r_q0 = xp.abs(xp.abs(Q) - 2.0)

    bits = xp.vstack([b_i1, b_i0, b_q1, b_q0]).T.reshape(-1)
    rel  = xp.vstack([r_i1, r_i0, r_q1, r_q0]).T.reshape(-1).astype(xp.float32, copy=False)
    return asnumpy(bits), asnumpy(rel)

def _16qam_demod(symbols) -> "np.ndarray":
    bits, _ = _16qam_demod_with_rel(symbols)
    return bits

# ---------- Modulator wrapper ----------
class Modulator:
    def __init__(self, scheme: str = "qpsk"):
        s = scheme.lower()
        if s not in ("bpsk", "qpsk", "16qam"):
            raise ValueError("Unsupported modulation")
        self.scheme = s

    @property
    def bits_per_symbol(self) -> int:
        return {"bpsk": 1, "qpsk": 2, "16qam": 4}[self.scheme]

    def modulate(self, bits) -> "xp.ndarray":
        if self.scheme == "bpsk":  return _bpsk_mod(bits)
        if self.scheme == "qpsk":  return _qpsk_mod(bits)
        if self.scheme == "16qam": return _16qam_mod(bits)
        raise RuntimeError

    def demodulate(self, symbols) -> "np.ndarray":
        if self.scheme == "bpsk":  return _bpsk_demod(symbols)
        if self.scheme == "qpsk":  return _qpsk_demod(symbols)
        if self.scheme == "16qam": return _16qam_demod(symbols)
        raise RuntimeError

    def demodulate_with_reliability(self, symbols) -> Tuple["np.ndarray","np.ndarray"]:
        if self.scheme == "bpsk":  return _bpsk_demod_with_rel(symbols)
        if self.scheme == "qpsk":  return _qpsk_demod_with_rel(symbols)
        if self.scheme == "16qam": return _16qam_demod_with_rel(symbols)
        raise RuntimeError

# ---------- PHY frame builder ----------
def build_phy_frame(bits, mod: Modulator,
                    preamble_len_bits: int = 32,
                    pilot_len_symbols: int = 16
                    ) -> tuple["xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    # preamble: 1010...(BPSK)
    base = xp.array([1,0], dtype=xp.uint8)
    pre_bits = xp.tile(base, preamble_len_bits // 2)
    if (preamble_len_bits % 2) == 1:
        pre_bits = xp.concatenate([pre_bits, xp.array([1], dtype=xp.uint8)])
    pre_sym = _bpsk_mod(pre_bits).astype(xp.complex128)

    pilot = xp.ones(pilot_len_symbols, dtype=xp.complex128) * (1.0 + 1j) / xp.sqrt(2.0)
    data_syms = mod.modulate(bits).astype(xp.complex128)
    all_syms = xp.concatenate([pre_sym, pilot, data_syms])
    return all_syms, pilot, data_syms


# receiver/receive.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from common.backend import to_xp
from common.config import SimulationConfig
from data_link_layer.encoding import (
    reverse_fec_and_deinterleave,
    reverse_fec_and_deinterleave_soft,
    reassemble_and_check,
)
from physical_layer.modulation import Modulator
from channel.channel_model import equalize

def recover_from_symbols(rx_symbols, tx_meta: Dict, cfg: SimulationConfig) -> Tuple[bytes, bytes, dict]:
    mod = Modulator(tx_meta["mod_scheme"])
    frames_bits: List[np.ndarray] = []
    frames_rel:  List[np.ndarray] = []
    est_channels: List[complex] = []

    z = to_xp(rx_symbols)  # GPU/CPU どちらでも OK

    for i, (start, end) in enumerate(tx_meta["frame_symbol_ranges"]):
        syms = z[start:end]
        pilot_len = len(tx_meta["pilots_tx"][i])
        preamble_len = cfg.pilot.preamble_len
        rx_pilot = syms[preamble_len:preamble_len+pilot_len]
        rx_data  = syms[preamble_len+pilot_len:]

        eq_data, h_hat = equalize(rx_data, tx_meta["pilots_tx"][i], rx_pilot)
        est_channels.append(h_hat)

        bits, rel = mod.demodulate_with_reliability(eq_data)  # ← LLR 近似
        frames_bits.append(bits.astype(np.uint8))
        frames_rel.append(rel.astype(np.float32))

    if cfg.link.fec_scheme.lower() == "hamming74":
        raw_frame_bytes = reverse_fec_and_deinterleave_soft(
            frames_bits, frames_rel,
            tx_meta["orig_bit_lengths"],
            fec_scheme=cfg.link.fec_scheme,
            repeat_k=cfg.link.repeat_k,
            interleaver_depth=cfg.link.interleaver_depth,
            strong_header=cfg.link.strong_header_protection,
            header_copies=cfg.link.header_copies,
            header_rep_k=cfg.link.header_rep_k
        )
    else:
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

    app_hdr_bytes, payload_bytes, stats = reassemble_and_check(
        raw_frame_bytes,
        header_copies=cfg.link.header_copies,
        drop_bad=cfg.link.drop_bad_frames,
        verbose=cfg.link.verbose
    )
    stats["h_estimates"] = est_channels
    return app_hdr_bytes, payload_bytes, stats


# transmitter/send.py
from __future__ import annotations
from typing import List, Tuple

from common.backend import xp, to_xp
from common.config import SimulationConfig
from common.byte_mapping import map_bytes
from data_link_layer.encoding import segment_message, apply_fec_and_interleave
from physical_layer.modulation import Modulator, build_phy_frame

def build_transmission(app_header_bytes: bytes, payload_bytes: bytes, cfg: SimulationConfig):
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    mapped_payload = map_bytes(
        payload_bytes, mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme, seed=mapping_seed
    )

    frames = segment_message(
        app_header_bytes, mapped_payload,
        mtu_bytes=cfg.link.mtu_bytes, header_copies=cfg.link.header_copies
    )

    enc_bits_list, orig_bit_lengths = apply_fec_and_interleave(
        frames, fec_scheme=cfg.link.fec_scheme, repeat_k=cfg.link.repeat_k,
        interleaver_depth=cfg.link.interleaver_depth,
        strong_header=cfg.link.strong_header_protection,
        header_copies=cfg.link.header_copies,
        header_rep_k=cfg.link.header_rep_k
    )

    mod = Modulator(cfg.mod.scheme)
    frame_symbol_ranges: List[Tuple[int,int]] = []
    pilots_tx: List = []
    data_symbol_counts: List[int] = []
    all_syms: List = []
    cursor = 0
    for bits in enc_bits_list:
        syms, pilot, data_syms = build_phy_frame(
            bits, mod,
            preamble_len_bits=cfg.pilot.preamble_len,
            pilot_len_symbols=cfg.pilot.pilot_len
        )
        # ここで backend に統一
        syms = to_xp(syms, dtype=xp.complex128)
        pilot = to_xp(pilot, dtype=xp.complex128)
        data_syms = to_xp(data_syms, dtype=xp.complex128)

        all_syms.append(syms)
        pilots_tx.append(pilot)
        data_symbol_counts.append(int(data_syms.size))
        L = int(syms.size)
        frame_symbol_ranges.append((cursor, cursor + L))
        cursor += L

    tx_symbols = xp.concatenate(all_syms) if all_syms else xp.zeros(0, dtype=xp.complex128)
    tx_meta = {
        "mod_scheme": cfg.mod.scheme,
        "frame_symbol_ranges": frame_symbol_ranges,
        "data_symbol_counts": data_symbol_counts,
        "pilots_tx": pilots_tx,
        "orig_bit_lengths": orig_bit_lengths,
    }
    return tx_symbols, tx_meta
