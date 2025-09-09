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


# channel/channel_model.py
"""
Channel models: AWGN and Rayleigh fading with optional Doppler correlation.
We operate on complex baseband symbols.
"""

from __future__ import annotations
import numpy as np

__all__ = ["awgn_channel", "rayleigh_fading", "equalize"]

def awgn_channel(symbols: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """
    Complex AWGN: SNR is Es/N0．noise variance per complex dimension is N0/2．
    """
    rng = np.random.default_rng(seed)
    Es = np.mean(np.abs(symbols)**2) if symbols.size else 1.0
    snr_lin = 10**(snr_db/10.0)
    N0 = Es / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=symbols.shape) +
             1j * rng.normal(0, np.sqrt(N0/2), size=symbols.shape))
    return symbols + noise

def rayleigh_fading(symbols: np.ndarray,
                    snr_db: float,
                    seed: int | None = None,
                    doppler_hz: float = 30.0,
                    symbol_rate: float = 1e6,
                    block_fading: bool = False,
                    snr_reference: str = "rx") -> tuple[np.ndarray, np.ndarray]:
    """
    Rayleigh flat fading + AWGN．戻り値は (受信シンボル, フェージング h[n]) ．
    snr_reference:
      "rx" … N0 を平均受信電力 Es*E[|h|^2] に対して設定（既定）．
      "tx" … 送信 Es に対して設定．
    """
    rng = np.random.default_rng(seed)
    N = len(symbols)
    if N == 0:
        return symbols.copy(), np.ones(0, dtype=np.complex128)

    # 時間相関（AR(1) 近似）
    Ts = 1.0 / max(1.0, float(symbol_rate))
    fd = abs(float(doppler_hz))
    rho = np.exp(-0.5 * (2*np.pi*fd*Ts)**2)
    rho = min(0.9999, max(0.0, rho))

    if block_fading:
        h0 = (rng.normal(0, np.sqrt(0.5)) + 1j*rng.normal(0, np.sqrt(0.5)))
        h = np.ones(N, dtype=np.complex128) * h0
    else:
        v = (rng.normal(0, 1.0, size=N) + 1j*rng.normal(0, 1.0, size=N)) / np.sqrt(2.0)  # CN(0,1)
        h = np.zeros(N, dtype=np.complex128)
        h[0] = v[0]
        for n in range(1, N):
            h[n] = rho*h[n-1] + np.sqrt(1 - rho**2)*v[n]

    faded = symbols * h

    # 平均 SNR 較正
    Es = np.mean(np.abs(symbols)**2) if N else 1.0
    snr_lin = 10**(snr_db/10.0)
    gain = float(np.mean(np.abs(h)**2)) if str(snr_reference).lower() == "rx" else 1.0
    N0 = Es * gain / snr_lin
    noise = (rng.normal(0, np.sqrt(N0/2), size=N) + 1j*rng.normal(0, np.sqrt(N0/2), size=N))

    return faded + noise, h

def equalize(rx_symbols: np.ndarray, tx_pilot: np.ndarray, rx_pilot: np.ndarray) -> tuple[np.ndarray, complex]:
    """
    1 タップ等化．h_hat = mean(rx_pilot / tx_pilot)．
    """
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


# examples/run_uep_experiment.py
"""
Run EEP vs UEP experiments with equal airtime (equal total symbols) across modalities.

- Scenarios:
    * eep        : Equal Error Protection (uniform MCS for all modalities)
    * uep_edge   : Edge-prioritized protection
    * uep_depth  : Depth-prioritized protection
    * all        : Run all three above

- SNR sweep: default "1,4,8,12"
- Channel:  rayleigh (default) or awgn
- Outputs:
    outputs/experiments/<scenario>/snr_<XdB>/<modality>/(received.png|received.txt, rx_stats.json)
    outputs/experiments/<scenario>/summary_<scenario>.csv (append rows per SNR)

Notes:
- Keeps the #2 baseline intact. Only orchestrates per-modality configs from outside.
- If a requested FEC scheme is not supported in your local tree, it falls back to "hamming74".
- Equal airtime is matched to EEP's total symbol count within a default tolerance of 1%.
"""

from __future__ import annotations
import os, sys, csv, json, argparse
import copy
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project utilities (already present in your tree)
from common.config import SimulationConfig
from common.utils import set_seed, psnr
from common.run_utils import write_json
from common.byte_mapping import unmap_bytes

from app_layer.application import (
    serialize_content, AppHeader, deserialize_content, save_output
)
from transmitter.send import build_transmission
from channel.channel_model import awgn_channel, rayleigh_fading
from receiver.receive import recover_from_symbols

# Default inputs (we add 'text' path here via image_config)
from configs import image_config as CFG

# ---------------------- Helpers ----------------------

MODALITIES = ["edge", "depth", "segmentation", "text"]
BPS = {"bpsk": 1, "qpsk": 2, "16qam": 4}

class MCS:
    def __init__(self, mod: str = "qpsk", fec: str = "hamming74", ilv: int = 16):
        self.mod = mod
        self.fec = fec
        self.ilv = ilv
    def copy(self): return MCS(self.mod, self.fec, self.ilv)
    def as_dict(self): return {"mod": self.mod, "fec": self.fec, "interleaver_depth": self.ilv}

def _ensure_input_path(modality: str, path: str | None) -> str:
    if path is not None and os.path.isfile(path):
        return path
    if modality in CFG.INPUTS and os.path.isfile(CFG.INPUTS[modality]):
        return CFG.INPUTS[modality]
    # final fallbacks for robustness
    fallback = {
        "text": "examples/sample.txt",
        "edge": "examples/edge_00001_.png",
        "depth": "examples/depth_00001_.png",
        "segmentation": "examples/segmentation_00001_.png",
    }[modality]
    if os.path.isfile(fallback):
        return fallback
    # allow /mnt/data paths if user placed samples there
    mnt = {
        "text": "/mnt/data/sample.txt",
        "edge": "/mnt/data/edge_aachen_72_00001_.png",
        "depth": "/mnt/data/depth_aachen_72_00001_.png",
        "segmentation": "/mnt/data/segmentation_aachen_72_00001_.png",
    }[modality]
    if os.path.isfile(mnt):
        return mnt
    raise FileNotFoundError(f"No input found for modality={modality}")

def _apply_mcs_to_cfg(cfg: SimulationConfig, mcs: MCS, drop_bad_frames: bool) -> None:
    cfg.mod.scheme = mcs.mod
    cfg.link.fec_scheme = mcs.fec
    cfg.link.interleaver_depth = int(mcs.ilv)
    cfg.link.drop_bad_frames = bool(drop_bad_frames)
    # keep mapping & header protection as in your tree
    if cfg.link.byte_mapping_seed is None:
        cfg.link.byte_mapping_seed = cfg.chan.seed

def _estimate_symbols_for_modality(modality: str, input_path: str,
                                   cfg: SimulationConfig) -> tuple[int, AppHeader, bytes]:
    """
    Build a transmission once and return exact symbol length, header bytes.
    """
    # Serialize
    hdr, payload = serialize_content(modality, input_path, app_cfg=cfg.app,
                                     text_encoding="utf-8", validate_image_mode=True)
    hdr_bytes = hdr.to_bytes()
    # Try building; if FEC unsupported, fallback to hamming74
    try:
        tx_syms, _ = build_transmission(hdr_bytes, payload, cfg)
    except Exception as e:
        # FEC unknown or not implemented in local tree
        if "Unknown FEC scheme" in str(e) or "rs255_223" in str(e) or "conv_k7" in str(e):
            old = cfg.link.fec_scheme
            cfg.link.fec_scheme = "hamming74"
            tx_syms, _ = build_transmission(hdr_bytes, payload, cfg)
            print(f"[WARN] FEC '{old}' unsupported here. Fell back to 'hamming74' for {modality}.")
        else:
            raise
    return int(len(tx_syms)), hdr, payload

def _run_single_transmission(modality: str, input_path: str, cfg: SimulationConfig,
                             channel: str, snr_db: float, out_dir: str,
                             prebuilt_hdr: AppHeader | None = None,
                             prebuilt_payload: bytes | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Prepare header & payload
    if prebuilt_hdr is None or prebuilt_payload is None:
        hdr, payload = serialize_content(modality, input_path, app_cfg=cfg.app,
                                         text_encoding="utf-8", validate_image_mode=True)
    else:
        hdr, payload = prebuilt_hdr, prebuilt_payload
    hdr_bytes = hdr.to_bytes()

    # TX
    tx_syms, tx_meta = build_transmission(hdr_bytes, payload, cfg)

    # Channel
    if channel == "awgn":
        rx_syms = awgn_channel(tx_syms, snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading,  # snr_reference="rx" by default in your channel model
        )

    # RX
    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    # Try parse header; if failed and force_output_on_hdr_fail, fall back to TX hdr
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = hdr

    # Byte unmapping
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes,
    )

    # Deserialize & Save
    text_str, img_arr = deserialize_content(rx_hdr, rx_payload_b, app_cfg=cfg.app, text_encoding="utf-8")
    if modality == "text":
        out_path = os.path.join(out_dir, "received.txt")
        save_output(rx_hdr, text_str, img_arr, out_path)
        psnr_db = None
    else:
        out_path = os.path.join(out_dir, "received.png")
        save_output(rx_hdr, text_str, img_arr, out_path)
        # PSNR quick check (for your internal reference; ComfyUI will handle FID/LPIPS)
        try:
            im_orig = Image.open(input_path)
            if modality in ("edge", "depth"): im_orig = im_orig.convert("L")
            elif modality == "segmentation": im_orig = im_orig.convert("RGB")
            arr_orig = np.array(im_orig)
            if arr_orig.shape != img_arr.shape:
                arr_orig = arr_orig[:img_arr.shape[0], :img_arr.shape[1], ...]
            psnr_db = float(psnr(arr_orig.astype(np.uint8), img_arr.astype(np.uint8), data_range=255))
        except Exception:
            psnr_db = None

    # Write stats json with extra fields
    rec = {
        "modality": modality,
        "fec": cfg.link.fec_scheme,
        "modulation": cfg.mod.scheme,
        "interleaver_depth": cfg.link.interleaver_depth,
        "snr_db": snr_db,
        "channel": channel,
        "output_path": out_path,
        "psnr_db": psnr_db,
    }
    # merge rx stats
    for k, v in stats.items():
        if isinstance(v, (np.bool_, bool)):
            rec[k] = bool(v)
        elif isinstance(v, (np.integer, int)):
            rec[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            rec[k] = float(v)
        else:
            try:
                rec[k] = int(v)
            except Exception:
                pass

    write_json(os.path.join(out_dir, "rx_stats.json"), rec)
    return rec

# ---------------------- Scenario & Scheduler ----------------------

def _baseline_eep_mcs() -> dict:
    # 強化EEP: QPSK + Conv(K=7) R=2/3 + interleaver 32
    return {m: MCS(mod="qpsk", fec="conv_k7_r23", ilv=32) for m in MODALITIES}


def _uep_edge_initial() -> dict:
    return {
        "edge": MCS(mod="bpsk", fec="conv_k7_r12", ilv=32),
        "depth": MCS(mod="qpsk", fec="conv_k7_r23", ilv=24),
        "segmentation": MCS(mod="qpsk", fec="conv_k7_r34", ilv=16),
        "text": MCS(mod="qpsk", fec="conv_k7_r12", ilv=32),
    }

def _uep_depth_initial() -> dict:
    return {
        "depth": MCS(mod="bpsk", fec="conv_k7_r12", ilv=32),
        "edge": MCS(mod="qpsk", fec="conv_k7_r23", ilv=24),
        "segmentation": MCS(mod="qpsk", fec="conv_k7_r34", ilv=16),
        "text": MCS(mod="qpsk", fec="conv_k7_r12", ilv=32),
    }

def _next_lighter(mcs: MCS) -> MCS | None:
    # lighten = consume fewer symbols (higher throughput): try FEC 2/3->3/4, 1/2->2/3->3/4; then modulation BPSK->QPSK->16QAM
    order_fec = ["conv_k7_r12", "conv_k7_r23", "conv_k7_r34", "hamming74"]  # treat hamming ~ light-ish
    order_mod = ["bpsk", "qpsk", "16qam"]
    m = mcs.copy()
    # fec step up
    if m.fec in order_fec:
        idx = order_fec.index(m.fec)
        if idx + 1 < len(order_fec):
            m.fec = order_fec[idx + 1]; return m
    # modulation step up
    if m.mod in order_mod:
        im = order_mod.index(m.mod)
        if im + 1 < len(order_mod):
            m.mod = order_mod[im + 1]; return m
    return None

def _next_stronger(mcs: MCS) -> MCS | None:
    # stronger = more redundancy (more symbols): 3/4->2/3->1/2; 16QAM->QPSK->BPSK
    order_fec = ["hamming74", "conv_k7_r34", "conv_k7_r23", "conv_k7_r12"]
    order_mod = ["16qam", "qpsk", "bpsk"]
    m = mcs.copy()
    if m.fec in order_fec:
        idx = order_fec.index(m.fec)
        if idx + 1 < len(order_fec):
            m.fec = order_fec[idx + 1]; return m
    if m.mod in order_mod:
        im = order_mod.index(m.mod)
        if im + 1 < len(order_mod):
            m.mod = order_mod[im + 1]; return m
    return None

def _build_cfg(modality: str, mcs: MCS, chan_type: str, snr_db: float, seed: int,
               drop_bad_frames: bool) -> SimulationConfig:
    cfg: SimulationConfig = CFG.build_config(modality=modality)
    cfg.chan.channel_type = chan_type
    cfg.chan.snr_db = float(snr_db)
    cfg.chan.seed = int(seed)
    _apply_mcs_to_cfg(cfg, mcs, drop_bad_frames=drop_bad_frames)
    return cfg

def _equal_airtime_adjust(target_total_syms: int,
                          init_mcs: dict,
                          inputs: dict,
                          chan_type: str,
                          snr_db: float,
                          seed: int,
                          drop_bad_frames: bool,
                          priority: str) -> tuple[dict, dict]:
    """
    Adjust non-priority modalities' MCS so that sum(symbols) ~= target_total_syms (±1%).
    Returns (final_mcs_dict, prebuilt_cache) where prebuilt_cache carries (hdr, payload, sym_len) for reuse.
    """
    # Prebuild headers/payloads once per modality (independent of MCS)
    prebuilt = {}
    # We'll estimate symbols per modality for current MCS by actually building transmissions.
    def estimate_for_mod(m, mcs_obj):
        cfg = _build_cfg(m, mcs_obj, chan_type, snr_db, seed, drop_bad_frames)
        if m in prebuilt and prebuilt[m].get("payload") is not None and prebuilt[m].get("hdr") is not None:
            # reuse payload if available
            hdr = prebuilt[m]["hdr"]; payload = prebuilt[m]["payload"]
            hdr_b = hdr.to_bytes()
            try:
                tx_syms, _ = build_transmission(hdr_b, payload, cfg)
            except Exception as e:
                if "Unknown FEC scheme" in str(e) or "rs255_223" in str(e) or "conv_k7" in str(e):
                    old = cfg.link.fec_scheme
                    cfg.link.fec_scheme = "hamming74"
                    tx_syms, _ = build_transmission(hdr_b, payload, cfg)
                    print(f"[WARN] FEC '{old}' unsupported. Fallback to 'hamming74' for {m}.")
                else:
                    raise
            return len(tx_syms), hdr, payload
        else:
            # serialize fresh
            cfg_serialize = CFG.build_config(modality=m)  # app-layer settings only
            inp = inputs[m]
            hdr, payload = serialize_content(m, inp, app_cfg=cfg_serialize.app, text_encoding="utf-8", validate_image_mode=True)
            prebuilt[m] = {"hdr": hdr, "payload": payload}
            hdr_b = hdr.to_bytes()
            try:
                tx_syms, _ = build_transmission(hdr_b, payload, cfg)
            except Exception as e:
                if "Unknown FEC scheme" in str(e) or "rs255_223" in str(e) or "conv_k7" in str(e):
                    old = cfg.link.fec_scheme
                    cfg.link.fec_scheme = "hamming74"
                    tx_syms, _ = build_transmission(hdr_b, payload, cfg)
                    print(f"[WARN] FEC '{old}' unsupported. Fallback to 'hamming74' for {m}.")
                else:
                    raise
            return len(tx_syms), hdr, payload

    mcs = {m: init_mcs[m].copy() for m in MODALITIES}
    # initial estimate
    sym_len = {}
    total = 0
    for m in MODALITIES:
        n, hdr, payload = estimate_for_mod(m, mcs[m])
        sym_len[m] = n; total += n
        prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}

    tol = int(max(1, round(target_total_syms * 0.01)))  # ±1% tolerance
    max_iter = 20

    # Which modalities to adjust first?
    if priority == "edge":
        lighten_order = ["segmentation", "depth", "text"]  # do not lighten 'edge'
        strengthen_order = ["segmentation", "depth", "text"]  # fill budget here first
    elif priority == "depth":
        lighten_order = ["segmentation", "edge", "text"]
        strengthen_order = ["segmentation", "edge", "text"]
    else:
        lighten_order = ["segmentation", "depth", "edge", "text"]
        strengthen_order = ["segmentation", "depth", "edge", "text"]

    it = 0
    while abs(total - target_total_syms) > tol and it < max_iter:
        it += 1
        if total > target_total_syms:
            # Too heavy → lighten non-priority modalities
            changed = False
            for m in lighten_order:
                if m == priority: 
                    continue
                cand = _next_lighter(mcs[m])
                if cand is None: 
                    continue
                # try this step
                mcs[m] = cand
                n, hdr, payload = estimate_for_mod(m, mcs[m])
                total = sum(sym_len[x] if x != m else n for x in MODALITIES)
                sym_len[m] = n
                prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}
                changed = True
                if abs(total - target_total_syms) <= tol:
                    break
            if not changed:
                break  # cannot lighten further
        else:
            # Too light → strengthen non-priority modalities to fill budget
            changed = False
            for m in strengthen_order:
                if m == priority:
                    continue
                cand = _next_stronger(mcs[m])
                if cand is None:
                    continue
                mcs[m] = cand
                n, hdr, payload = estimate_for_mod(m, mcs[m])
                total = sum(sym_len[x] if x != m else n for x in MODALITIES)
                sym_len[m] = n
                prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}
                changed = True
                if abs(total - target_total_syms) <= tol:
                    break
            if not changed:
                break

    return mcs, prebuilt

# ---------------------- Main runner ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, choices=["eep", "uep_edge", "uep_depth", "all"], default="all")
    ap.add_argument("--snrs", type=str, default="1,4,8,12")
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default="rayleigh")
    ap.add_argument("--output_root", type=str, default="outputs/experiments")
    ap.add_argument("--drop_bad_frames", type=int, choices=[0,1], default=1)
    # Optional: custom inputs
    ap.add_argument("--edge", type=str, default=None)
    ap.add_argument("--depth", type=str, default=None)
    ap.add_argument("--seg", type=str, default=None)
    ap.add_argument("--text", type=str, default=None)
    args = ap.parse_args()

    snr_list = [float(s) for s in args.snrs.split(",") if s.strip()]
    drop_bad = bool(args.drop_bad_frames)

    # Resolve inputs
    inputs = {
        "edge": _ensure_input_path("edge", args.edge),
        "depth": _ensure_input_path("depth", args.depth),
        "segmentation": _ensure_input_path("segmentation", args.seg),
        "text": _ensure_input_path("text", args.text),
    }

    scenarios = []
    if args.scenario == "all":
        scenarios = ["eep", "uep_edge", "uep_depth"]
    else:
        scenarios = [args.scenario]

    # Seeds: fix per (snr, modality) so EEP/UEP are comparable
    base_seed = 12345

    for scen in scenarios:
        out_root = os.path.join(args.output_root, scen)
        os.makedirs(out_root, exist_ok=True)
        summary_csv = os.path.join(out_root, f"summary_{scen}.csv")
        if not os.path.isfile(summary_csv):
            with open(summary_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "scenario","snr_db","modality","fec","modulation","interleaver_depth",
                    "symbols_tx","frames","bad_frames","all_crc_ok","psnr_db","output_path"
                ])

        # 1) Baseline EEP total symbols (target for equal airtime)
        eep_mcs = _baseline_eep_mcs()
        # estimate once (snr doesn't affect symbol count)
        eep_sym_total = 0
        eep_prebuilt = {}
        for m in MODALITIES:
            cfg = _build_cfg(m, eep_mcs[m], args.channel, snr_db=snr_list[0], seed=base_seed, drop_bad_frames=drop_bad)
            n, hdr, payload = _estimate_symbols_for_modality(m, inputs[m], cfg)
            eep_sym_total += n
            eep_prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}

        # 2) Scenario initial MCS
        if scen == "eep":
            final_mcs = eep_mcs
            prebuilt = eep_prebuilt
            priority = "none"
        elif scen == "uep_edge":
            init = _uep_edge_initial()
            final_mcs, prebuilt = _equal_airtime_adjust(
                target_total_syms=eep_sym_total, init_mcs=init, inputs=inputs,
                chan_type=args.channel, snr_db=snr_list[0], seed=base_seed,
                drop_bad_frames=drop_bad, priority="edge"
            )
            priority = "edge"
        else:  # uep_depth
            init = _uep_depth_initial()
            final_mcs, prebuilt = _equal_airtime_adjust(
                target_total_syms=eep_sym_total, init_mcs=init, inputs=inputs,
                chan_type=args.channel, snr_db=snr_list[0], seed=base_seed,
                drop_bad_frames=drop_bad, priority="depth"
            )
            priority = "depth"

        # Report equal-airtime check (one-line print)
        total_syms = sum(prebuilt[m]["sym_len"] for m in MODALITIES) if scen != "eep" else eep_sym_total
        diff_pct = 100.0 * (total_syms - eep_sym_total) / max(1, eep_sym_total)
        print(f"[{scen}] Equal-airtime: total_syms={total_syms}, baseline={eep_sym_total}, diff={diff_pct:+.2f}%")

        # 3) Run transmissions for each SNR
        for snr_db in snr_list:
            snr_dir = os.path.join(out_root, f"snr_{int(snr_db)}dB")
            os.makedirs(snr_dir, exist_ok=True)
            # stable per (snr, modality)
            records = []
            for idx, m in enumerate(MODALITIES):
                # build cfg for this modality
                seed = base_seed + int(snr_db) * 100 + idx
                cfg = _build_cfg(m, final_mcs[m], args.channel, snr_db=snr_db, seed=seed, drop_bad_frames=drop_bad)

                # make output dir
                mdir = os.path.join(snr_dir, m)
                os.makedirs(mdir, exist_ok=True)

                # run
                rec = _run_single_transmission(
                    modality=m,
                    input_path=inputs[m],
                    cfg=cfg,
                    channel=args.channel,
                    snr_db=snr_db,
                    out_dir=mdir,
                    prebuilt_hdr=prebuilt[m]["hdr"] if m in prebuilt else None,
                    prebuilt_payload=prebuilt[m]["payload"] if m in prebuilt else None,
                )
                # add actual symbols used (recalculate here for exactness)
                # (Symbol count depends only on link/mod, not on channel)
                cfg_count = _build_cfg(m, final_mcs[m], args.channel, snr_db=snr_db, seed=seed, drop_bad_frames=drop_bad)
                n_sym, _, _ = _estimate_symbols_for_modality(m, inputs[m], cfg_count)
                rec["symbols_tx"] = int(n_sym)

                # write per-modality meta
                meta = {
                    "scenario": scen, "snr_db": snr_db, "modality": m,
                    "mcs": final_mcs[m].as_dict(), "symbols_tx": int(n_sym),
                    "priority": priority
                }
                write_json(os.path.join(mdir, "meta.json"), meta)

                # to summary.csv
                row = [
                    scen, snr_db, m, rec.get("fec"), rec.get("modulation"), rec.get("interleaver_depth"),
                    int(rec.get("symbols_tx", n_sym)),
                    int(rec.get("n_frames", rec.get("frames", 0))),
                    int(rec.get("n_bad_frames", rec.get("bad_frames", 0))),
                    bool(rec.get("all_crc_ok", False)),
                    (None if rec.get("psnr_db") is None else float(rec["psnr_db"])),
                    rec.get("output_path", ""),
                ]
                records.append(row)

            # append to scenario summary CSV
            with open(summary_csv, "a", newline="") as f:
                w = csv.writer(f); [w.writerow(r) for r in records]

    print("Done. See outputs under:", os.path.abspath(args.output_root))

if __name__ == "__main__":
    main()


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
# -------------------------
def _qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(b) % 2 != 0:
        b = np.concatenate([b, np.zeros(1, dtype=np.uint8)])
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
# -------------------------
def _16qam_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-len(b)) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    quads = b.reshape(-1, 4).astype(np.float64)
    I = (1.0 - 2.0 * quads[:, 0]) * (3.0 - 2.0 * quads[:, 1])
    Q = (1.0 - 2.0 * quads[:, 2]) * (3.0 - 2.0 * quads[:, 3])
    syms = (I + 1j * Q) / np.sqrt(10.0)
    return syms

def _16qam_demod(symbols: np.ndarray) -> np.ndarray:
    x = symbols * np.sqrt(10.0)
    I = x.real
    Q = x.imag
    def level_to_bits(vals):
        b1 = (vals < 0).astype(np.uint8)          # sign
        b0 = (np.abs(vals) < 2).astype(np.uint8)  # inner(1) / outer(0)
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
    # Preamble: alternating 1010... (BPSK, real) — **exactly** preamble_len_bits long
    pre_bits = np.tile(np.array([1, 0], dtype=np.uint8), preamble_len_bits // 2)
    if preamble_len_bits % 2 == 1:
        pre_bits = np.concatenate([pre_bits, np.array([1], dtype=np.uint8)])
    pre_sym = _bpsk_mod(pre_bits).astype(np.complex128)

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



