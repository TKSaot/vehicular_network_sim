from __future__ import annotations
import os, argparse, json, datetime, math
import numpy as np
from PIL import Image

# MODIFIED: Import the hamming module directly to access both decoders
from . import hamming74 as ham
from .config import ExperimentConfig
from .application import (
    serialize_content, AppHeader, deserialize_content,
    _build_seg_ids_and_palette, _suppress_white_boundaries, _load_image,
    _build_text_codebook, _POPCNT_9
)
from .utils import (
    bytes_to_bits, bits_to_bytes, append_crc32,
    verify_and_strip_crc32, block_deinterleave,
    derepeat_bits_majority,
    permute_bytes, unpermute_bytes, derive_modality_seed,
)
from .ofdm import assemble_grid
from .channel import rayleigh_ofdm, awgn_ofdm
from .metrics import psnr, miou_from_ids, f1_binary_edge, ssim
from .presets import POWER_PRESETS

MODS = ["text","edge","depth","segmentation"]
# Heuristic to validate parsed headers per modality (more realistic for TEXT where H/W=0).
def _hdr_is_plausible(m: str, cand: AppHeader, exp_payload_len_bytes: int) -> bool:
    try:
        if cand is None: return False
        if cand.modality != m: return False
        if m == "text":
            # Expect bits=8 or 9 and a sensible payload length near the TX length
            if int(getattr(cand, "bits_per_sample", 0)) not in (8, 9): return False
            P = int(getattr(cand, "payload_len_bytes", -1))
            if P <= 0: return False
            # Allow ±10%% tolerance to survive a few header bit errors that pass Hamming.
            lo = int(0.9 * exp_payload_len_bytes)
            hi = int(1.1 * exp_payload_len_bytes) + 8
            return lo <= P <= hi
        else:
            # Images: need positive dims and a reasonable payload_len
            H = int(getattr(cand, "height", 0)); W = int(getattr(cand, "width", 0))
            if H <= 0 or W <= 0: return False
            P = int(getattr(cand, "payload_len_bytes", -1))
            if P <= 0: return False
            return True
    except Exception:
        return False
ZERO_MODS: set[str] = set()

# ---------- helpers ----------
def _parse_kv_floats(s: str) -> dict:
    d = {}
    if not s: return d
    for kv in s.split(","):
        if not kv.strip(): continue
        k,v = kv.split("="); d[k.strip()] = float(v.strip())
    return d

def _parse_kv_ints(s: str) -> dict:
    d = {}
    if not s: return d
    for kv in s.split(","):
        if not kv.strip(): continue
        k,v = kv.split("="); d[k.strip()] = int(round(float(v.strip())))
    return d

def _slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_+.") else "-" for ch in s)

def _power_tag(weights: dict, preset: str | None, mode: str) -> str:
    if mode.lower() == "eep": return "EEP"
    if preset: return f"UEP_preset-{_slug(preset)}"
    items = [f"{k[:3]}{weights.get(k,1.0):.3g}" for k in ("text","edge","depth","segmentation") if k in weights]
    return "UEP_" + "-".join(_slug(x) for x in items)

def _resolve_examples_dir(args_examples_dir: str | None) -> str:
    if args_examples_dir: return os.path.abspath(args_examples_dir)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "examples")

def _ecc_link_tag(cfg) -> str:
    fec_tag = "FEC" if cfg.link.fec_enabled else "noFEC"
    decoder_tag = cfg.link.decoder_type
    rep = getattr(cfg.link, 'payload_rep_k', {})
    return (f"link_{fec_tag}-{decoder_tag}_i{int(cfg.link.interleaver_depth)}_hdr{int(cfg.link.header_rep_k)}"
            f"_rep[t{int(rep.get('text',1))}e{int(rep.get('edge',1))}d{int(rep.get('depth',1))}s{int(rep.get('segmentation',1))}]"
            f"__map-{cfg.link.byte_mapping}")

def _ecc_rx_tag(cfg) -> str:
    return ("RXec_"
            f"text-{'none'}_"
            f"edge-{_slug(str(cfg.app.edge_denoise))}_"
            f"depth-{_slug(str(cfg.app.depth_denoise))}_"
            f"seg-{_slug(str(cfg.app.seg_mode))}")

def _after_interleave_len(n_bits_enc: int, depth: int) -> int:
    d = max(1, int(depth))
    cols = int(math.ceil(float(n_bits_enc) / d))
    return d * cols

def _encoded_len(n_bits_raw: int, fec_enabled: bool) -> int:
    if not fec_enabled:
        return n_bits_raw
    return 7 * ((int(n_bits_raw) + 3) // 4)

def _majority_bits(bit_arrays: list[np.ndarray]) -> np.ndarray:
    if not bit_arrays: return np.zeros(0, dtype=np.uint8)
    L = min(len(b) for b in bit_arrays)
    if L == 0: return np.zeros(0, dtype=np.uint8)
    M = np.stack([b[:L].astype(np.uint8, copy=False) for b in bit_arrays], axis=0)
    s = np.sum(M, axis=0)
    return (s >= (M.shape[0]//2 + 1)).astype(np.uint8)

def _average_soft(soft_arrays: list[np.ndarray]) -> np.ndarray:
    if not soft_arrays: return np.zeros(0, dtype=np.float32)
    L = min(len(s) for s in soft_arrays)
    if L == 0: return np.zeros(0, dtype=np.float32)
    M = np.stack([s[:L] for s in soft_arrays], axis=0)
    return np.mean(M, axis=0)

def _flatten_hard_from_cols(cols_complex: np.ndarray, n_need: int) -> tuple[np.ndarray, np.ndarray]:
    R = cols_complex.real
    bC = (R.reshape(-1) >= 0).astype(np.uint8)[:n_need]
    bF = (R.T.reshape(-1) >= 0).astype(np.uint8)[:n_need]
    if bC.size < n_need: bC = np.pad(bC, (0, n_need - bC.size))
    if bF.size < n_need: bF = np.pad(bF, (0, n_need - bF.size))
    return bC, bF

def _flatten_soft_from_cols(cols_complex: np.ndarray, n_need: int) -> tuple[np.ndarray, np.ndarray]:
    R = cols_complex.real
    sC = R.reshape(-1).astype(np.float32, copy=False)[:n_need]
    sF = R.T.reshape(-1).astype(np.float32, copy=False)[:n_need]
    if sC.size < n_need: sC = np.pad(sC, (0, n_need - sC.size))
    if sF.size < n_need: sF = np.pad(sF, (0, n_need - sF.size))
    return sC, sF

def _estimate_snr_db_from_real(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if r.size == 0: return 0.0
    s = np.where(r >= 0, 1.0, -1.0)
    proj = r * s
    mu = float(np.mean(proj))
    resid = r - s * mu
    var = float(np.var(resid)) if resid.size else 0.0
    if var <= 0: return 99.0
    snr = (mu * mu) / var
    return float(10.0 * np.log10(max(snr, 1e-12)))

def _text_payload_score(payload_bytes: bytes, bits: int, symbols: str) -> int:
    import numpy as _np
    enc_map, chosen_codes, lut_idx = _build_text_codebook(symbols, bits)
    if bits != 8:
        arr = _np.frombuffer(payload_bytes, dtype=_np.uint8)
        cc = _np.array(chosen_codes, dtype=_np.uint16) & 0xFF
        x = (arr.astype(_np.uint16)[:, None] ^ cc[None, :]).astype(_np.int64)
        d = _np.take(_POPCNT_9, x)
        return int(_np.min(d, axis=1).sum())
    else:
        arr = _np.frombuffer(payload_bytes, dtype=_np.uint8)
        cc = _np.array(chosen_codes, dtype=_np.uint16)
        x = (arr.astype(_np.uint16)[:, None] ^ cc[None, :]).astype(_np.int64)
        d = _np.take(_POPCNT_9, x)
        return int(_np.min(d, axis=1).sum())

def _fix_img_shape(arr: np.ndarray, H: int, W: int, ch3: bool) -> np.ndarray:
    a = np.asarray(arr)
    H = max(1,int(H)); W = max(1,int(W))
    if ch3:
        if a.ndim == 3 and a.shape[0] == H and a.shape[1] == W and a.shape[2] >= 3:
            return np.ascontiguousarray(a[..., :3].astype(np.uint8, copy=False))
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        flat = a.reshape(-1)
        need = H * W * 3
        if flat.size < need: flat = np.pad(flat, (0, need - flat.size))
        return flat[:need].reshape(H, W, 3).astype(np.uint8, copy=False)
    else:
        if a.ndim == 2 and a.shape[0] == H and a.shape[1] == W:
            return np.ascontiguousarray(a.astype(np.uint8, copy=False))
        flat = a.reshape(-1)
        need = H * W
        if flat.size < need: flat = np.pad(flat, (0, need - flat.size))
        return flat[:need].reshape(H, W).astype(np.uint8, copy=False)

def _ids_from_rgb_nearest_palette(rx_rgb_u8: np.ndarray, palette_u8: np.ndarray, max_bytes: int = 200_000_000) -> np.ndarray:
    rx = rx_rgb_u8.reshape(-1, 3).astype(np.float32, copy=False)
    pal = palette_u8.astype(np.float32, copy=False)
    N, K = rx.shape[0], pal.shape[0]
    if K <= 0 or N <= 0:
        return np.zeros((rx_rgb_u8.shape[0], rx_rgb_u8.shape[1]), dtype=np.int64)
    chunk = max(1, min(N, int(max_bytes // (max(K,1) * 4))))
    out = np.empty(N, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        blk = rx[start:end, :]
        d2 = ((blk[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        out[start:end] = np.argmin(d2, axis=1).astype(np.int64, copy=False)
    H, W = rx_rgb_u8.shape[:2]
    return out.reshape(H, W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--channel", type=str, choices=["rayleigh","awgn"], default=None)
    ap.add_argument("--mode", type=str, choices=["eep","uep"], default="eep")
    ap.add_argument("--power-preset", type=str, default="")
    ap.add_argument("--power", type=str, default="")
    ap.add_argument("--power-file", type=str, default="")
    ap.add_argument("--examples-dir", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--save-policy", type=str, choices=["all","crc_only","nonzero_only","crc_and_nonzero_only"], default="all")
    ap.add_argument("--byte-mapping", type=str, choices=["none","permute"], default=None)
    ap.add_argument("--byte-seed", type=int, default=None)
    ap.add_argument("--payload-rep", type=str, default="")
    ap.add_argument("--no-fec", action="store_true", help="Disable Hamming(7,4) FEC")
    ap.add_argument("--decoder-type", type=str, choices=["hard", "soft"], default=None, help="Type of FEC decoder to use")
    ap.add_argument("--ecc-profile", type=str, choices=["none","rxstrong"], default=None)
    ap.add_argument("--edge-denoise", type=str, choices=["none","gentle","medium","strong"], default=None)
    ap.add_argument("--depth-denoise", type=str, choices=["none","median3","median5"], default=None)
    ap.add_argument("--seg-mode", type=str, choices=["none","majority3","majority5","strong"], default=None)
    ap.add_argument("--metrics", type=str, default="all", help="Comma list among: all|none|edge|depth|seg|text")
    ap.add_argument("--metrics-if-crc", action="store_true", help="Only compute metric if its CRC passed")
    # MODIFIED: Add a command-line argument for the channel seed
    ap.add_argument("--seed", type=int, default=None, help="Seed for the channel random number generator")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.snr_db is not None: cfg.chan.snr_db = float(args.snr_db)
    if args.channel is not None: cfg.chan.channel = args.channel
    if args.out_root is not None: cfg.paths.output_root = args.out_root
    if args.byte_mapping is not None: cfg.link.byte_mapping = args.byte_mapping
    if args.byte_seed is not None: cfg.link.byte_seed = int(args.byte_seed)
    if args.no_fec: cfg.link.fec_enabled = False
    if args.decoder_type is not None:
        cfg.link.decoder_type = args.decoder_type
    if args.payload_rep:
        cur = getattr(cfg.link, 'payload_rep_k', {}).copy()
        cur.update(_parse_kv_ints(args.payload_rep))
        cfg.link.payload_rep_k = cur
    # MODIFIED: Set the channel seed if provided via the command line
    if args.seed is not None:
        cfg.chan.seed = args.seed

    if args.ecc_profile == "none":
        cfg.app.edge_denoise = "none"
        cfg.app.depth_denoise = "none"
        cfg.app.seg_mode = "none"
    elif args.ecc_profile == "rxstrong":
        cfg.app.edge_denoise = "strong"
        cfg.app.depth_denoise = "median5"
        cfg.app.seg_mode = "strong"
    if args.edge_denoise is not None: cfg.app.edge_denoise = args.edge_denoise
    if args.depth_denoise is not None: cfg.app.depth_denoise = args.depth_denoise
    if args.seg_mode is not None: cfg.app.seg_mode = args.seg_mode

    examples_dir = _resolve_examples_dir(args.examples_dir)
    input_paths = {
        "text": os.path.join(examples_dir, "sample.txt"),
        "edge": os.path.join(examples_dir, "edge_00001_.png"),
        "depth": os.path.join(examples_dir, "depth_00001_.png"),
        "segmentation": os.path.join(examples_dir, "segmentation_00001_.png"),
    }

    # --- serialize ---
    hdrs = {}; payloads = {}
    for m in MODS:
        hdr, pl = serialize_content(m, input_paths[m], app_cfg=cfg.app)
        hdrs[m] = hdr.to_bytes()
        if cfg.link.byte_mapping == "permute":
            seed_m = derive_modality_seed(cfg.link.byte_seed, m)
            pl = permute_bytes(pl, seed_m)
        payloads[m] = append_crc32(pl)

    # --- Power selection ---
    mode = args.mode.lower()
    preset_name = None
    weights = {m: 1.0 for m in MODS}
    user = _parse_kv_floats(args.power)
    if args.power_file:
        with open(args.power_file, "r", encoding="utf-8") as f:
            js = json.load(f)
        if isinstance(js, dict) and all(isinstance(v,(int,float)) for v in js.values()):
            weights.update({k: float(v) for k,v in js.items()}); mode = "uep"
        else:
            raise ValueError("power-file must be a dict of floats for this runner")
    elif user:
        weights.update(user); mode = "uep"
    elif args.power_preset:
        key = args.power_preset.strip().lower()
        if key not in POWER_PRESETS:
            raise ValueError(f"Unknown power preset: {args.power_preset}")
        weights = POWER_PRESETS[key].copy()
        preset_name = key
        mode = "eep" if key == "eep" else "uep"

    s = sum(weights.values()); weights = {k: v*len(MODS)/s for k,v in weights.items()}
    ZERO_MODS.clear()
    for m,w in weights.items():
        if w <= 0.0: ZERO_MODS.add(m)

    # --- OFDM -> channel -> equalize ---
    X, sc_slices, syms_per_mod = assemble_grid(
        payload_per_mod=payloads,
        header_per_mod=hdrs,
        cfg_ofdm=cfg.ofdm,
        cfg_link=cfg.link,
        power_linear_per_mod=weights
    )
    if cfg.chan.channel == "rayleigh":
        Y, H = rayleigh_ofdm(X, cfg.chan.snr_db, seed=cfg.chan.seed,
                             n_fft=cfg.ofdm.n_fft, cp_len=cfg.ofdm.cp_len)
    else:
        Y = awgn_ofdm(X, cfg.chan.snr_db, seed=cfg.chan.seed,
                      n_fft=cfg.ofdm.n_fft, cp_len=cfg.ofdm.cp_len)
        H = np.ones(X.shape[0], dtype=np.complex128)

    if cfg.chan.channel == "awgn":
        Yeq = Y
    else:
        pilot_tx = X[:, 0]
        denom = np.where(np.abs(pilot_tx) < 1e-12, 1.0+0j, pilot_tx)
        Hhat = Y[:, 0] / denom
        eps = 1e-3
        Hhat_reg = np.where(np.abs(Hhat) < eps, eps, Hhat) # Regularize to avoid division by zero
        Yeq = (Y / Hhat_reg[:, None])

    # --- decode per modality ---
    results = {}
    snr_by_mod = {}
    for m in MODS:
        sl = sc_slices.get(m)
        if sl is None: continue
        per = sl.stop - sl.start
        
        L_hdr0 = len(bytes_to_bits(hdrs[m]))
        L_pay0 = len(bytes_to_bits(payloads[m]))
        D = cfg.link.interleaver_depth; Krep = cfg.link.header_rep_k
        rep_k = int(getattr(cfg.link, 'payload_rep_k', {}).get(m, 1))

        L_hdr1 = _encoded_len(L_hdr0, cfg.link.fec_enabled)
        L_hdr2 = _after_interleave_len(L_hdr1, D)
        L_hdr3 = Krep * L_hdr2
        n_hdr_cols = int(math.ceil(L_hdr3 / per)) if per > 0 else 0

        L_pay1 = _encoded_len(L_pay0, cfg.link.fec_enabled)
        L_pay1r = rep_k * L_pay1
        L_pay2 = _after_interleave_len(L_pay1r, D)
        n_pay_cols = int(math.ceil(L_pay2 / per)) if per > 0 else 0

        hdr_cols = Yeq[sl, 1:1+n_hdr_cols]

        if cfg.link.decoder_type == 'soft' and cfg.link.fec_enabled:
            def _try_decode_hdr(soft_flat: np.ndarray) -> np.ndarray:
                chunks = []
                for i in range(Krep):
                    start, end = i*L_hdr2, min((i+1)*L_hdr2, soft_flat.size)
                    chunk = soft_flat[start:end]
                    if chunk.size < L_hdr2:
                        chunk = np.pad(chunk, (0, L_hdr2 - chunk.size))
                    chunks.append(chunk)
                
                avg_soft_chunk = _average_soft(chunks)
                d_inter = block_deinterleave(avg_soft_chunk, D, original_len=L_hdr1)
                decoded_bits = ham.decode_soft(d_inter)
                return decoded_bits[:L_hdr0]

            sC, sF = _flatten_soft_from_cols(hdr_cols, L_hdr3)
            hdr_bits_candidates = [_try_decode_hdr(sC), _try_decode_hdr(sF)]
        else:
            def _try_decode_hdr(bits_flat: np.ndarray) -> np.ndarray:
                chunks = []
                for i in range(Krep):
                    start, end = i*L_hdr2, min((i+1)*L_hdr2, bits_flat.size)
                    chunk = bits_flat[start:end]
                    if chunk.size < L_hdr2:
                        chunk = np.pad(chunk, (0, L_hdr2 - chunk.size), 'constant')
                    d_inter = block_deinterleave(chunk, D, original_len=L_hdr1)
                    if cfg.link.fec_enabled:
                        decoded_chunk = ham.decode(d_inter)
                    else:
                        decoded_chunk = d_inter
                    chunks.append(decoded_chunk[:L_hdr0])
                return _majority_bits(chunks)
            
            bC, bF = _flatten_hard_from_cols(hdr_cols, L_hdr3)
            hdr_bits_candidates = [_try_decode_hdr(bC), _try_decode_hdr(bF)]

        order_idx = 0
        rx_hdr = None
        for idx, hb in enumerate(hdr_bits_candidates):
            try:
                cand = AppHeader.from_bytes(bits_to_bytes(hb))
                if _hdr_is_plausible(m, cand, exp_payload_len_bytes=len(payloads[m]) - 4):
                    rx_hdr = cand; order_idx = idx; break
            except Exception:
                pass
        if rx_hdr is None:
            for idx, hb in enumerate(hdr_bits_candidates):
                try:
                    rx_hdr = AppHeader.from_bytes(bits_to_bytes(hb)); order_idx = idx; break
                except Exception:
                    pass
        if rx_hdr is None:
            rx_hdr = AppHeader.from_bytes(hdrs[m])

        pay_cols = Yeq[sl, 1+n_hdr_cols : 1+n_hdr_cols+n_pay_cols]
        
        payload2 = b''
        
        if cfg.link.decoder_type == 'soft' and cfg.link.fec_enabled:
            def _decode_payload_soft(soft_in: np.ndarray) -> tuple[bool, bytes]:
                deinter = block_deinterleave(soft_in, D, original_len=L_pay1r)
                if rep_k > 1:
                    reshaped = deinter.reshape(-1, rep_k)
                    averaged = np.mean(reshaped, axis=1)
                    derep = averaged
                else:
                    derep = deinter
                dec = ham.decode_soft(derep)[:L_pay0]
                bb = bits_to_bytes(dec)
                return verify_and_strip_crc32(bb)

            sC_pay, sF_pay = _flatten_soft_from_cols(pay_cols, L_pay2)
            first, other = (sC_pay, sF_pay) if order_idx == 0 else (sF_pay, sC_pay)
            ok_crc, payload_perm = _decode_payload_soft(first)
            if not ok_crc:
                ok2, payload2 = _decode_payload_soft(other)
                if ok2:
                    ok_crc, payload_perm = ok2, payload2
                    order_idx = 1 - order_idx
        else:
            def _decode_payload_hard(bits_in: np.ndarray) -> tuple[bool, bytes]:
                deinter = block_deinterleave(bits_in, D, original_len=L_pay1r)
                if rep_k > 1:
                    derep = derepeat_bits_majority(deinter, rep_k, original_len=L_pay1)
                else:
                    derep = deinter
                if cfg.link.fec_enabled:
                    dec = ham.decode(derep)[:L_pay0]
                else:
                    dec = derep[:L_pay0]
                bb = bits_to_bytes(dec)
                return verify_and_strip_crc32(bb)

            bC_pay, bF_pay = _flatten_hard_from_cols(pay_cols, L_pay2)
            first, other = (bC_pay, bF_pay) if order_idx == 0 else (bF_pay, bC_pay)
            ok_crc, payload_perm = _decode_payload_hard(first)
            if not ok_crc:
                ok2, payload2 = _decode_payload_hard(other)
                if ok2:
                    ok_crc, payload_perm = ok2, payload2
                    order_idx = 1 - order_idx
        
        if not ok_crc:
            if m == "text":
                bits_ps = int(getattr(cfg.app, "text_bits_per_char", 8))
                symbols_ps = getattr(cfg.app, "text_symbols", "abcdefghijklmnopqrstuvwxyz1234567890, .\n")
                seed_m = derive_modality_seed(cfg.link.byte_seed, m) if cfg.link.byte_mapping == "permute" else None
                cand1 = unpermute_bytes(payload_perm, seed_m) if seed_m is not None else payload_perm
                cand2 = unpermute_bytes(payload2,     seed_m) if seed_m is not None else payload2
                s1 = _text_payload_score(cand1, bits=bits_ps, symbols=symbols_ps)
                s2 = _text_payload_score(cand2, bits=bits_ps, symbols=symbols_ps)
                if s2 < s1:
                    payload_perm = payload2
                    order_idx = 1 - order_idx
            if not ok_crc:
                try:
                    rx_hdr = AppHeader.from_bytes(hdrs[m])
                except Exception:
                    pass

        r_pay = pay_cols.real.reshape(-1)
        snr_by_mod[m] = _estimate_snr_db_from_real(r_pay)

        if cfg.link.byte_mapping == "permute":
            seed_m = derive_modality_seed(cfg.link.byte_seed, m)
            payload_clean = unpermute_bytes(payload_perm, seed_m)
        else:
            payload_clean = payload_perm

        try:
            _ = rx_hdr.modality
        except Exception:
            rx_hdr = AppHeader.from_bytes(hdrs[m]); ok_crc = False

        text_str, img_arr = deserialize_content(rx_hdr, payload_clean, app_cfg=cfg.app)
        if m != "text":
            Hh, Ww = max(1,int(rx_hdr.height)), max(1,int(rx_hdr.width))
            if not isinstance(img_arr, np.ndarray) or img_arr.size == 0:
                img_arr = (np.zeros((Hh, Ww), np.uint8) if m in ("edge","depth") else np.zeros((Hh,Ww,3), np.uint8))

        results[m] = {
            "crc_ok": bool(ok_crc),
            "rx_hdr": rx_hdr,
            "text": text_str if m == "text" else "",
            "img": img_arr if m != "text" else None,
            "order": "C" if order_idx==0 else "F"
        }

    # --- outputs ---
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    power_tag = _power_tag(weights, preset_name, mode)
    tag_extra = f"__{_slug(args.tag)}" if args.tag else ""
    out_root = cfg.paths.output_root if not args.out_root else args.out_root
    ecc_tag = _ecc_rx_tag(cfg) + "__" + _ecc_link_tag(cfg)
    out_dir = os.path.join(
        out_root,
        cfg.chan.channel,
        ecc_tag,
        f"snr{int(round(cfg.chan.snr_db))}",
        f"{power_tag}__N{cfg.ofdm.n_fft}CP{cfg.ofdm.cp_len}{tag_extra}",
        ts
    )
    os.makedirs(out_dir, exist_ok=True)

    outputs = {}
    p_txt = os.path.join(out_dir, "text_received.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(results.get("text", {}).get("text", ""))
    outputs["text"] = p_txt

    def _should_save(mod: str) -> bool:
        policy = args.save_policy
        if policy == 'all': return True
        if mod not in results: return False
        if policy == 'crc_only': return bool(results[mod]['crc_ok'])
        if policy == 'nonzero_only': return mod not in ZERO_MODS
        if policy == 'crc_and_nonzero_only': return (mod not in ZERO_MODS) and bool(results[mod]['crc_ok'])
        return True

    p_edge = os.path.join(out_dir, "edge_received.png")
    p_depth = os.path.join(out_dir, "depth_received.png")
    p_seg  = os.path.join(out_dir, "segmentation_received.png")
    saved = []
    if _should_save("edge"):
        a = _fix_img_shape(results["edge"]["img"], results["edge"]["rx_hdr"].height, results["edge"]["rx_hdr"].width, ch3=False)
        Image.fromarray(a).save(p_edge, format="PNG")
        outputs["edge"] = p_edge; saved.append("edge")
    else:
        outputs["edge"] = None
    if _should_save("depth"):
        a = _fix_img_shape(results["depth"]["img"], results["depth"]["rx_hdr"].height, results["depth"]["rx_hdr"].width, ch3=False)
        Image.fromarray(a).save(p_depth, format="PNG")
        outputs["depth"] = p_depth; saved.append("depth")
    else:
        outputs["depth"] = None
    if _should_save("segmentation"):
        a = _fix_img_shape(results["segmentation"]["img"], results["segmentation"]["rx_hdr"].height, results["segmentation"]["rx_hdr"].width, ch3=True)
        Image.fromarray(a).save(p_seg, format="PNG")
        outputs["segmentation"] = p_seg; saved.append("segmentation")
    else:
        outputs["segmentation"] = None

    # --- metrics (selective + safe) ---
    metric_flags = set([t.strip().lower() for t in (args.metrics or "all").split(",") if t.strip()])
    if "all" in metric_flags: metric_flags = {"edge","depth","seg","text"}
    if "none" in metric_flags: metric_flags = set()

    metrics = {}
    metric_errors = {}

    try:
        if ("edge" in metric_flags) and results.get("edge") and (not args.metrics_if_crc or results["edge"]["crc_ok"]):
            e_gt = (_load_image(input_paths["edge"], "L") >= 128).astype(np.uint8) * 255
            e_rx = _fix_img_shape(results["edge"]["img"], results["edge"]["rx_hdr"].height, results["edge"]["rx_hdr"].width, ch3=False)
            metrics["edge_f1"] = float(f1_binary_edge(e_gt, e_rx))
            metrics["edge_ssim"] = float(ssim(e_gt, e_rx, data_range=255))
        else:
            metrics["edge_f1"] = None; metrics["edge_ssim"] = None
    except Exception as ex:
        metric_errors["edge"] = str(ex); metrics["edge_f1"] = None; metrics["edge_ssim"] = None

    try:
        if ("depth" in metric_flags) and results.get("depth") and (not args.metrics_if_crc or results["depth"]["crc_ok"]):
            d_gt = _load_image(input_paths["depth"], "L").astype(np.uint8)
            d_rx = _fix_img_shape(results["depth"]["img"], results["depth"]["rx_hdr"].height, results["depth"]["rx_hdr"].width, ch3=False)
            metrics["depth_psnr"] = float(psnr(d_gt, d_rx, data_range=255))
            metrics["depth_ssim"] = float(ssim(d_gt, d_rx, data_range=255))
        else:
            metrics["depth_psnr"] = None; metrics["depth_ssim"] = None
    except Exception as ex:
        metric_errors["depth"] = str(ex); metrics["depth_psnr"] = None; metrics["depth_ssim"] = None

    try:
        if ("seg" in metric_flags) and results.get("segmentation") and (not args.metrics_if_crc or results["segmentation"]["crc_ok"]):
            seg_rgb_true = _suppress_white_boundaries(_load_image(input_paths["segmentation"], "RGB"),
                                                      getattr(cfg.app, "seg_white_thresh", 250),
                                                      getattr(cfg.app, "seg_iters", 2))
            ids_true, pal_true, _ = _build_seg_ids_and_palette(seg_rgb_true,
                                                               getattr(cfg.app, "seg_white_thresh", 250))
            rx_rgb = _fix_img_shape(results["segmentation"]["img"],
                                    results["segmentation"]["rx_hdr"].height,
                                    results["segmentation"]["rx_hdr"].width,
                                    ch3=True).astype(np.uint8, copy=False)
            ids_pred = _ids_from_rgb_nearest_palette(rx_rgb, pal_true.astype(np.uint8, copy=False),
                                                     max_bytes=200_000_000)
            metrics["seg_mIoU"] = float(miou_from_ids(ids_true.astype(np.int64),
                                                      ids_pred.astype(np.int64),
                                                      K=int(pal_true.shape[0])))
        else:
            metrics["seg_mIoU"] = None
    except Exception as ex:
        metric_errors["seg"] = str(ex); metrics["seg_mIoU"] = None

    try:
        if ("text" in metric_flags) and results.get("text") and (not args.metrics_if_crc or results["text"]["crc_ok"]):
            with open(input_paths["text"], "r", encoding="utf-8") as f:
                orig_text = f.read()
            recv_text = results["text"]["text"]
            minlen = min(len(orig_text), len(recv_text))
            mism = sum(1 for i in range(minlen) if orig_text[i] != recv_text[i]) + abs(len(orig_text)-len(recv_text))
            metrics["text_cer"] = float(mism / max(1, len(orig_text)))
        else:
            metrics["text_cer"] = None
    except Exception as ex:
        metric_errors["text"] = str(ex); metrics["text_cer"] = None

    snr_global = float(np.nanmean([snr_by_mod.get(k, np.nan) for k in MODS]))

    report = {
        "snr_db": cfg.chan.snr_db,
        "channel": cfg.chan.channel,
        "fec_enabled": cfg.link.fec_enabled,
        "decoder_type": cfg.link.decoder_type,
        "ofdm": {"n_fft": cfg.ofdm.n_fft, "cp_len": cfg.ofdm.cp_len, "used_subcarriers": cfg.ofdm.used_subcarriers},
        "mode": mode.upper(),
        "power_linear": weights,
        "power_preset": preset_name or "",
        "byte_mapping": cfg.link.byte_mapping,
        "byte_seed": cfg.link.byte_seed,
        "payload_rep_k": getattr(cfg.link, "payload_rep_k", {}),
        "subcarrier_slices": {k: [v.start, v.stop] for k,v in sc_slices.items()},
        "inputs": input_paths,
        "outputs": outputs,
        "saved_modalities": saved,
        "metrics": metrics,
        "metrics_config": {"selected": sorted(list(metric_flags)), "only_if_crc": bool(args.metrics_if_crc)},
        "metrics_errors": metric_errors,
        "crc_by_modality": {m: results.get(m, {}).get("crc_ok") for m in MODS},
        "bit_order_by_modality": {m: results.get(m, {}).get("order") for m in MODS},
        "rx_ecc_modes": {
            "edge_denoise": cfg.app.edge_denoise,
            "depth_denoise": cfg.app.depth_denoise,
            "seg_mode": cfg.app.seg_mode,
        },
        "tag": args.tag,
        "snr_est_db_by_modality": snr_by_mod,
        "snr_est_db_overall": snr_global,
        "zero_power_modalities": sorted(list(ZERO_MODS)),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Multimodal OFDM Report ===")
    print(f"Output dir: {out_dir}")
    print(f"SNR(dB): {cfg.chan.snr_db}  Channel: {cfg.chan.channel} (seed={cfg.chan.seed})")
    print(f"FEC: {'Enabled' if cfg.link.fec_enabled else 'Disabled'} (Decoder: {cfg.link.decoder_type})")
    print(f"Mode: {mode.upper()}  Power: {weights}  Preset: {preset_name or '-'}")
    print(f"Byte mapping: {cfg.link.byte_mapping} (seed={cfg.link.byte_seed})")
    print(f"Payload repetition k: {getattr(cfg.link, 'payload_rep_k', {})}")
    print(f"RX ECC: edge={cfg.app.edge_denoise}  depth={cfg.app.depth_denoise}  seg={cfg.app.seg_mode}")
    print("CRC: " + ", ".join([f"{m}:{'OK' if results.get(m, {}).get('crc_ok') else 'NG'}({results.get(m, {}).get('order', '-')})" for m in MODS]))
    
    if metrics.get("edge_f1") is not None: print(f"Edge F1: {metrics['edge_f1']:.3f}  SSIM: {metrics.get('edge_ssim', float('nan')):.3f}", end=" | ")
    else: print("Edge F1: N/A", end=" | ")
    
    if metrics.get("depth_psnr") is not None: print(f"Depth PSNR: {metrics['depth_psnr']:.2f} dB  SSIM: {metrics.get('depth_ssim', float('nan')):.3f}", end=" | ")
    else: print("Depth PSNR: N/A", end=" | ")

    if metrics.get("seg_mIoU") is not None: print(f"Seg mIoU: {metrics['seg_mIoU']:.3f}")
    else: print("Seg mIoU: N/A")

    if metrics.get("text_cer") is not None: print(f"Text CER≈{metrics['text_cer']:.4f}", end=" | ")
    else: print("Text CER: N/A", end=" | ")

    print(f"SNR(est): {snr_global:.2f} dB  {snr_by_mod}")
    print(f"Files: {outputs}")
    if metric_errors:
        print(f"[metrics warnings] {metric_errors}")

if __name__ == "__main__":
    main()