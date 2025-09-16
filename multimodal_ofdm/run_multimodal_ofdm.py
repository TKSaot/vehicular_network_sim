from __future__ import annotations
import os, argparse, json, datetime
import numpy as np
from PIL import Image

from .config import ExperimentConfig
from .application import (
    serialize_content, AppHeader, deserialize_content,
    _build_seg_ids_and_palette, _suppress_white_boundaries
)
from .utils import (
    bytes_to_bits, bits_to_bytes, append_crc32,
    verify_and_strip_crc32, block_deinterleave
)
from .ofdm import assemble_grid
from .channel import rayleigh_ofdm, awgn_ofdm
from .metrics import psnr, miou_from_ids, f1_binary_edge, ssim
from .presets import POWER_PRESETS

MODS = ["text","edge","depth","segmentation"]

# ---------- helpers ----------
def _parse_power_arg(s: str):
    d = {}
    if not s: return d
    for kv in s.split(","):
        if not kv.strip(): continue
        k,v = kv.split("="); d[k.strip()] = float(v.strip())
    return d

def _slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_+.") else "-" for ch in s)

def _power_tag(weights: dict, preset: str | None, mode: str) -> str:
    if mode.lower() == "eep": return "EEP"
    if preset: return f"UEP_preset-{_slug(preset)}"
    items = [f"{k[:3]}{weights[k]}" for k in ("text","edge","depth","segmentation") if k in weights]
    return "UEP_" + "-".join(_slug(x) for x in items)

def _resolve_examples_dir(args_examples_dir: str | None) -> str:
    if args_examples_dir: return os.path.abspath(args_examples_dir)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "examples")

def _abs_or_join(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))

def _ham_encoded_len(n_bits_raw: int) -> int:
    return 7 * ((int(n_bits_raw) + 3) // 4)

def _after_interleave_len(n_bits_enc: int, depth: int) -> int:
    d = max(1, int(depth))
    cols = int(np.ceil(float(n_bits_enc) / d))
    return d * cols

def _majority_bits(bit_arrays: list[np.ndarray]) -> np.ndarray:
    if not bit_arrays: return np.zeros(0, dtype=np.uint8)
    L = min(len(b) for b in bit_arrays)
    if L == 0: return np.zeros(0, dtype=np.uint8)
    M = np.stack([b[:L].astype(np.uint8, copy=False) for b in bit_arrays], axis=0)
    s = np.sum(M, axis=0)
    return (s >= (M.shape[0]//2 + 1)).astype(np.uint8)

# ---- robust saver（まとめ.py 流儀：明示モード） ----
def _fix_img_shape(arr: np.ndarray, H: int, W: int, ch3: bool) -> np.ndarray:
    a = np.asarray(arr, dtype=np.uint8)
    if ch3:
        if a.ndim == 3 and a.shape[:2] == (H, W) and a.shape[2] >= 3:
            return np.ascontiguousarray(a[..., :3], dtype=np.uint8)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        flat = a.reshape(-1)
        need = H * W * 3
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size), mode="constant")
        return flat[:need].reshape(H, W, 3).astype(np.uint8, copy=False)
    else:
        if a.ndim == 2 and a.shape == (H, W):
            return np.ascontiguousarray(a, dtype=np.uint8)
        flat = a.reshape(-1)
        need = H * W
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size), mode="constant")
        return flat[:need].reshape(H, W).astype(np.uint8, copy=False)

def _save_png_uint8(path: str, arr: np.ndarray, modality: str, rx_hdr: AppHeader) -> None:
    H = int(rx_hdr.height) if rx_hdr.height > 0 else arr.shape[0]
    W = int(rx_hdr.width)  if rx_hdr.width  > 0 else arr.shape[1]
    if modality in ("edge","depth"):
        a = _fix_img_shape(arr, H, W, ch3=False)
        Image.frombytes("L", (W, H), np.ascontiguousarray(a).tobytes()).copy().save(path, format="PNG")
    elif modality == "segmentation":
        a = _fix_img_shape(arr, H, W, ch3=True)
        Image.frombytes("RGB", (W, H), np.ascontiguousarray(a).tobytes()).copy().save(path, format="PNG")
    else:
        raise ValueError("unknown modality for saving")

# ---- flatten helpers：C順／F順の両案 ----
def _flatten_bits_from_cols(cols_complex: np.ndarray, n_need: int) -> tuple[np.ndarray, np.ndarray]:
    R = cols_complex.real
    bC = (R.reshape(-1) >= 0).astype(np.uint8)[:n_need]
    bF = (R.T.reshape(-1) >= 0).astype(np.uint8)[:n_need]
    if bC.size < n_need: bC = np.pad(bC, (0, n_need - bC.size))
    if bF.size < n_need: bF = np.pad(bF, (0, n_need - bF.size))
    return bC, bF

# ---- 事後 SNR 推定（BPSK 前提） ----
def _estimate_snr_db_from_real(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return 0.0
    s = np.where(r >= 0, 1.0, -1.0)      # 符号で hard decision
    proj = r * s                          # ±1 方向への射影
    mu = float(np.mean(proj))            # 実効振幅の推定
    resid = r - s * mu
    var = float(np.var(resid)) if resid.size else 0.0
    if var <= 0:
        return 99.0
    snr = (mu * mu) / var
    return float(10.0 * np.log10(max(snr, 1e-12)))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--channel", type=str, choices=["rayleigh","awgn"], default=None)
    ap.add_argument("--mode", type=str, choices=["eep","uep"], default="eep")
    ap.add_argument("--power-preset", type=str, default="", help="eep|text|edge|edge_depth|seg|segmentation")
    ap.add_argument("--power", type=str, default="")
    ap.add_argument("--power-file", type=str, default="")
    ap.add_argument("--examples-dir", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.snr_db is not None: cfg.chan.snr_db = float(args.snr_db)
    if args.channel is not None: cfg.chan.channel = args.channel
    if args.out_root is not None: cfg.paths.output_root = args.out_root

    examples_dir = _resolve_examples_dir(args.examples_dir)
    input_paths = {
        "text": _abs_or_join(examples_dir, "sample.txt"),
        "edge": _abs_or_join(examples_dir, "edge_00001_.png"),
        "depth": _abs_or_join(examples_dir, "depth_00001_.png"),
        "segmentation": _abs_or_join(examples_dir, "segmentation_00001_.png"),
    }

    # --- serialize 全モダリティ ---
    hdrs = {}; payloads = {}
    for m in MODS:
        hdr, pl = serialize_content(m, input_paths[m], app_cfg=None)
        hdrs[m] = hdr.to_bytes()
        payloads[m] = append_crc32(pl)

    # --- Power ---
    mode = args.mode.lower()
    preset_name = None
    weights = {m: 1.0 for m in MODS}
    user = _parse_power_arg(args.power)
    if user:
        weights.update(user); mode = "uep"
    elif args.power_file:
        with open(args.power_file, "r", encoding="utf-8") as f:
            js = json.load(f)
        weights.update({k: float(v) for k, v in js.items()}); mode = "uep"
    elif args.power_preset:
        key = args.power_preset.strip().lower()
        if key not in POWER_PRESETS:
            raise ValueError(f"Unknown power preset: {args.power_preset}")
        weights = POWER_PRESETS[key].copy()
        preset_name = key
        mode = "eep" if key == "eep" else "uep"
    s = sum(weights.values()); weights = {k: v*len(MODS)/s for k,v in weights.items()}

    # --- OFDM Grid → Channel → Equalize ---
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

    pilot_tx = X[:, 0]
    denom = np.where(np.abs(pilot_tx) < 1e-12, 1.0+0j, pilot_tx)
    Hhat = Y[:, 0] / denom
    Hhat = np.where(np.abs(Hhat) < 1e-12, 1.0+0j, Hhat)
    Yeq = (Y / Hhat[:, None])

    # --- decode per modality（C/F 自動判定，CRC 使用） ---
    results = {}
    snr_by_mod = {}
    for m in MODS:
        sl = sc_slices[m]; per = sl.stop - sl.start
        L_hdr0 = len(bytes_to_bits(hdrs[m]))
        L_pay0 = len(bytes_to_bits(payloads[m]))
        D = cfg.link.interleaver_depth; Krep = cfg.link.header_rep_k

        L_hdr1 = _ham_encoded_len(L_hdr0)
        L_hdr2 = _after_interleave_len(L_hdr1, D)
        L_hdr3 = Krep * L_hdr2
        n_hdr_cols = int(np.ceil(L_hdr3 / per))

        L_pay1 = _ham_encoded_len(L_pay0)
        L_pay2 = _after_interleave_len(L_pay1, D)
        n_pay_cols = int(np.ceil(L_pay2 / per))

        hdr_cols = Yeq[sl, 1:1+n_hdr_cols]
        def _try_decode_hdr(bits_flat: np.ndarray) -> np.ndarray:
            from . import hamming74 as ham
            chunks = []
            for i in range(Krep):
                start, end = i*L_hdr2, min((i+1)*L_hdr2, bits_flat.size)
                chunk = bits_flat[start:end]
                if chunk.size < L_hdr2:
                    chunk = np.concatenate([chunk, np.zeros(L_hdr2 - chunk.size, dtype=np.uint8)])
                d_inter = block_deinterleave(chunk, D, original_len=L_hdr1)
                chunks.append(ham.decode(d_inter)[:L_hdr0])
            return _majority_bits(chunks)

        bC, bF = _flatten_bits_from_cols(hdr_cols, L_hdr3)
        hdr_bits_candidates = [ _try_decode_hdr(bC), _try_decode_hdr(bF) ]

        order_idx = 0
        rx_hdr = None
        for idx, hb in enumerate(hdr_bits_candidates):
            try:
                cand = AppHeader.from_bytes(bits_to_bytes(hb))
                if (cand.modality == m) and (cand.height > 0 and cand.width > 0):
                    rx_hdr = cand; order_idx = idx; break
            except Exception:
                pass
        if rx_hdr is None:
            # どちらかが parse できれば採用，無理なら TX ヘッダ
            for idx, hb in enumerate(hdr_bits_candidates):
                try:
                    rx_hdr = AppHeader.from_bytes(bits_to_bytes(hb)); order_idx = idx; break
                except Exception:
                    pass
        if rx_hdr is None:
            rx_hdr = AppHeader.from_bytes(hdrs[m])

        pay_cols = Yeq[sl, 1+n_hdr_cols : 1+n_hdr_cols+n_pay_cols]
        bC_pay, bF_pay = _flatten_bits_from_cols(pay_cols, L_pay2)
        from . import hamming74 as ham
        def _decode_payload(bits_in: np.ndarray) -> tuple[bool, bytes]:
            deinter = block_deinterleave(bits_in, D, original_len=L_pay1)
            dec = ham.decode(deinter)[:L_pay0]
            bb = bits_to_bytes(dec)
            return verify_and_strip_crc32(bb)

        first = (bC_pay if order_idx == 0 else bF_pay)
        ok_crc, payload_clean = _decode_payload(first)
        if not ok_crc:
            other = (bF_pay if order_idx == 0 else bC_pay)
            ok2, payload2 = _decode_payload(other)
            if ok2:
                ok_crc, payload_clean = ok2, payload2
                order_idx = 1 - order_idx

        # 事後 SNR を推定（このモダリティの全ペイロード列）
        r_pay = pay_cols.real.reshape(-1)
        snr_by_mod[m] = _estimate_snr_db_from_real(r_pay)

        # 画像／テキスト復元
        try:
            _ = rx_hdr.modality
        except Exception:
            rx_hdr = AppHeader.from_bytes(hdrs[m]); ok_crc = False

        text_str, img_arr = deserialize_content(rx_hdr, payload_clean, app_cfg=None)
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

    # --- 出力（フラット配置） ---
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    power_tag = _power_tag(weights, preset_name, mode)
    tag_extra = f"__{_slug(args.tag)}" if args.tag else ""
    out_root = cfg.paths.output_root if not args.out_root else args.out_root
    folder = f"{ts}__ALL__{cfg.chan.channel}_snr{int(round(cfg.chan.snr_db))}__N{cfg.ofdm.n_fft}CP{cfg.ofdm.cp_len}__{power_tag}{tag_extra}"
    out_dir = os.path.join(out_root, folder)
    os.makedirs(out_dir, exist_ok=True)

    outputs = {}
    # テキスト
    p_txt = os.path.join(out_dir, "text_received.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(results["text"]["text"])
    outputs["text"] = p_txt
    # 画像
    p_edge = os.path.join(out_dir, "edge_received.png")
    p_depth = os.path.join(out_dir, "depth_received.png")
    p_seg  = os.path.join(out_dir, "segmentation_received.png")
    _save_png_uint8(p_edge,  results["edge"]["img"], modality="edge", rx_hdr=results["edge"]["rx_hdr"])
    _save_png_uint8(p_depth, results["depth"]["img"], modality="depth", rx_hdr=results["depth"]["rx_hdr"])
    _save_png_uint8(p_seg,   results["segmentation"]["img"], modality="segmentation", rx_hdr=results["segmentation"]["rx_hdr"])
    outputs["edge"] = p_edge; outputs["depth"] = p_depth; outputs["segmentation"] = p_seg

    # --- メトリクス（SSIM と CER 追加） ---
    from .application import _load_image
    origs = {
        "edge": _load_image(input_paths["edge"], "L"),
        "depth": _load_image(input_paths["depth"], "L"),
        "segmentation": _load_image(input_paths["segmentation"], "RGB"),
    }
    ids_true, pal_true, _ = _build_seg_ids_and_palette(_suppress_white_boundaries(origs["segmentation"], 250, 2), 250)

    # Edge
    e_gt = (origs["edge"] >= 128).astype(np.uint8) * 255
    e_rx = results["edge"]["img"].astype(np.uint8)
    edge_f1 = float(f1_binary_edge(e_gt, e_rx))
    edge_ssim = float(ssim(e_gt, e_rx, data_range=255))

    # Depth
    d_gt = origs["depth"].astype(np.uint8)
    d_rx = results["depth"]["img"].astype(np.uint8)
    depth_psnr = float(psnr(d_gt, d_rx, data_range=255))
    depth_ssim = float(ssim(d_gt, d_rx, data_range=255))

    # Segmentation
    rx_rgb = results["segmentation"]["img"].astype(np.int32)
    pal = pal_true.astype(np.int32)
    Hh, Ww, _ = rx_rgb.shape
    rx_flat = rx_rgb.reshape(-1, 3)
    d2 = ((rx_flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    ids_pred = np.argmin(d2, axis=1).reshape(Hh, Ww).astype(np.int64)
    seg_miou = float(miou_from_ids(ids_true.astype(np.int64), ids_pred, K=int(pal_true.shape[0])))

    # Text CER 近似
    with open(input_paths["text"], "r", encoding="utf-8") as f:
        orig_text = f.read()
    recv_text = results["text"]["text"]
    minlen = min(len(orig_text), len(recv_text))
    mism = sum(1 for i in range(minlen) if orig_text[i] != recv_text[i]) + abs(len(orig_text)-len(recv_text))
    text_cer = float(mism / max(1, len(orig_text)))

    # 事後 SNR（全体）
    all_real = []
    for m in MODS:
        if m == "text": continue
        # pay_cols は上で閉じてしまっているので簡易に画像からは取れない．
        # 代わりに per-mod の推定値の平均を “全体” として出す．
    snr_global = float(np.mean([snr_by_mod[k] for k in snr_by_mod.keys()]))

    metrics = {
        "edge_f1": edge_f1,
        "edge_ssim": edge_ssim,
        "depth_psnr": depth_psnr,
        "depth_ssim": depth_ssim,
        "seg_mIoU": seg_miou,
        "text_cer": text_cer,
        "snr_est_db_by_modality": snr_by_mod,
        "snr_est_db_overall": snr_global,
    }

    report = {
        "snr_db": cfg.chan.snr_db,
        "channel": cfg.chan.channel,
        "ofdm": {"n_fft": cfg.ofdm.n_fft, "cp_len": cfg.ofdm.cp_len, "used_subcarriers": cfg.ofdm.used_subcarriers},
        "mode": mode.upper(),
        "power_linear": weights,
        "power_preset": preset_name or "",
        "subcarrier_slices": {k: [v.start, v.stop] for k,v in sc_slices.items()},
        "inputs": input_paths,
        "outputs": outputs,
        "metrics": metrics,
        "crc_by_modality": {m: results[m]["crc_ok"] for m in MODS},
        "bit_order_by_modality": {m: results[m]["order"] for m in MODS},
        "tag": args.tag,
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Multimodal OFDM Report ===")
    print(f"Output dir: {out_dir}")
    print(f"SNR(dB): {cfg.chan.snr_db}  Channel: {cfg.chan.channel}")
    print(f"Mode: {mode.upper()}  Power: {weights}  Preset: {preset_name or '-'}")
    print(f"CRC: " + ", ".join([f"{m}:{'OK' if results[m]['crc_ok'] else 'NG'}({results[m]['order']})" for m in MODS]))
    print(f"Edge F1: {edge_f1:.3f}  SSIM: {edge_ssim:.3f} | Depth PSNR: {depth_psnr:.2f} dB  SSIM: {depth_ssim:.3f} | Seg mIoU: {seg_miou:.3f}")
    print(f"Text CER≈{text_cer:.4f} | SNR(est): {snr_global:.2f} dB  {snr_by_mod}")
    print(f"Files: {outputs}")

if __name__ == "__main__":
    main()
