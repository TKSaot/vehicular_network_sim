#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_psnr_ssim_lpips.py（拡張・不具合修正版）

- 入力1枚（original）と複数の再構成画像（recons）を比較し，
  PSNR，SSIM，LPIPS，RMSE，FSIM（任意），FID 1枚版近似（任意）を算出．
- CSVと各指標のグラフ（SNR-軸，presetごと）を出力．
- ファイル名から preset と snr を堅牢に抽出：
    * 既存の preset に加え，depth_x2／edge_x2／text_x2／text_x4 を正式対応
    * edgex2／depthx2／textx2／textx4 などアンダースコア無し表記も拾う
    * 未検出時は "snr" 以前のトークンを自動で preset 候補として採用（救済）
- 画像サイズは original に合わせてリサイズ（バイリニア）
- 軽微な平行移動ズレは phase correlation で自動補正（--align 指定時）

依存：
  pip install numpy pillow scikit-image pandas matplotlib torch torchvision lpips scipy
  # FSIM を使う場合（推奨）
  pip install piq
"""
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
import matplotlib.pyplot as plt

# --- LPIPS（任意） -----------------------------------------------------------
try:
    import torch
    import lpips  # type: ignore
    _LPIPS_AVAILABLE = True
except Exception:
    torch = None
    lpips = None
    _LPIPS_AVAILABLE = False

# --- FSIM（任意：piq） -------------------------------------------------------
try:
    import piq  # type: ignore
    _FSIM_AVAILABLE = True
except Exception:
    piq = None
    _FSIM_AVAILABLE = False

# --- torchvision / InceptionV3（FID近似で使用） ------------------------------
_TV_AVAILABLE = False
_INCEPTION_READY = False
_INCEPTION_TRANSFORM = None
_INCEPTION_MODEL = None
_DEVICE = None

def _setup_inception():
    """InceptionV3 の pool3 特徴取得を準備（成功しなければ FID≈NaN）"""
    global _TV_AVAILABLE, _INCEPTION_READY, _INCEPTION_TRANSFORM, _INCEPTION_MODEL, _DEVICE
    if torch is None:
        return
    try:
        import torchvision
        from torchvision.models import inception_v3
        _TV_AVAILABLE = True
    except Exception:
        _TV_AVAILABLE = False
        return

    try:
        # 新API（PyTorch 2.0+）
        from torchvision.models import Inception_V3_Weights  # type: ignore
        weights = Inception_V3_Weights.DEFAULT
        _INCEPTION_TRANSFORM = weights.transforms()
        model = inception_v3(weights=weights, aux_logits=False, transform_input=False)
    except Exception:
        # 旧APIフォールバック
        try:
            from torchvision.models import inception_v3
            import torchvision.transforms as T
            _INCEPTION_TRANSFORM = T.Compose([
                T.Resize((299, 299), interpolation=Image.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            model = inception_v3(pretrained=True, aux_logits=False, transform_input=False)
        except Exception:
            return

    model.eval()
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_DEVICE)

    feats_holder = []
    def _hook(module, inp, out):
        feats_holder.append(out)
    handle = model.avgpool.register_forward_hook(_hook)

    def _get_feats(img_np: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
        x = _INCEPTION_TRANSFORM(pil).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            feats_holder.clear()
            _ = model(x)
            if not feats_holder:
                return np.full((2048,), np.nan, dtype=np.float32)
            f = feats_holder[-1]              # (1,2048,1,1)
            f = torch.flatten(f, 1)[0]        # (2048,)
            return f.detach().cpu().numpy().astype(np.float32)

    _INCEPTION_MODEL = (model, handle, _get_feats)
    _INCEPTION_READY = True

# --- I/O & 前処理 ------------------------------------------------------------
def imread_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.asarray(img).astype(np.float32) / 255.0

def match_size(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    if ref.shape[:2] == img.shape[:2]:
        return img
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    pil = pil.resize((ref.shape[1], ref.shape[0]), Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0

def to_gray(arr: np.ndarray) -> np.ndarray:
    return np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

def align_translation(ref: np.ndarray, img: np.ndarray, upsample_factor: int = 10) -> np.ndarray:
    ref_g = to_gray(ref)
    img_g = to_gray(img)
    try:
        shift_est, _, _ = phase_cross_correlation(ref_g, img_g, upsample_factor=upsample_factor)
        aligned = np.stack([ndi_shift(img[..., c], shift=shift_est, mode='reflect') for c in range(3)], axis=-1)
        return np.clip(aligned, 0.0, 1.0).astype(np.float32)
    except Exception:
        return img

# --- 指標 --------------------------------------------------------------------
def psnr_rgb(ref: np.ndarray, img: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, img, data_range=1.0))

def ssim_rgb(ref: np.ndarray, img: np.ndarray) -> float:
    try:
        return float(structural_similarity(ref, img, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(structural_similarity(ref, img, multichannel=True, data_range=1.0))

def rmse_rgb(ref: np.ndarray, img: np.ndarray) -> float:
    d = ref - img
    return float(np.sqrt(np.mean(d * d)))

def setup_lpips():
    if not _LPIPS_AVAILABLE:
        return None
    try:
        net = lpips.LPIPS(net='alex')
        net.eval()
        return net
    except Exception:
        return None

def compute_lpips(lpips_net, ref: np.ndarray, img: np.ndarray) -> float:
    if lpips_net is None or torch is None:
        return float('nan')
    ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float() * 2 - 1
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        d = lpips_net(ref_t, img_t)
    return float(d.item())

def compute_fsim(ref: np.ndarray, img: np.ndarray) -> float:
    if not _FSIM_AVAILABLE or torch is None:
        return float('nan')
    try:
        ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float()
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_t = ref_t.to(device)
        img_t = img_t.to(device)
        with torch.no_grad():
            val = piq.fsim(ref_t, img_t, reduction='none')
        return float(val.mean().item())
    except Exception:
        return float('nan')

def setup_inception_once():
    if not _INCEPTION_READY:
        _setup_inception()

def compute_fid_single_approx(ref: np.ndarray, img: np.ndarray) -> float:
    """FID（集合）ではなく，1枚どうしの近似：||f_ref - f_img||^2（fはInceptionV3 pool3特徴）"""
    if not _INCEPTION_READY:
        return float('nan')
    try:
        _, _, get_feats = _INCEPTION_MODEL
        f_ref = get_feats(ref)
        f_img = get_feats(img)
        if np.any(np.isnan(f_ref)) or np.any(np.isnan(f_img)):
            return float('nan')
        diff = f_ref - f_img
        return float(np.dot(diff, diff))
    except Exception:
        return float('nan')

# --- preset 抽出（拡張・頑健化） --------------------------------------------
# 既知のプリセットとエイリアス（→正規化先）
PRESET_ALIASES: Dict[str, List[str]] = {
    # 既存
    "eep": ["eep"],
    "boost_text_x2": ["boost_text_x2"],
    "boost_text_x4": ["boost_text_x4"],
    "boost_edge_x2": ["boost_edge_x2"],
    "boost_depth_x2": ["boost_depth_x2"],
    "geom_pair_x2": ["geom_pair_x2"],
    "text_edge_x2": ["text_edge_x2"],
    "text_depth_x2": ["text_depth_x2"],
    "low_seg": ["low_seg"],
    # 今回要望
    "edge_x2":  ["edge_x2", "edgex2"],
    "depth_x2": ["depth_x2", "depthx2"],
    "text_x2":  ["text_x2", "textx2"],
    "text_x4":  ["text_x4", "textx4"],
}

def parse_preset_and_snr(fname: str, snr_keys: List[str] = ["1","6","12"]) -> Tuple[Optional[str], Optional[int]]:
    base = fname.lower()
    # まずは既知エイリアスでヒットさせて正規化
    for canonical, aliases in PRESET_ALIASES.items():
        for al in aliases:
            if al in base:
                preset = canonical
                break
        else:
            continue
        break
    else:
        # 予備： 'snr'より前をpreset候補として救済（例：depth_x2_snr6 → depth_x2）
        mhead = re.split(r"snr[ _-]?", base, maxsplit=1)
        preset = mhead[0].rstrip("_-") if mhead and mhead[0] else None
        if not preset:
            preset = None

    # SNR を抽出
    snr = None
    m = re.search(r"snr[ _-]?(\d+)", base)
    if m:
        try:
            snr = int(m.group(1))
        except Exception:
            snr = None
    if snr is None:
        m2 = re.search(r"[_-](\d+)(?=\D|$)", base)
        if m2 and m2.group(1) in snr_keys:
            snr = int(m2.group(1))
    return preset, snr

# --- 可視化・サマリ ----------------------------------------------------------
def plot_metric_vs_snr(df: pd.DataFrame, metric: str, out_dir: Path):
    have = [metric, "snr", "preset"]
    if any(h not in df.columns for h in have):
        return
    sub = df.dropna(subset=[metric, "snr", "preset"]).copy()
    if sub.empty:
        return
    piv = sub.groupby(["snr","preset"], as_index=False)[metric].mean()
    snrs = sorted(piv["snr"].dropna().unique().tolist())
    presets = sorted(piv["preset"].dropna().unique().tolist())

    plt.figure()
    for p in presets:
        ys = []
        for s in snrs:
            filt = (piv["snr"]==s) & (piv["preset"]==p)
            ys.append(float(piv.loc[filt, metric].iloc[0]) if filt.any() else np.nan)
        plt.plot(snrs, ys, marker='o', label=p)
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs SNR")
    plt.legend(loc="best")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{metric}_vs_snr.png", bbox_inches="tight", dpi=200)
    plt.close()

def save_summaries(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ["psnr","ssim","lpips","rmse","fsim","fid"] if m in df.columns]
    if not metrics:
        return
    if "preset" in df.columns and "snr" in df.columns:
        grp = df.dropna(subset=["preset","snr"]).groupby(["preset","snr"], as_index=False)[metrics].mean()
        grp.sort_values(["preset","snr"], inplace=True)
        grp.to_csv(out_dir / "summary_by_preset_snr.csv", index=False)
    if "preset" in df.columns:
        grp_p = df.dropna(subset=["preset"]).groupby(["preset"], as_index=False)[metrics].mean()
        grp_p.sort_values(["preset"], inplace=True)
        grp_p.to_csv(out_dir / "summary_by_preset.csv", index=False)
    if "snr" in df.columns:
        grp_s = df.dropna(subset=["snr"]).groupby(["snr"], as_index=False)[metrics].mean()
        grp_s.sort_values(["snr"], inplace=True)
        grp_s.to_csv(out_dir / "summary_by_snr.csv", index=False)

# --- メイン -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--original", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recons_glob")
    g.add_argument("--recons_list")
    ap.add_argument("--out_csv", default="metrics_psnr_ssim_lpips.csv")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--out_dir", default="plots_psnr_ssim_lpips", help="プロット画像の出力先")
    args = ap.parse_args()

    import glob
    if args.recons_glob:
        recons = sorted(glob.glob(args.recons_glob, recursive=True))
    else:
        with open(args.recons_list, "r", encoding="utf-8") as f:
            recons = [ln.strip() for ln in f if ln.strip()]
    if len(recons) == 0:
        raise SystemExit("No reconstructed images found. Check your glob/list.")

    ref = imread_rgb(args.original)

    # オプション指標の初期化
    lpips_net = setup_lpips()   # LPIPS（なければ NaN）
    setup_inception_once()      # FID近似（準備失敗なら NaN）

    rows = []
    for p in recons:
        try:
            img = imread_rgb(p)
            img = match_size(ref, img)
            if args.align:
                img = align_translation(ref, img)

            m_psnr  = psnr_rgb(ref, img)
            m_ssim  = ssim_rgb(ref, img)
            m_lpips = compute_lpips(lpips_net, ref, img) if lpips_net is not None else float('nan')
            m_rmse  = rmse_rgb(ref, img)
            m_fsim  = compute_fsim(ref, img)   # piq 未導入なら NaN
            m_fid   = compute_fid_single_approx(ref, img)  # Inception 未導入/無通信なら NaN

            preset, snr = parse_preset_and_snr(Path(p).name)
            rows.append({
                "file": p,
                "preset": preset,
                "snr": snr,
                "psnr": m_psnr,
                "ssim": m_ssim,
                "lpips": m_lpips,
                "rmse": m_rmse,
                "fsim": m_fsim,
                "fid": m_fid,
            })
        except Exception as ex:
            rows.append({"file": p, "error": str(ex)})

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] CSVを書き出し：{out_csv.resolve()}")

    # 何が検出されたかの可視ログ（デバッグ寄り）
    if "preset" in df.columns:
        print("[INFO] 検出された preset 一覧：", sorted(df["preset"].dropna().unique().tolist()))
    if "snr" in df.columns:
        print("[INFO] 検出された SNR 一覧：", sorted(df["snr"].dropna().unique().tolist()))

    out_dir = Path(args.out_dir)
    for metric in ["psnr","ssim","lpips","rmse","fsim","fid"]:
        if metric in df.columns:
            plot_metric_vs_snr(df, metric, out_dir)

    save_summaries(df, out_dir)
    print(f"[OK] グラフとサマリを保存：{out_dir.resolve()}")

if __name__ == "__main__":
    main()
