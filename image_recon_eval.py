#!/usr/bin/env python3
"""
image_recon_eval.py

Compare one original image against many reconstructed images with a battery of
full-reference and task-leaning metrics suited for wireless/IVC + diffusion reconstructions.

Metrics implemented (no extra installs):
  - PSNR (RGB + Y/luma)
  - SSIM (single-scale)
  - CIEDE2000 (mean / 95th percentile)
  - Edge preservation: F1 score with 1px tolerance
  - Pratt's Figure of Merit (edge localization)

Optional (auto-detected if libraries are present):
  - LPIPS (requires: torch, lpips)
  - DISTS (requires: torch, dists-pytorch or 'DISTS_pytorch' from the official repo)
  - MS-SSIM (requires: torch, pytorch_msssim)

Extras:
  - Optional subpixel translation alignment via phase correlation (skimage.registration)
  - Batch processing via glob or a textfile list of paths
  - Regex-based parsing of preset type and SNR from filenames (customizable)
  - CSV output for analysis

Usage examples:
  python image_recon_eval.py --original path/to/original.png \
      --recons_glob "outputs/**/*.png" \
      --out_csv results.csv --align --compute_lpips --compute_dists

  # Or pass a list file (one path per line)
  python image_recon_eval.py --original original.png --recons_list paths.txt --out_csv results.csv

Author: (you)
License: MIT
"""
import argparse
import re
import sys
import math
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image

# Core metrics & helpers
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.feature import canny
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift, binary_dilation, distance_transform_edt

# Optional deep metrics (loaded lazily if available)
try:
    import torch
except Exception:
    torch = None

# --- Utility -----------------------------------------------------------------
def imread_rgb(path: str) -> np.ndarray:
    """Read image as float32 RGB in [0,1]."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def to_gray(arr: np.ndarray) -> np.ndarray:
    """RGB->gray luminance using BT.601 weights (approx)."""
    return np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

def match_size(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Resize img to ref size if necessary. Bilinear to minimize ringing.
    Note: resizing can bias pixel-wise metrics; prefer using same size renders.
    """
    if ref.shape[:2] == img.shape[:2]:
        return img
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    pil = pil.resize((ref.shape[1], ref.shape[0]), Image.BILINEAR)
    return np.asarray(pil).astype(np.float32)/255.0

def align_translation(ref: np.ndarray, img: np.ndarray, upsample_factor: int = 10) -> np.ndarray:
    """Subpixel translation alignment via phase correlation on grayscale images.
    Returns shifted image; only shift (no rotation/scale).
    """
    ref_g = to_gray(ref)
    img_g = to_gray(img)
    try:
        shift_est, error, diffphase = phase_cross_correlation(ref_g, img_g, upsample_factor=upsample_factor)
        # shift_est is (dy, dx); apply to all channels with edge mode='reflect'
        aligned = np.stack([ndi_shift(img[..., c], shift=shift_est, mode='reflect') for c in range(3)], axis=-1)
        aligned = np.clip(aligned, 0.0, 1.0).astype(np.float32)
        return aligned
    except Exception:
        return img

# --- Metrics -----------------------------------------------------------------
def psnr_rgb(ref: np.ndarray, img: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, img, data_range=1.0))

def psnr_y(ref: np.ndarray, img: np.ndarray) -> float:
    ref_y = to_gray(ref)
    img_y = to_gray(img)
    return float(peak_signal_noise_ratio(ref_y, img_y, data_range=1.0))

def ssim_rgb(ref: np.ndarray, img: np.ndarray) -> float:
    # structural_similarity returns mean over image; channel_axis for skimage>=0.19
    try:
        return float(structural_similarity(ref, img, channel_axis=2, data_range=1.0))
    except TypeError:
        # Fallback for older skimage
        return float(structural_similarity(ref, img, multichannel=True, data_range=1.0))

def ciede2000_stats(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    """Return mean and 95th percentile of ΔE2000 (lower is better)."""
    ref_lab = color.rgb2lab(ref)
    img_lab = color.rgb2lab(img)
    try:
        delta = color.deltaE_ciede2000(ref_lab, img_lab)
    except Exception:
        # old skimage may not have ciede2000; fall back to simpler metric
        delta = color.deltaE_cie76(ref_lab, img_lab)
    delta = delta.astype(np.float32)
    mean = float(np.mean(delta))
    p95 = float(np.percentile(delta, 95.0))
    return mean, p95

def edges_f1_and_pratt(ref: np.ndarray, img: np.ndarray, sigma: float = 1.0, tol: int = 1) -> Tuple[float, float]:
    """Compute edge F1 (with tolerance) and Pratt's FOM (β=1/9) using canny edges on gray images.
    - F1 is computed by dilating the reference edges by 'tol' pixels to allow small misalignments.
    - Pratt's FOM penalizes localization error quadratically; ranges ~[0,1]. Higher is better.
    """
    ref_e = canny(to_gray(ref), sigma=sigma)
    img_e = canny(to_gray(img), sigma=sigma)

    if tol > 0:
        selem = np.ones((2*tol+1, 2*tol+1), dtype=bool)
        ref_dil = binary_dilation(ref_e, structure=selem)
        img_dil = binary_dilation(img_e, structure=selem)
    else:
        ref_dil = ref_e
        img_dil = img_e

    # F1 with tolerant matching (symmetrized)
    tp1 = np.logical_and(img_e, ref_dil).sum()
    tp2 = np.logical_and(ref_e, img_dil).sum()
    # Average the two "true positives" to reduce asymmetry
    tp = 0.5 * (tp1 + tp2)
    fp = np.logical_and(img_e, np.logical_not(ref_dil)).sum()
    fn = np.logical_and(ref_e, np.logical_not(img_dil)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Pratt's FOM: distance from img edges to nearest ref edge
    # Use distance transform on ref edges complement to get nearest-edge distance.
    # Typical β = 1/9 to strongly penalize >3px errors.
    beta = 1.0 / 9.0
    # distance transform needs distance to nearest True; compute on inverted ref edges
    dist = distance_transform_edt(~ref_e)
    # For each detected edge pixel in img_e, get its distance to nearest ref edge (in pixels)
    di = dist[img_e]
    if di.size == 0:
        pratt = 0.0
    else:
        # Normalize by max(N_ref, N_img) as in standard definition
        denom = max(ref_e.sum(), img_e.sum(), 1)
        pratt = float(np.sum(1.0 / (1.0 + beta * (di ** 2))) / denom)

    return float(f1), float(pratt)

# Optional Deep Metrics --------------------------------------------------------
def maybe_lpips():
    if torch is None:
        return None
    try:
        import lpips  # type: ignore
        net = lpips.LPIPS(net='alex')  # default
        net.eval()
        return net
    except Exception:
        return None

def compute_lpips(lpips_net, ref: np.ndarray, img: np.ndarray) -> float:
    """LPIPS expects [-1,1] normalized torch tensors in NCHW."""
    if lpips_net is None:
        return float('nan')
    ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float() * 2 - 1
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        d = lpips_net(ref_t, img_t)
    return float(d.item())

def maybe_dists():
    if torch is None:
        return None
    try:
        # Two common module names seen in the wild
        try:
            import DISTS_pytorch as DISTS  # official repo name
        except Exception:
            import dists_pytorch as DISTS  # pip name fallback
        model = DISTS.DISTS()
        model.eval()
        return model
    except Exception:
        return None

def compute_dists(dists_net, ref: np.ndarray, img: np.ndarray) -> float:
    if dists_net is None:
        return float('nan')
    # DISTS expects [0,1] torch tensors in NCHW
    ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float()
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        d = dists_net(ref_t, img_t)
    return float(d.item())

def maybe_msssim():
    if torch is None:
        return None
    try:
        from pytorch_msssim import ms_ssim  # type: ignore
        return ms_ssim
    except Exception:
        return None

def compute_msssim(ms_ssim_fn, ref: np.ndarray, img: np.ndarray) -> float:
    if ms_ssim_fn is None:
        return float('nan')
    ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float()
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    # data_range=1.0, channel=3
    with torch.no_grad():
        val = ms_ssim_fn(ref_t, img_t, data_range=1.0, size_average=True)
    return float(val.item())

# Filename parsing -------------------------------------------------------------
PRESET_KEYS = [
    "eep",
    "boost_text_x2","boost_text_x4","boost_edge_x2","boost_depth_x2",
    "geom_pair_x2","text_edge_x2","text_depth_x2",
    "low_seg",
]

def parse_preset_and_snr(fname: str, snr_keys: List[str] = ["1","6","12"]) -> Tuple[Optional[str], Optional[int]]:
    base = fname.lower()
    preset = None
    for k in PRESET_KEYS:
        if k in base:
            preset = k
            break
    # try generic regex snr(\d+)
    snr = None
    m = re.search(r"snr[ _-]?(\d+)", base)
    if m:
        try:
            snr = int(m.group(1))
        except Exception:
            snr = None
    # fallback: match standalone _1 _6 _12 or -1 -6 -12
    if snr is None:
        m2 = re.search(r"[_-](\d+)(?=\D|$)", base)
        if m2 and m2.group(1) in snr_keys:
            snr = int(m2.group(1))
    return preset, snr

# Orchestrator -----------------------------------------------------------------
@dataclass
class Config:
    original: str
    recons: List[str]
    out_csv: str
    align: bool = False
    compute_lpips: bool = False
    compute_dists: bool = False
    compute_msssim: bool = False
    edge_sigma: float = 1.0
    edge_tol: int = 1

def evaluate(cfg: Config):
    import pandas as pd

    ref = imread_rgb(cfg.original)

    # Prepare optional nets
    lpips_net = maybe_lpips() if cfg.compute_lpips else None
    dists_net = maybe_dists() if cfg.compute_dists else None
    ms_ssim_fn = maybe_msssim() if cfg.compute_msssim else None

    rows = []
    for p in cfg.recons:
        try:
            img = imread_rgb(p)
            img = match_size(ref, img)
            if cfg.align:
                img = align_translation(ref, img, upsample_factor=10)

            # Metrics
            m_psnr = psnr_rgb(ref, img)
            m_psnr_y = psnr_y(ref, img)
            m_ssim = ssim_rgb(ref, img)
            de_mean, de_p95 = ciede2000_stats(ref, img)
            edge_f1, pratt = edges_f1_and_pratt(ref, img, sigma=cfg.edge_sigma, tol=cfg.edge_tol)

            row = {
                "file": p,
                "preset": None,
                "snr": None,
                "psnr_rgb": m_psnr,
                "psnr_y": m_psnr_y,
                "ssim": m_ssim,
                "deltaE00_mean": de_mean,
                "deltaE00_p95": de_p95,
                "edge_F1_tol": edge_f1,
                "pratt_FOM": pratt,
            }

            if lpips_net is not None:
                row["lpips"] = compute_lpips(lpips_net, ref, img)
            if dists_net is not None:
                row["dists"] = compute_dists(dists_net, ref, img)
            if ms_ssim_fn is not None:
                row["ms_ssim"] = compute_msssim(ms_ssim_fn, ref, img)

            # parse labels
            preset, snr = parse_preset_and_snr(Path(p).name)
            row["preset"] = preset
            row["snr"] = snr

            rows.append(row)
        except Exception as ex:
            rows.append({
                "file": p, "error": str(ex)
            })

    df = pd.DataFrame(rows)

    # Helpful derived ranks (lower-is-better for some metrics)
    def add_rank_cols(df):
        rank_specs = [
            ("psnr_rgb", False),  # higher better
            ("psnr_y", False),
            ("ssim", False),
            ("ms_ssim", False),
            ("lpips", True),
            ("dists", True),
            ("deltaE00_mean", True),
            ("edge_F1_tol", False),
            ("pratt_FOM", False),
        ]
        for col, lower_better in rank_specs:
            if col in df.columns:
                df[f"rank_{col}"] = df[col].rank(ascending=lower_better, method="min")
        # quick combined rank over available rank_* columns
        rank_cols = [c for c in df.columns if c.startswith("rank_")]
        if rank_cols:
            df["rank_avg"] = df[rank_cols].mean(axis=1)
        return df

    df = add_rank_cols(df)

    # Save
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote metrics to {out_path.resolve()}")
    # Also print a small leaderboard
    safe_cols = [c for c in ["file","preset","snr","psnr_rgb","ssim","lpips","dists","deltaE00_mean","edge_F1_tol","pratt_FOM","rank_avg"] if c in df.columns]
    with pd.option_context('display.max_colwidth', 120):
        print(df.sort_values(by=[c for c in ["rank_avg","lpips","dists","ssim","psnr_rgb"] if c in df.columns],
                             ascending=[True, True, True, False, False]).head(10)[safe_cols])

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--original", required=True, help="Path to the original reference image.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recons_glob", help="Glob for reconstructed images, e.g., 'outputs/**/*.png'")
    g.add_argument("--recons_list", help="Text file with one reconstructed image path per line.")
    ap.add_argument("--out_csv", default="recon_metrics.csv", help="Where to save the CSV of metrics.")
    ap.add_argument("--align", action="store_true", help="Apply subpixel translation alignment before metrics.")
    ap.add_argument("--edge_sigma", type=float, default=1.0, help="Canny sigma for edge maps.")
    ap.add_argument("--edge_tol", type=int, default=1, help="Pixel tolerance for edge F1 (via dilation).")
    ap.add_argument("--compute_lpips", action="store_true", help="Compute LPIPS if 'lpips' and torch are installed.")
    ap.add_argument("--compute_dists", action="store_true", help="Compute DISTS if 'DISTS_pytorch' or 'dists-pytorch' is installed.")
    ap.add_argument("--compute_msssim", action="store_true", help="Compute MS-SSIM if 'pytorch_msssim' is installed.")
    args = ap.parse_args()

    # Collect reconstructed paths
    recons: List[str] = []
    if args.recons_glob:
        import glob
        recons = sorted(glob.glob(args.recons_glob, recursive=True))
    else:
        with open(args.recons_list, "r", encoding="utf-8") as f:
            recons = [ln.strip() for ln in f if ln.strip()]
    if len(recons) == 0:
        print("No reconstructed images found. Check your glob/list.", file=sys.stderr)
        sys.exit(2)

    cfg = Config(
        original=args.original,
        recons=recons,
        out_csv=args.out_csv,
        align=bool(args.align),
        compute_lpips=bool(args.compute_lpips),
        compute_dists=bool(args.compute_dists),
        compute_msssim=bool(args.compute_msssim),
        edge_sigma=float(args.edge_sigma),
        edge_tol=int(args.edge_tol),
    )
    evaluate(cfg)

if __name__ == "__main__":
    main()
