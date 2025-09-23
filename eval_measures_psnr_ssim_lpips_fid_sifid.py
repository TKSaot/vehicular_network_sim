#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, sys, traceback
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
from scipy.linalg import sqrtm

# ---- Torch / LPIPS / Inception (optional) ----
try:
    import torch
    from torchvision import transforms
    from torchvision.models import inception_v3, Inception_V3_Weights
    import lpips  # type: ignore
    _TORCH_OK = True
except Exception:
    torch = None; transforms = None; inception_v3 = None; Inception_V3_Weights = None; lpips = None
    _TORCH_OK = False

# ---- I/O ----
def imread_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.asarray(img).astype(np.float32) / 255.0

def match_size(ref, img):
    if ref.shape[:2] == img.shape[:2]:
        return img
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    pil = pil.resize((ref.shape[1], ref.shape[0]), Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0

def to_gray(a):
    return np.dot(a[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)

def align_translation(ref, img, upsample_factor=10):
    try:
        shift_est, _, _ = phase_cross_correlation(to_gray(ref), to_gray(img), upsample_factor=upsample_factor)
        aligned = np.stack([ndi_shift(img[..., c], shift=shift_est, mode='reflect') for c in range(3)], axis=-1)
        return np.clip(aligned, 0.0, 1.0).astype(np.float32)
    except Exception:
        return img

# ---- Metrics ----
def rmse(ref, img): return float(np.sqrt(np.mean((ref - img) ** 2)))
def psnr_rgb(ref, img): return float(peak_signal_noise_ratio(ref, img, data_range=1.0))
def psnr_y(ref, img):
    y1, y2 = to_gray(ref), to_gray(img)
    return float(peak_signal_noise_ratio(y1, y2, data_range=1.0))
def ssim_rgb(ref, img):
    try: return float(structural_similarity(ref, img, channel_axis=2, data_range=1.0))
    except TypeError: return float(structural_similarity(ref, img, multichannel=True, data_range=1.0))

def setup_lpips(device: str = "cpu"):
    if not _TORCH_OK: return None, None
    try:
        net = lpips.LPIPS(net='alex').eval()
        dev = torch.device(device)
        net.to(dev)
        return net, dev
    except Exception:
        return None, None

def compute_lpips(lpips_net, device, ref, img):
    if lpips_net is None: return float('nan')
    ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float()*2-1
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()*2-1
    ref_t = ref_t.to(device)
    img_t = img_t.to(device)
    with torch.no_grad():
        d = lpips_net(ref_t, img_t)
    return float(d.item())

# ---- Inception features / FID ----
class InceptionPool3:
    def __init__(self, device='cpu'):
        assert _TORCH_OK, "PyTorch/torchvision が必要です．"
        w = Inception_V3_Weights.IMAGENET1K_V1
        # 重要：aux_logits=True（torchvision側がこの値を期待）
        self.model = inception_v3(weights=w, transform_input=False, aux_logits=True)
        self.model.eval().to(device); self.device = device; self._feat = None
        def _hook(m, inp, out): self._feat = out.detach().flatten(1)
        self.model.avgpool.register_forward_hook(_hook)
        mean, std = w.transforms().mean, w.transforms().std
        self.tx = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
        ])
    @torch.no_grad()
    def featurize_pil_batch(self, pil_list, batch_size=32):
        xs, feats = [], []
        for i, im in enumerate(pil_list):
            xs.append(self.tx(im))
            if len(xs) == batch_size or i == len(pil_list) - 1:
                x = torch.stack(xs, dim=0).to(self.device); _ = self.model(x)
                feats.append(self._feat.cpu().numpy().copy()); xs.clear()
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float32)

def sample_paired_crops(ref_img, rec_img, num_crops=256, crop_sizes=[64,96,128], rng=np.random.default_rng(1234), grid=False):
    W, H = ref_img.size; ref_patches = []; rec_patches = []
    if grid:
        step = min(crop_sizes); xs = list(range(0, max(W - step, 1), step)); ys = list(range(0, max(H - step, 1), step))
        coords = [(x, y, step) for y in ys for x in xs]; rng.shuffle(coords); coords = coords[:num_crops]
        for x, y, s in coords:
            x = min(max(x, 0), W - s); y = min(max(y, 0), H - s); box = (x, y, x + s, y + s)
            ref_patches.append(ref_img.crop(box)); rec_patches.append(rec_img.crop(box))
    else:
        for _ in range(num_crops):
            s = int(rng.choice(crop_sizes))
            if W < s or H < s: s = min(W, H)
            x = int(rng.integers(0, max(W - s, 1))); y = int(rng.integers(0, max(H - s, 1)))
            box = (x, y, x + s, y + s); ref_patches.append(ref_img.crop(box)); rec_patches.append(rec_img.crop(box))
    return ref_patches, rec_patches

def fid_from_gaussians(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    cov_mean = sqrtm((s1 + eps*np.eye(s1.shape[0])) @ (s2 + eps*np.eye(s2.shape[0])))
    if np.iscomplexobj(cov_mean): cov_mean = cov_mean.real
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2 * cov_mean))

def compute_fid_single_image(ref, img, inc: InceptionPool3):
    ref_pil = Image.fromarray((np.clip(ref,0,1)*255).astype(np.uint8))
    img_pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    f1 = inc.featurize_pil_batch([ref_pil]); f2 = inc.featurize_pil_batch([img_pil])
    mu1, mu2 = f1[0], f2[0]; z = np.zeros((mu1.size, mu1.size), dtype=np.float64)
    return fid_from_gaussians(mu1, z, mu2, z)

def compute_sifid(ref, img, inc: InceptionPool3, num_crops=256, crop_sizes=[64,96,128], seed=1234, grid=False):
    ref_pil = Image.fromarray((np.clip(ref,0,1)*255).astype(np.uint8))
    img_pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    rng = np.random.default_rng(seed)
    ref_crops, img_crops = sample_paired_crops(ref_pil, img_pil, num_crops=num_crops, crop_sizes=crop_sizes, rng=rng, grid=grid)
    f1 = inc.featurize_pil_batch(ref_crops, 32); f2 = inc.featurize_pil_batch(img_crops, 32)
    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0); c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    return fid_from_gaussians(mu1, c1, mu2, c2)

# ---- Label parsing ----
PRESET_KEYS = ["eep","boost_text_x2","boost_text_x4","boost_edge_x2","boost_depth_x2",
               "geom_pair_x2","text_edge_x2","text_depth_x2","low_seg","edge_x2","depth_x2","text_x2","text_x4"]

def parse_preset_and_snr(path_str, snr_keys=["1","6","12"], user_regex=None):
    base = path_str.lower(); preset = None
    for k in PRESET_KEYS:
        if k in base: preset = k; break
    snr = None; patterns = []
    if user_regex: patterns.append(user_regex)
    patterns += [
        r"snr[ _=-]?(\d+)",
        r"[\/_-]s(?:nr)?[ _=-]?(\d+)[\/_-]?",
        r"[\/_-](\d+)(?=\/|_|-|\.)",
    ]
    for pat in patterns:
        m = re.search(pat, base)
        if m:
            try:
                cand = int(m.group(1))
                if 0 <= cand <= 120: snr = cand; break
            except Exception:
                pass
    if snr is None:
        m2 = re.search(r"[_-](\d+)(?=\D|$)", base)
        if m2 and m2.group(1) in snr_keys: snr = int(m2.group(1))
    return preset, snr

# ---- Plotting ----
def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(); drew = False
    if "snr" in df.columns and df["snr"].notna().any():
        piv = df.dropna(subset=["snr", metric]).groupby(["snr","preset"], as_index=False)[metric].mean()
        if len(piv) > 0:
            snrs = sorted(piv["snr"].dropna().unique().tolist())
            presets = sorted(piv["preset"].dropna().unique().tolist())
            for p in presets:
                ys = [float(piv.loc[(piv["snr"]==s)&(piv["preset"]==p), metric].iloc[0]) if ((piv["snr"]==s)&(piv["preset"]==p)).any() else np.nan for s in snrs]
                plt.plot(snrs, ys, marker='o', label=p); drew = True
            plt.xlabel("SNR (dB)")
    if not drew:
        sub = df.dropna(subset=[metric]).copy()
        if sub.empty:
            plt.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        else:
            presets = sorted(sub["preset"].fillna("unknown").unique().tolist())
            for p in presets:
                y = sub.loc[sub["preset"].fillna("unknown")==p, metric].to_list()
                x = list(range(1, len(y)+1))
                plt.plot(x, y, marker='o', label=p)
            plt.xlabel("index")
    plt.grid(True); plt.ylabel(metric.upper()); plt.title(metric.upper()); plt.legend(loc="best")
    out_path = out_dir / f"{metric}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"[PLOT] {metric} -> {out_path}")
    plt.close()

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--original", required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--recons_glob"); g.add_argument("--recons_list")
    ap.add_argument("--out_csv", default="metrics_all.csv"); ap.add_argument("--out_dir", default="plots_all")
    ap.add_argument("--align", action="store_true"); ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--snr_regex", type=str, default=None)
    ap.add_argument("--compute_fid", action="store_true"); ap.add_argument("--fid_mode", type=str, default="image", choices=["image","patches"])
    ap.add_argument("--compute_sifid", action="store_true")
    ap.add_argument("--sifid_crops", type=int, default=256); ap.add_argument("--sifid_crop_sizes", type=str, default="64,96,128")
    ap.add_argument("--sifid_grid", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    import glob
    recons = sorted(glob.glob(args.recons_glob, recursive=True)) if args.recons_glob else [ln.strip() for ln in open(args.recons_list, "r", encoding="utf-8") if ln.strip()]
    print(f"[INFO] found {len(recons)} reconstructed images")
    if len(recons) == 0:
        print("[ERROR] No reconstructed images found. Check --recons_glob / --recons_list", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] plots will be saved under: {out_dir.resolve()}")

    ref = imread_rgb(args.original)

    # nets
    lpips_net, lpips_dev = setup_lpips(args.device)
    inc = None
    if _TORCH_OK and (args.compute_fid or args.compute_sifid):
        try:
            inc = InceptionPool3(device=args.device)
        except Exception:
            print("[ERROR] Inception initialization failed:\n" + traceback.format_exc(), file=sys.stderr)

    crop_sizes = [int(x) for x in args.sifid_crop_sizes.split(",")] if args.sifid_crop_sizes else [96]

    rows = []
    for i, p in enumerate(recons, 1):
        try:
            img = imread_rgb(p); img = match_size(ref, img)
            if args.no_align: pass
            elif args.align: img = align_translation(ref, img)

            row = {"file": p}
            row["preset"], row["snr"] = parse_preset_and_snr(p, user_regex=args.snr_regex)
            row["rmse"] = rmse(ref, img)
            row["psnr"] = psnr_rgb(ref, img)
            row["psnr_y"] = psnr_y(ref, img)
            row["ssim"] = ssim_rgb(ref, img)
            row["lpips"] = compute_lpips(lpips_net, lpips_dev, ref, img) if lpips_net is not None else float('nan')

            if inc is not None and args.compute_fid:
                if args.fid_mode == "image":
                    row["fid"] = compute_fid_single_image(ref, img, inc)
                else:
                    row["fid"] = compute_sifid(ref, img, inc, num_crops=args.sifid_crops, crop_sizes=crop_sizes, seed=1234, grid=bool(args.sifid_grid))
            if inc is not None and args.compute_sifid:
                row["sifid"] = compute_sifid(ref, img, inc, num_crops=args.sifid_crops, crop_sizes=crop_sizes, seed=1234, grid=bool(args.sifid_grid))

            rows.append(row)
            if i % 5 == 0 or i == len(recons):
                print(f"[INFO] processed {i}/{len(recons)} images")
        except Exception:
            print(f"[ERROR] while processing {p}\n{traceback.format_exc()}", file=sys.stderr)
            rows.append({"file": p, "error": "processing_failed"})

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] CSV -> {out_csv.resolve()}  rows={len(df)}  cols={list(df.columns)}")

    for m in ["rmse","psnr","psnr_y","ssim","lpips","fid","sifid"]:
        if m in df.columns:
            plot_metric(df, m, out_dir)

if __name__ == "__main__":
    main()
