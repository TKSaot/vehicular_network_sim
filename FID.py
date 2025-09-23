
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FID.py ー FID vs SNR の描画（eval_psnr_ssim_lpips.py と質感を揃えた版）

- FID.csv（ワイド形式 or ロング形式）を読み込み，プリセットごとの折れ線グラフを出力．
- グリッド，凡例，保存画質（dpi），tight_layout などの見た目を eval_psnr_ssim_lpips.py に寄せています．
- 列名が "Unnamed: *" の並びにも対応．

使い方：
  python FID.py --csv FID.csv --out fid_vs_snr.png --title "FID vs SNR"

依存：pandas, matplotlib, numpy
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_long_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    # 「Unnamed」でない列をプリセット開始地点とみなす（先頭のindex列は除外）
    preset_idx = [i for i,c in enumerate(cols) if not str(c).lower().startswith("unnamed") and i != 0]
    preset_names = [cols[i] for i in preset_idx]
    if len(df) < 2 or len(preset_idx) == 0:
        raise SystemExit("FID.csv の形式を解釈できません．(行が2未満 / 有効な列名が無い)")

    snr_row = df.iloc[0]
    fid_row = df.iloc[1]

    recs = []
    for idx, name in zip(preset_idx, preset_names):
        end = next((j for j in preset_idx if j > idx), len(cols))
        snrs = pd.to_numeric(snr_row.iloc[idx:end], errors="coerce").tolist()
        fids = pd.to_numeric(fid_row.iloc[idx:end], errors="coerce").tolist()
        for s, f in zip(snrs, fids):
            if np.isfinite(s) and np.isfinite(f):
                recs.append({"preset": str(name), "snr": float(s), "fid": float(f)})
    return pd.DataFrame(recs)

def normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    # すでにロング形式ならそのまま正規化
    cols = [c.lower() for c in df.columns]
    if "fid" in cols and ("snr" in cols) and ("preset" in cols):
        return df.rename(columns={df.columns[cols.index("fid")]:"fid",
                                  df.columns[cols.index("snr")]:"snr",
                                  df.columns[cols.index("preset")]:"preset"})[["snr","preset","fid"]]
    # そうでなければワイド→ロングへ
    return load_long_from_wide(df)

def plot_fid(long: pd.DataFrame, out_path: str, title: str = "FID vs SNR"):
    # スタイルを eval_psnr_ssim_lpips.py に合わせる
    # - grid on, bbox_inches="tight", dpi=200, マーカーo, 太さ1.8, 凡例best
    long = long.dropna(subset=["snr","fid","preset"]).copy()
    if long.empty:
        raise SystemExit("入力に有効なデータがありません．")

    snrs = sorted(long["snr"].unique().tolist())
    presets = sorted(long["preset"].unique().tolist())

    plt.figure(figsize=(7.5, 5.0))
    for p in presets:
        g = long[long["preset"] == p].sort_values("snr")
        plt.plot(g["snr"], g["fid"], marker="o", linewidth=1.8, label=p)

    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("FID (lower is better)")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved plot -> {out_path}")

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", default="FID.csv")
    ap.add_argument("--out", default="fid_vs_snr.png")
    ap.add_argument("--title", default="FID vs SNR")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    long = normalize_long(df)
    plot_fid(long, args.out, args.title)

if __name__ == "__main__":
    main()
