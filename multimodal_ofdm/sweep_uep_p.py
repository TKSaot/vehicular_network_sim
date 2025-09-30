#!/usr/bin/env python3
"""
sweep_uep_p.py
Run fixed-bandwidth (UEP-P) experiments across SNR and RX profiles,
using only the specified presets, and write a summary CSV.

- Sequential execution (no race conditions)
- Works with the patched run_multimodal_ofdm.py (metrics gating etc.)
- Uses --ecc-profile to switch receiver denoising (none / rxstrong)
- Defaults to metrics=edge,depth,text (seg can be added later on finalists)
"""

import argparse, subprocess, sys, json, time, os, csv, shlex
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DEFAULT_PRESETS = {
    # EC = none
    1.0: ["eep", "snr1_none_base", "snr1_none_drop_seg", "snr1_none_geom_only"],
    6.0: ["eep", "snr6_none_base", "snr6_none_geom_plus"],
    12.0:["eep", "snr12_none_base", "snr12_none_textplus"],
}

RXSTRONG_PRESETS = {
    # EC = rxstrong
    1.0: ["eep", "snr1_rxstrong_base", "snr1_rxstrong_seglo"],
    6.0: ["eep", "snr6_rxstrong_base", "snr6_rxstrong_segx"],
    12.0:["eep", "snr12_rxstrong_base", "snr12_rxstrong_depthx"],
}

def build_cmd(pkg: str, snr: float, channel: str, preset: str,
              ecc_profile: str, examples_dir: str, payload_rep: str,
              metrics: str, out_root: Optional[str], extra_tag: str) -> List[str]:
    cmd = [
        sys.executable, "-m", f"{pkg}",
        "--snr_db", str(snr),
        "--channel", channel,
        "--power-preset", preset,
        "--payload-rep", payload_rep,
        "--metrics", metrics,
        "--ecc-profile", ecc_profile,
        "--tag", extra_tag
    ]
    if examples_dir:
        cmd += ["--examples-dir", examples_dir]
    if out_root:
        cmd += ["--out-root", out_root]
    return cmd

def parse_output_dir(proc_out: str) -> Optional[str]:
    for line in proc_out.splitlines():
        line = line.strip()
        if line.startswith("Output dir:"):
            return line.split("Output dir:", 1)[1].strip()
    return None

def read_report(report_path: Path) -> Dict:
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.environ.get("PKG", "multimodal_ofdm"), help="Python package/module containing run_multimodal_ofdm")
    ap.add_argument("--channel", choices=["rayleigh","awgn"], default=os.environ.get("CHAN","rayleigh"))
    ap.add_argument("--snr-list", default="1,6,12", help="Comma list of SNR dB values (e.g., '1,6,12')")
    ap.add_argument("--rx-profiles", default="none,rxstrong", help="Comma list among 'none,rxstrong'")
    ap.add_argument("--examples-dir", default=os.environ.get("EXDIR","examples"))
    ap.add_argument("--payload-rep", default=os.environ.get("FAIR","text=1,edge=1,depth=1,segmentation=1"))
    ap.add_argument("--metrics", default="edge,depth,text", help="Metrics to compute (avoid seg for speed; add later)")
    ap.add_argument("--out-root", default=None, help="Optional outputs root")
    ap.add_argument("--summary", default="uep_p_summary.csv", help="Where to write the CSV summary")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    args = ap.parse_args()

    pkg = args.pkg
    snrs = [float(s.strip()) for s in args.snr_list.split(",") if s.strip()]
    rx_profiles = [p.strip() for p in args.rx_profiles.split(",") if p.strip()]
    presets_map = {"none": DEFAULT_PRESETS, "rxstrong": RXSTRONG_PRESETS}

    # sanity check imports
    try:
        __import__(pkg)
    except Exception as e:
        print(f"[ERROR] Cannot import package '{pkg}': {e}", file=sys.stderr)
        sys.exit(2)

    rows = []
    start_time = time.time()

    for rxp in rx_profiles:
        if rxp not in presets_map:
            print(f"[WARN] Unknown rx profile '{rxp}', skipping", file=sys.stderr)
            continue
        per_snr = presets_map[rxp]
        for snr in snrs:
            preset_list = per_snr.get(float(snr))
            if not preset_list:
                print(f"[WARN] No preset list for rx='{rxp}' snr='{snr}'", file=sys.stderr)
                continue
            print(f"===== Running UEP-P | ecc-profile={rxp} | SNR={snr} dB | channel={args.channel} =====")
            for preset in preset_list:
                tag = f"uepp,rx={rxp},snr={snr},preset={preset}"
                cmd = build_cmd(pkg=pkg, snr=snr, channel=args.channel, preset=preset,
                                ecc_profile=rxp, examples_dir=args.examples_dir,
                                payload_rep=args.payload_rep, metrics=args.metrics,
                                out_root=args.out_root, extra_tag=tag)
                print(">>>", " ".join(shlex.quote(x) for x in cmd))
                if args.dry_run:
                    continue
                proc = subprocess.run(cmd, capture_output=True, text=True)
                sys.stdout.write(proc.stdout)
                sys.stderr.write(proc.stderr)
                if proc.returncode != 0:
                    rows.append({
                        "snr_db": snr, "channel": args.channel, "rx_profile": rxp, "preset": preset,
                        "status": f"ERROR({proc.returncode})", "report_path": "", "edge_f1": "", "depth_psnr": "",
                        "seg_mIoU": "", "text_cer": "", "crc_text": "", "crc_edge": "", "crc_depth": "", "crc_seg": ""
                    })
                    continue

                outdir = parse_output_dir(proc.stdout)
                if not outdir:
                    rows.append({
                        "snr_db": snr, "channel": args.channel, "rx_profile": rxp, "preset": preset,
                        "status": "NO_OUTDIR", "report_path": "", "edge_f1": "", "depth_psnr": "",
                        "seg_mIoU": "", "text_cer": "", "crc_text": "", "crc_edge": "", "crc_depth": "", "crc_seg": ""
                    })
                    continue

                report_path = Path(outdir) / "report.json"
                if not report_path.exists():
                    rows.append({
                        "snr_db": snr, "channel": args.channel, "rx_profile": rxp, "preset": preset,
                        "status": "NO_REPORT", "report_path": str(report_path), "edge_f1": "", "depth_psnr": "",
                        "seg_mIoU": "", "text_cer": "", "crc_text": "", "crc_edge": "", "crc_depth": "", "crc_seg": ""
                    })
                    continue

                rep = read_report(report_path)
                crc = rep.get("crc_by_modality", {})
                metrics = rep.get("metrics", {})
                rows.append({
                    "snr_db": rep.get("snr_db", snr),
                    "channel": rep.get("channel", args.channel),
                    "rx_profile": rxp,
                    "preset": rep.get("power_preset", preset),
                    "status": "OK",
                    "report_path": str(report_path),
                    "edge_f1": metrics.get("edge_f1", ""),
                    "depth_psnr": metrics.get("depth_psnr", ""),
                    "seg_mIoU": metrics.get("seg_mIoU", ""),
                    "text_cer": metrics.get("text_cer", ""),
                    "crc_text": crc.get("text", ""),
                    "crc_edge": crc.get("edge", ""),
                    "crc_depth": crc.get("depth", ""),
                    "crc_seg": crc.get("segmentation", ""),
                })

    # Write summary CSV
    out_csv = Path(args.summary)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["snr_db","channel","rx_profile","preset","status","report_path",
                  "edge_f1","depth_psnr","seg_mIoU","text_cer","crc_text","crc_edge","crc_depth","crc_seg"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    took = time.time() - start_time
    print(f"\n[Done] wrote {len(rows)} rows to {out_csv} in {took:.1f}s")

if __name__ == "__main__":
    main()
