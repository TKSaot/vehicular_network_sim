# examples/run_sweep.py
from __future__ import annotations
import os, sys, time, argparse, subprocess

def _run_image(modality: str, snr: float, channel: str, env: dict) -> int:
    cmd = [sys.executable, "-u", "examples/simulate_image_transmission.py",
           "--modality", modality, "--channel", channel, "--snr_db", str(snr)]
    return subprocess.run(cmd, env=env).returncode

def _run_text(snr: float, channel: str, env: dict) -> int:
    cmd = [sys.executable, "-u", "examples/simulate_text_transmission.py",
           "--channel", channel, "--snr_db", str(snr)]
    return subprocess.run(cmd, env=env).returncode

def run_once(modality: str, snr: float, channel: str, use_gpu: bool, show_inner: bool) -> int:
    env = os.environ.copy()
    env["VC_USE_CUDA"] = "1" if use_gpu else "0"
    if show_inner:
        env["TX_TQDM"] = "1"; env["RX_TQDM"] = "1"
    else:
        env.pop("TX_TQDM", None); env.pop("RX_TQDM", None)

    print(f"\n=== Run: {modality:>12s} @ {snr:>4.1f} dB | {channel} | device={'GPU' if use_gpu else 'CPU'} ===")
    t0 = time.time()
    if modality == "text":
        rc = _run_text(snr, channel, env)
    else:
        rc = _run_image(modality, snr, channel, env)
    dt = time.time() - t0
    print(f"--- finished ({modality}@{snr} dB) in {dt:.1f}s, rc={rc} ---")
    return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[1.0, 12.0])
    # ★ 既定に text を含める
    ap.add_argument("--modalities", type=str, nargs="+", default=["edge","depth","segmentation","text"])
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default="rayleigh")
    ap.add_argument("--device", type=str, choices=["gpu","cpu"], default="cpu")
    ap.add_argument("--show_inner", action="store_true", help="TX/RX のフレーム単位 tqdm を表示")
    args = ap.parse_args()

    use_gpu = (args.device == "gpu")
    rc_total = 0
    for m in args.modalities:
        for snr in args.snrs:
            rc_total |= run_once(m, snr, args.channel, use_gpu, args.show_inner)
    sys.exit(rc_total)

if __name__ == "__main__":
    main()
