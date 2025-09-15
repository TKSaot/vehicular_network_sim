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

def run_once(modality: str, snr: float, channel: str, device: str, show_inner: bool, uep_mode: str) -> int:
    env = os.environ.copy()
    env["VC_USE_CUDA"] = "1" if device == "gpu" else "0"
    env["UEP_MODE"] = uep_mode.upper()
    if show_inner:
        env["TX_TQDM"] = "1"; env["RX_TQDM"] = "1"
    else:
        env.pop("TX_TQDM", None); env.pop("RX_TQDM", None)

    print(f"\n=== Run: {modality:>12s} @ {snr:>4.1f} dB | {channel} | device={device.upper()} | UEP={uep_mode} ===")
    t0 = time.time()
    rc = _run_text(snr, channel, env) if modality == "text" else _run_image(modality, snr, channel, env)
    dt = time.time() - t0
    print(f"--- finished ({modality}@{snr} dB, UEP={uep_mode}) in {dt:.1f}s, rc={rc} ---")
    return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[1.0, 4.0, 8.0, 12.0])
    ap.add_argument("--modalities", type=str, nargs="+", default=["edge","depth","segmentation","text"])
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default="rayleigh")
    ap.add_argument("--device", type=str, choices=["gpu","cpu"], default="cpu")
    ap.add_argument("--show_inner", action="store_true", help="TX/RX のフレーム単位 tqdm を表示")
    ap.add_argument("--uep", type=str, choices=["off","S","G","B"], default="off",
                    help="UEP policy: off=EEP baseline, S=Structure-first, G=Geometry-first, B=Balanced")
    args = ap.parse_args()

    rc_total = 0
    for m in args.modalities:
        for snr in args.snrs:
            rc_total |= run_once(m, snr, args.channel, args.device, args.show_inner, args.uep)
    sys.exit(rc_total)

if __name__ == "__main__":
    main()
