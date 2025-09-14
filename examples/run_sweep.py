# examples/run_sweep.py
from __future__ import annotations
import os, sys, time, argparse, subprocess
from tqdm.auto import tqdm

def _run(kind: str, modality: str | None, snr: float, channel: str, extra_env: dict[str,str] | None) -> int:
    py = sys.executable
    env = os.environ.copy()
    env["VC_USE_CUDA"] = env.get("VC_USE_CUDA", "1")  # GPU 既定ON
    if extra_env:
        env.update(extra_env)
    if kind == "image":
        cmd = [py, "-u", "examples/simulate_image_transmission.py",
               "--modality", modality, "--channel", channel, "--snr_db", str(snr)]
    else:
        cmd = [py, "-u", "examples/simulate_text_transmission.py",
               "--channel", channel, "--snr_db", str(snr)]
    p = subprocess.run(cmd, env=env)
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[1.0, 12.0])
    ap.add_argument("--modalities", type=str, nargs="+", default=["edge","depth","segmentation"])
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default="rayleigh")
    ap.add_argument("--fec_preset", type=str, choices=["conv","hamm_robust"], default=os.getenv("FEC_PRESET","conv"))
    args = ap.parse_args()

    jobs = []
    for m in args.modalities:
        kind = "image"
        for snr in args.snrs:
            jobs.append((kind, m, snr))

    bars = []
    # 各ジョブに独立のプログレスバーを割当
    for idx, (_, m, snr) in enumerate(jobs):
        desc = f"{m:12s} @ {snr:>4.1f} dB"
        bars.append(tqdm(total=1, position=idx, leave=True, desc=desc, unit="run"))

    extra_env = {"FEC_PRESET": args.fec_preset}
    for idx, (kind, m, snr) in enumerate(jobs):
        t0 = time.time()
        rc = _run(kind, m, snr, args.channel, extra_env)
        dt = time.time() - t0
        bars[idx].set_postfix_str(f"{args.fec_preset} done in {dt:.1f}s rc={rc}")
        bars[idx].update(1)

    for b in bars: b.close()

if __name__ == "__main__":
    main()
