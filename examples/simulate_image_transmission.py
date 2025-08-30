# examples/simulate_image_transmission.py
"""
設定は configs/image_config.py に集約。
最小コマンド例:
  python examples/simulate_image_transmission.py --modality depth
  python examples/simulate_image_transmission.py --modality edge
  python examples/simulate_image_transmission.py --modality segmentation
必要なら --snr_db 等だけ上書き。
"""

import os, sys, argparse, numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.utils import set_seed, psnr
from common.run_utils import make_output_dir, write_json
from common.config import SimulationConfig
from common.byte_mapping import unmap_bytes
from app_layer.application import serialize_content, AppHeader, deserialize_content, save_output
from transmitter.send import build_transmission
from channel.channel_model import awgn_channel, rayleigh_fading
from receiver.receive import recover_from_symbols

from configs import image_config as CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", type=str, choices=["edge","depth","segmentation"], default=None)
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default=None)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--output_root", type=str, default=None)
    args = ap.parse_args()

    modality = args.modality if args.modality is not None else CFG.DEFAULT_MODALITY
    cfg: SimulationConfig = CFG.build_config(modality=modality)

    input_path = args.input if args.input is not None else CFG.INPUTS[modality]
    output_root = args.output_root if args.output_root is not None else CFG.OUTPUT_ROOT

    if args.snr_db is not None:
        cfg.chan.snr_db = float(args.snr_db)
    if args.channel is not None:
        cfg.chan.channel_type = args.channel

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input PNG not found: {input_path}")

    set_seed(cfg.chan.seed)

    # TX (keep tx_hdr for last-resort)
    tx_hdr, payload = serialize_content(modality, input_path, text_encoding="utf-8", validate_image_mode=True)
    app_hdr_bytes = tx_hdr.to_bytes()

    tx_syms, tx_meta = build_transmission(app_hdr_bytes, payload, cfg)

    if cfg.chan.channel_type == "awgn":
        rx_syms = awgn_channel(tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading
        )

    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    # Header selection
    hdr_used_mode = "valid"
    if not stats.get("app_header_crc_ok", False):
        if stats.get("app_header_recovered_via_majority", False):
            hdr_used_mode = "majority"
        elif cfg.link.force_output_on_hdr_fail:
            rx_app_hdr_b = tx_hdr.to_bytes()
            hdr_used_mode = "forced"
        else:
            hdr_used_mode = "invalid"
    # Parse header now (may be forced/majority)
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = tx_hdr
        hdr_used_mode = "forced-parse-failed"

    # NEW: invert byte mapping using hdr.payload_len_bytes (original length)
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes
    )

    # Robust image decoding (safe reshape inside, as before)
    _, img_arr = deserialize_content(rx_hdr, rx_payload_b, text_encoding="utf-8")

    out_dir = make_output_dir(cfg, modality=modality, input_path=input_path, output_root=output_root)
    out_png = os.path.join(out_dir, "received.png")
    save_output(rx_hdr, "", img_arr, out_png)

    # PSNR (with coercion to same shape already ensured)
    im_orig = Image.open(input_path)
    if modality in ("edge","depth"): im_orig = im_orig.convert("L")
    elif modality == "segmentation": im_orig = im_orig.convert("RGB")
    arr_orig = np.array(im_orig)
    if arr_orig.shape != img_arr.shape:
        arr_orig = arr_orig[:img_arr.shape[0], :img_arr.shape[1], ...]
    val_psnr = psnr(arr_orig.astype(np.uint8), img_arr.astype(np.uint8), data_range=255)

    report = {
        "modality": modality,
        "frames": int(stats["n_frames"]),
        "bad_frames": int(stats["n_bad_frames"]),
        "all_crc_ok": bool(stats["all_crc_ok"]),
        "app_header_crc_ok": bool(stats["app_header_crc_ok"]),
        "app_header_recovered_via_majority": bool(stats.get("app_header_recovered_via_majority", False)),
        "app_header_used_mode": hdr_used_mode,
        "psnr_db": float(val_psnr),
        "output_png": out_png,
    }
    write_json(os.path.join(out_dir, "rx_stats.json"), report)

    print("=== IMAGE Transmission Report ===")
    print(f"Output dir: {out_dir}")
    print(f"Modality: {modality}")
    print(f"SNR(dB): {cfg.chan.snr_db}, Channel: {cfg.chan.channel_type}, Mod: {cfg.mod.scheme}, FEC: {cfg.link.fec_scheme}")
    print(f"Frames: {report['frames']}, Bad: {report['bad_frames']}, All CRC OK: {report['all_crc_ok']}")
    print(f"Header mode: {hdr_used_mode}")
    print(f"PSNR(dB): {report['psnr_db']:.2f}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
