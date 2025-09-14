# examples/simulate_text_transmission.py
"""
EEP(Hamming) 前提のテキスト送受信。configs/text_config.py で設定。
"""
import os, sys, argparse, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.utils import set_seed
from common.run_utils import make_output_dir, write_json
from common.config import SimulationConfig
from common.byte_mapping import unmap_bytes
from app_layer.application import serialize_content, AppHeader, deserialize_content
from transmitter.send import build_transmission
from channel.channel_model import awgn_channel, rayleigh_fading
from receiver.receive import recover_from_symbols

from configs import text_config as CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default=None)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--output_root", type=str, default=None)
    args = ap.parse_args()

    cfg: SimulationConfig = CFG.build_config()
    input_path = args.input if args.input is not None else CFG.INPUT
    output_root = args.output_root if args.output_root is not None else CFG.OUTPUT_ROOT

    if args.snr_db is not None:
        cfg.chan.snr_db = float(args.snr_db)
    if args.channel is not None:
        cfg.chan.channel_type = args.channel

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input text not found: {input_path}")

    set_seed(cfg.chan.seed)

    # --- TX: serialize (bytes) ---
    tx_hdr, payload = serialize_content(
        "text", input_path, app_cfg=cfg.app,
        text_encoding="utf-8", validate_image_mode=False
    )
    app_hdr_bytes = tx_hdr.to_bytes()

    # --- PHY TX ---
    tx_syms, tx_meta = build_transmission(app_hdr_bytes, payload, cfg)

    # --- Channel ---
    if cfg.chan.channel_type == "awgn":
        rx_syms = awgn_channel(tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading
        )

    # --- RX ---
    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    # header fallback
    hdr_used_mode = "ok"
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = tx_hdr
        hdr_used_mode = "forced-parse-failed"

    # Unmap (inverse permutation)
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes
    )

    # Keep undecodable bytes as \xAB (LLM 後処理が容易)
    text_str, _ = deserialize_content(
        rx_hdr, rx_payload_b, app_cfg=cfg.app,
        text_encoding="utf-8", text_errors="backslashreplace"
    )

    # --- Save ---
    out_dir = make_output_dir(cfg, modality="text", input_path=input_path, output_root=output_root)
    out_txt = os.path.join(out_dir, "received_text.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text_str)

    # --- Metrics ---
    orig_text = open(input_path, "r", encoding="utf-8").read()
    recv_text = text_str
    minlen = min(len(orig_text), len(recv_text))
    mismatches = sum(1 for i in range(minlen) if orig_text[i] != recv_text[i]) + abs(len(orig_text)-len(recv_text))
    cer = mismatches / max(1, len(orig_text))

    report = {
        "frames": int(stats["n_frames"]),
        "bad_frames": int(stats["n_bad_frames"]),
        "all_crc_ok": bool(stats["all_crc_ok"]),
        "app_header_crc_ok": bool(stats["app_header_crc_ok"]),
        "hdr_via_majority": bool(stats.get("app_header_recovered_via_majority", False)),
        "header_mode": hdr_used_mode,
        "cer_approx": float(cer),
        "snr_db": float(cfg.chan.snr_db),
        "channel": cfg.chan.channel_type,
        "mod_scheme": cfg.mod.scheme,
        "fec_scheme": cfg.link.fec_scheme,
        "output_txt": out_txt,
    }
    write_json(os.path.join(out_dir, "rx_stats.json"), report)

    print("=== TEXT Transmission Report ===")
    print(f"Output dir: {out_dir}")
    print(f"SNR(dB): {cfg.chan.snr_db}, Channel: {cfg.chan.channel_type}, Mod: {cfg.mod.scheme}, FEC: {cfg.link.fec_scheme}")
    print(f"Frames: {report['frames']}, Bad: {report['bad_frames']}, All CRC OK: {report['all_crc_ok']}")
    print(f"Header mode: {hdr_used_mode}")
    print(f"Approx. CER: {report['cer_approx']:.4f}")
    print(f"Saved: {out_txt}")

if __name__ == "__main__":
    main()
