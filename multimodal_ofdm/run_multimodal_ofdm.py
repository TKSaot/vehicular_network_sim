# -*- coding: utf-8 -*-
# multimodal_ofdm/run_multimodal_ofdm.py
"""
Compatibility runner for the reorganized codebase.

Usage examples:
  # EEP: all modalities
  python -m multimodal_ofdm.run_multimodal_ofdm --mode eep --channel rayleigh --snr_db 10 --modality all

  # UEP presets
  python -m multimodal_ofdm.run_multimodal_ofdm --mode uep --power-preset text --channel rayleigh --snr_db 10 --modality all
  python -m multimodal_ofdm.run_multimodal_ofdm --mode uep --power-preset edge+depth --snr_db 10

This wrapper calls the new pipeline (app_layer → transmitter → channel → receiver),
and saves results per run into outputs/<timestamp>__<modality>__... (no subfolders per modality).
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Ensure project root on sys.path (this file is under multimodal_ofdm/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- New codebase imports ---
from common.utils import set_seed, psnr  # type: ignore
from common.run_utils import make_output_dir, write_json  # type: ignore
from common.byte_mapping import unmap_bytes  # type: ignore

from app_layer.application import (  # type: ignore
    serialize_content, AppHeader, deserialize_content, save_output
)

from transmitter.send import build_transmission  # type: ignore
from receiver.receive import recover_from_symbols  # type: ignore
from channel.channel_model import awgn_channel, rayleigh_fading  # type: ignore

from configs import image_config as IMG_CFG  # type: ignore
from configs import text_config as TXT_CFG   # type: ignore


def _set_uep_env(tier: str | None) -> None:
    """
    Set UEP_MODE env consumed by configs/*.  Accepts 'off','S','G','B'.
    """
    if tier is None:
        os.environ["UEP_MODE"] = "off"
    else:
        tier = tier.strip().upper()
        if tier not in {"OFF", "S", "G", "B"}:
            raise ValueError(f"Invalid UEP tier: {tier}")
        os.environ["UEP_MODE"] = tier


def _decide_uep_for(modality: str, mode: str, power_preset: str | None, default_tier: str) -> str:
    """
    Map legacy CLI (mode/power-preset) onto UEP_MODE tiers per modality.

    - EEP → 'off' for all modalities.
    - UEP + power-preset:
        * text:          text='S', others='B'
        * edge:          edge='S', others='B'
        * edge+depth:    edge/depth='S', others='B'
        * segmentation:  segmentation='S', others='B'
      If power-preset is omitted in UEP mode, use default_tier (e.g., 'G') for all.
    """
    mode = mode.lower()
    if mode == "eep":
        return "off"
    # UEP
    pp = (power_preset or "").strip().lower()
    if pp == "text":
        return "S" if modality == "text" else "B"
    if pp == "edge":
        return "S" if modality == "edge" else "B"
    if pp in {"edge+depth", "edge_depth", "edge-depth"}:
        return "S" if modality in {"edge", "depth"} else "B"
    if pp == "segmentation":
        return "S" if modality == "segmentation" else "B"
    # No specific preset → same tier for all
    return default_tier


def _input_path_for(modality: str) -> str:
    if modality == "text":
        return TXT_CFG.INPUT
    return IMG_CFG.INPUTS[modality]


def _build_cfg_for(modality: str, snr_db: float | None, channel: str | None) -> "object":
    if modality == "text":
        cfg = TXT_CFG.build_config()
    else:
        cfg = IMG_CFG.build_config(modality=modality)  # segmentation / edge / depth
    if snr_db is not None:
        cfg.chan.snr_db = float(snr_db)
    if channel is not None:
        cfg.chan.channel_type = channel
    return cfg


def _run_one(modality: str, snr_db: float | None, channel: str | None) -> Dict:
    """
    Run serialize → TX → channel → RX → save, and return a report dict.
    """
    cfg = _build_cfg_for(modality, snr_db, channel)
    input_path = _input_path_for(modality)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found for {modality}: {input_path}")

    set_seed(cfg.chan.seed)

    # --- TX ---
    tx_hdr, payload = serialize_content(
        modality if modality != "text" else "text",
        input_path,
        app_cfg=cfg.app,
        text_encoding="utf-8",
        validate_image_mode=True if modality != "text" else False
    )
    app_hdr_bytes = tx_hdr.to_bytes()

    tx_syms, tx_meta = build_transmission(app_hdr_bytes, payload, cfg)

    # --- Channel ---
    if cfg.chan.channel_type == "awgn":
        rx_syms = awgn_channel(tx_syms, cfg.chan.snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms,
            cfg.chan.snr_db,
            seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz,
            symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading
        )

    # --- RX ---
    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    hdr_used_mode = "valid"
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = tx_hdr
        hdr_used_mode = "forced-parse-failed"

    # Inverse byte mapping (length-clamped to original)
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes
    )

    # Application decode
    text_out, img_arr = deserialize_content(
        rx_hdr, rx_payload_b, app_cfg=cfg.app,
        text_encoding="utf-8",
        text_errors="backslashreplace"
    )

    # Save & metrics
    out_dir = make_output_dir(cfg, modality=modality, input_path=input_path, output_root="outputs")

    if modality == "text":
        out_txt = os.path.join(out_dir, "received_text.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text_out)
        # quick CER-like metric
        orig_text = open(input_path, "r", encoding="utf-8").read()
        recv_text = text_out
        L = min(len(orig_text), len(recv_text))
        mism = sum(1 for i in range(L) if orig_text[i] != recv_text[i]) + abs(len(orig_text) - len(recv_text))
        cer = mism / max(1, len(orig_text))
        report = {
            "modality": modality,
            "output_txt": out_txt,
            "frames": int(stats["n_frames"]),
            "bad_frames": int(stats["n_bad_frames"]),
            "all_crc_ok": bool(stats["all_crc_ok"]),
            "app_header_crc_ok": bool(stats["app_header_crc_ok"]),
            "app_header_used_mode": hdr_used_mode,
            "cer_approx": float(cer),
            "snr_db": float(cfg.chan.snr_db),
            "channel": cfg.chan.channel_type,
            "fec": cfg.link.fec_scheme,
            "mod": cfg.mod.scheme,
        }
        write_json(os.path.join(out_dir, "rx_stats.json"), report)
        print(f"[TEXT] saved: {out_txt} | CER≈{report['cer_approx']:.4f}")
        return report

    # image-like modalities
    out_png = os.path.join(out_dir, "received.png")
    save_output(rx_hdr, "", img_arr, out_png)

    im_orig = Image.open(input_path)
    if modality in ("edge", "depth"):
        im_orig = im_orig.convert("L")
    elif modality == "segmentation":
        im_orig = im_orig.convert("RGB")
    arr_orig = np.array(im_orig).astype(np.uint8)
    arr_rx = img_arr.astype(np.uint8)
    if arr_orig.shape != arr_rx.shape:
        # safety clamp
        arr_orig = arr_orig[:arr_rx.shape[0], :arr_rx.shape[1], ...]
    val_psnr = psnr(arr_orig, arr_rx, data_range=255)

    report = {
        "modality": modality,
        "output_png": out_png,
        "frames": int(stats["n_frames"]),
        "bad_frames": int(stats["n_bad_frames"]),
        "all_crc_ok": bool(stats["all_crc_ok"]),
        "app_header_crc_ok": bool(stats["app_header_crc_ok"]),
        "app_header_used_mode": hdr_used_mode,
        "psnr_db": float(val_psnr),
        "snr_db": float(cfg.chan.snr_db),
        "channel": cfg.chan.channel_type,
        "fec": cfg.link.fec_scheme,
        "mod": cfg.mod.scheme,
    }
    write_json(os.path.join(out_dir, "rx_stats.json"), report)
    print(f"[{modality.upper()}] saved: {out_png} | PSNR={report['psnr_db']:.2f} dB")
    return report


def main():
    ap = argparse.ArgumentParser(description="Compatibility runner for multimodal experiments.")
    ap.add_argument("--mode", type=str, choices=["eep", "uep"], default="eep",
                    help="EEP: UEP off (UEP_MODE=off). UEP: enable UEP tiers via --power-preset or --uep-tier.")
    ap.add_argument("--uep-tier", type=str, choices=["S", "G", "B"], default="G",
                    help="UEP mode only: default tier for modalities when preset not specific (S=strong, G=good, B=balanced).")
    ap.add_argument("--power-preset", type=str,
                    choices=["text", "edge", "edge+depth", "segmentation"],
                    default=None,
                    help="UEP mode only: prioritize particular modality group.")
    ap.add_argument("--modality", type=str,
                    choices=["all", "text", "edge", "depth", "segmentation"],
                    default="all",
                    help="Which modality(ies) to run.")
    ap.add_argument("--channel", type=str, choices=["awgn", "rayleigh"], default="rayleigh")
    ap.add_argument("--snr_db", type=float, default=10.0)
    args = ap.parse_args()

    # Decide list to run
    mods: List[str]
    if args.modality == "all":
        mods = ["text", "edge", "depth", "segmentation"]
    else:
        mods = [args.modality]

    reports: List[Dict] = []
    for m in mods:
        tier = _decide_uep_for(m, args.mode, args.power_preset, args.uep_tier)
        _set_uep_env(tier)  # consumed by configs/*.py
        print(f"\n=== Running {m} | mode={args.mode.upper()} | UEP_MODE={os.environ.get('UEP_MODE')} | "
              f"ch={args.channel} | SNR={args.snr_db} dB ===")
        rep = _run_one(m, args.snr_db, args.channel)
        reports.append(rep)

    # Print compact summary
    print("\n=== Summary ===")
    for r in reports:
        if r["modality"] == "text":
            print(f"  [text] CER≈{r['cer_approx']:.4f}")
        else:
            print(f"  [{r['modality']}] PSNR={r['psnr_db']:.2f} dB")


if __name__ == "__main__":
    main()
