from __future__ import annotations
from typing import Dict

# UEP presets: linear power weights (the runner normalizes total power)
POWER_PRESETS: Dict[str, Dict[str, float]] = {
    # Baseline
    "eep": {"text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},

    # --- Previously provided examples ---
    "boost_text_x2":  {"text": 2.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},
    "boost_text_x4":  {"text": 4.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},
    "boost_edge_x2":  {"text": 1.0, "edge": 2.0, "depth": 1.0, "segmentation": 1.0},
    "boost_depth_x2": {"text": 1.0, "edge": 1.0, "depth": 2.0, "segmentation": 1.0},
    "geom_pair_x2":   {"text": 1.0, "edge": 2.0, "depth": 2.0, "segmentation": 1.0},
    "text_edge_x2":   {"text": 2.0, "edge": 2.0, "depth": 1.0, "segmentation": 1.0},
    "text_depth_x2":  {"text": 2.0, "edge": 1.0, "depth": 2.0, "segmentation": 1.0},
    "low_seg":        {"text": 1.2, "edge": 1.2, "depth": 1.2, "segmentation": 0.5},

    # ===============================================================
    # New: SNR-specific "starting points" for your sweeps.
    # Two RX conditions:
    #   - none     => (seg_mode=edge_denoise=depth_denoise='none')
    #   - rxstrong => (seg_mode='strong', edge_denoise='strong', depth_denoise='median5')
    # These names do NOT change the RX config; they are just power presets.
    # Drive RX behavior via CLI flags (see run_multimodal_ofdm.py).
    # ===============================================================

    # ---- SNR=1 dB ----
    "snr1_none_base":       {"text": 0.6, "edge": 2.6, "depth": 2.2, "segmentation": 0.6},
    "snr1_none_drop_seg":   {"text": 0.6, "edge": 2.8, "depth": 2.6, "segmentation": 0.0},
    "snr1_none_geom_only":  {"text": 0.0, "edge": 3.2, "depth": 2.8, "segmentation": 0.0},

    "snr1_rxstrong_base":   {"text": 0.8, "edge": 2.5, "depth": 2.5, "segmentation": 0.2},
    "snr1_rxstrong_seglo":  {"text": 0.7, "edge": 2.4, "depth": 2.4, "segmentation": 0.5},

    # ---- SNR=6 dB ----
    "snr6_none_base":       {"text": 1.4, "edge": 2.0, "depth": 2.0, "segmentation": 0.6},
    "snr6_none_geom_plus":  {"text": 1.2, "edge": 2.2, "depth": 2.2, "segmentation": 0.4},

    "snr6_rxstrong_base":   {"text": 1.2, "edge": 1.8, "depth": 2.0, "segmentation": 1.0},
    "snr6_rxstrong_segx":   {"text": 1.0, "edge": 1.7, "depth": 2.1, "segmentation": 1.2},

    # ---- SNR=12 dB ----
    "snr12_none_base":      {"text": 1.2, "edge": 1.3, "depth": 1.5, "segmentation": 1.0},
    "snr12_none_textplus":  {"text": 1.5, "edge": 1.2, "depth": 1.4, "segmentation": 0.9},

    "snr12_rxstrong_base":  {"text": 1.2, "edge": 1.1, "depth": 1.4, "segmentation": 1.3},
    "snr12_rxstrong_depthx":{"text": 1.0, "edge": 1.1, "depth": 1.6, "segmentation": 1.3},
}