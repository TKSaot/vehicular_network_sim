from __future__ import annotations
from typing import Dict

# UEP presets: linear power weights (the runner normalizes total power)
POWER_PRESETS: Dict[str, Dict[str, float]] = {
    # Baseline
    "eep": {"text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},

    # --- Single-modality emphasis (dose levels) ---
    "boost_text_x2":  {"text": 2.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},
    "boost_text_x4":  {"text": 4.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},
    "boost_edge_x2":  {"text": 1.0, "edge": 2.0, "depth": 1.0, "segmentation": 1.0},
    "boost_depth_x2": {"text": 1.0, "edge": 1.0, "depth": 2.0, "segmentation": 1.0},

    # --- Two-modality patterns you actually care about ---
    "geom_pair_x2":   {"text": 1.0, "edge": 2.0, "depth": 2.0, "segmentation": 1.0},  # geometry
    "text_edge_x2":   {"text": 2.0, "edge": 2.0, "depth": 1.0, "segmentation": 1.0},
    "text_depth_x2":  {"text": 2.0, "edge": 1.0, "depth": 2.0, "segmentation": 1.0},

    # --- De-emphasize segmentation (your hypothesis) ---
    "low_seg":        {"text": 1.2, "edge": 1.2, "depth": 1.2, "segmentation": 0.5},
}
