from __future__ import annotations
from typing import Dict

# UEP presets: linear power weights (the runner normalizes total power)
POWER_PRESETS: Dict[str, Dict[str, float]] = {
    # Baselines
    "eep":            {"text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},
    "text":           {"text": 3.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0},
    "edge":           {"text": 1.0, "edge": 3.0, "depth": 1.0, "segmentation": 1.0},
    "edge_depth":     {"text": 1.0, "edge": 2.0, "depth": 2.0, "segmentation": 1.0},
    "seg":            {"text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 3.0},
    # Alias
    "segmentation":   {"text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 3.0},

    # --- Extra curated patterns ---
    # Text + Edge (medium): readable text + crisp structure at low SNR
    "text_edge_med":  {"text": 2.0, "edge": 2.0, "depth": 1.0, "segmentation": 1.0},
    # Text + Edge (strong): push hard when SNR=3 dB and those are priorities
    "text_edge_strong":{"text": 3.0, "edge": 3.0, "depth": 0.5, "segmentation": 0.5},
    # Geometry-heavy: keep edges & depth clean, accept mild text/seg loss
    "geometry":       {"text": 0.8, "edge": 2.2, "depth": 2.2, "segmentation": 0.8},
    # Semantics-heavy: protect text + region labels
    "semantics":      {"text": 2.2, "edge": 0.9, "depth": 0.9, "segmentation": 2.0},
    # Boundary-preserving: extra weight to edge + segmentation
    "edge_seg":       {"text": 0.9, "edge": 2.5, "depth": 1.0, "segmentation": 1.6},
    # Depth-only boost (for completeness)
    "depth":          {"text": 1.0, "edge": 1.0, "depth": 3.0, "segmentation": 1.0},
}
