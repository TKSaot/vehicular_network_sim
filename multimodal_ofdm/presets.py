
# multimodal_ofdm/presets.py
from __future__ import annotations
from typing import Dict

# 名前で選べる UEP プリセット（線形電力重み）
POWER_PRESETS: Dict[str, Dict[str, float]] = {
    "eep":           {"text":1, "edge":1, "depth":1, "segmentation":1},
    "text":          {"text":3, "edge":1, "depth":1, "segmentation":1},
    "edge":          {"text":1, "edge":3, "depth":1, "segmentation":1},
    "edge_depth":    {"text":1, "edge":2, "depth":2, "segmentation":1},
    "seg":           {"text":1, "edge":1, "depth":1, "segmentation":3},
    # 別名
    "segmentation":  {"text":1, "edge":1, "depth":1, "segmentation":3},
}
