from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal

Modal = Literal["text", "edge", "depth", "segmentation"]

@dataclass
class AppLayerConfig:
    # --- Segmentation (RX) ---
    # pre-clean for white-like boundaries (TX also reads this when available)
    seg_white_thresh: int = 250
    # receiver-side denoise strength
    seg_mode: Literal["none", "majority3", "majority5", "strong"] = "strong"
    seg_iters: int = 2
    seg_consensus_min_frac: float = 0.6  # majority confidence threshold
    seg_seed: int = 123

    # --- Edge (RX) ---
    # gentle keeps true thin lines; medium is a touch stronger; strong closes small gaps more aggressively
    edge_denoise: Literal["none", "gentle", "medium", "strong"] = "gentle"
    edge_iters: int = 1

    # --- Depth (RX) ---
    # median3 (3x3) is robust to salt-and-pepper; median5 applies two 3x3 passes roughly equivalent to 5x5
    depth_denoise: Literal["none", "median3", "median5"] = "median3"
    depth_iters: int = 1

@dataclass
class LinkConfig:
    # FEC: Hamming(7,4) + block interleaver (UEP/EEP does not change this)
    mtu_bytes: int = 256
    interleaver_depth: int = 128
    header_rep_k: int = 5
    header_boost_db: float = 6.0

    # --- Byte mapping (payload only) ---
    byte_mapping: Literal["none", "permute"] = "permute"
    byte_seed: int = 12345

@dataclass
class OfdmConfig:
    n_fft: int = 512
    used_subcarriers: int = 480
    cp_len: int = 64
    pilot_symbol_index: int = 0
    subcarrier_split: Dict[Modal, float] = field(default_factory=lambda: {
        "text": 0.25, "edge": 0.25, "depth": 0.25, "segmentation": 0.25
    })

@dataclass
class ChannelConfig:
    channel: Literal["rayleigh", "awgn"] = "rayleigh"
    snr_db: float = 10.0
    seed: int = 12345

@dataclass
class PowerConfig:
    weights: Dict[Modal, float] = field(default_factory=lambda: {
        "text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0
    })

@dataclass
class Paths:
    text_path: str = "examples/sample.txt"
    edge_path: str = "examples/edge_00001_.png"
    depth_path: str = "examples/depth_00001_.png"
    seg_path: str  = "examples/segmentation_00001_.png"
    output_root: str = "outputs"

@dataclass
class ExperimentConfig:
    app: AppLayerConfig = field(default_factory=AppLayerConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    ofdm: OfdmConfig = field(default_factory=OfdmConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    paths: Paths = field(default_factory=Paths)
