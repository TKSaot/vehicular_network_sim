from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Literal

Modal = Literal["text", "edge", "depth", "segmentation"]

# MODIFIED: LlmConfig now directly controls the feature
@dataclass
class LlmConfig:
    correction_enabled: bool = True
    # 環境変数からAPIキーを読み込む
    api_key: str = os.getenv("OPENAI_API_KEY") 
    model_name: str = "gpt-3.5-turbo"


@dataclass
class AppLayerConfig:
    # --- Text (TX/RX) ---
    text_symbols = "abcdefghijklmnopqrstuvwxyz1234567890, .\n"
    text_bits_per_char: int = 8
    text_casefold: bool = True

    # --- Segmentation (RX) ---
    seg_white_thresh: int = 250
    seg_mode: Literal["none", "majority3", "majority5", "strong"] = "none"
    seg_iters: int = 2
    seg_consensus_min_frac: float = 0.6
    seg_seed: int = 123

    # --- Edge (RX) ---
    edge_denoise: Literal["none", "gentle", "medium", "strong"] = "none"
    edge_iters: int = 1

    # --- Depth (RX) ---
    depth_denoise: Literal["none", "median3", "median5"] = "none"
    depth_iters: int = 1

@dataclass
class LinkConfig:
    # --- FEC / Decoder ---
    fec_enabled: bool = True
    decoder_type: Literal["hard", "soft"] = "soft"
    mtu_bytes: int = 256
    interleaver_depth: int = 256
    header_rep_k: int = 5

    # --- Payload Repetition ---
    payload_rep_k: Dict[Modal, int] = field(default_factory=lambda: {
        'text': 1, 'edge': 1, 'depth': 1, 'segmentation': 1
    })

    # --- Byte Mapping ---
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
    llm: LlmConfig = field(default_factory=LlmConfig)
    app: AppLayerConfig = field(default_factory=AppLayerConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    ofdm: OfdmConfig = field(default_factory=OfdmConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    paths: Paths = field(default_factory=Paths)