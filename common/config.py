# common/config.py
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class AppConfig:
    modality: Literal["text", "edge", "depth", "segmentation"] = "text"
    validate_image_mode: bool = True
    text_encoding: str = "utf-8"
    text_errors: str = "replace"

@dataclass
class LinkConfig:
    mtu_bytes: int = 1024
    interleaver_depth: int = 8
    # NEW: convolutional codes are now available
    fec_scheme: Literal[
        "none", "repeat", "hamming74", "rs255_223",
        "conv_k7_r12", "conv_k7_r23", "conv_k7_r34"
    ] = "hamming74"
    repeat_k: int = 3
    drop_bad_frames: bool = False

    strong_header_protection: bool = True
    header_copies: int = 7
    header_rep_k: int = 5
    force_output_on_hdr_fail: bool = True
    verbose: bool = False

    byte_mapping_scheme: Literal["none", "permute", "frame_block"] = "none"
    byte_mapping_seed: Optional[int] = None

@dataclass
class ModulationConfig:
    scheme: Literal["bpsk", "qpsk", "16qam"] = "qpsk"

@dataclass
class ChannelConfig:
    channel_type: Literal["awgn", "rayleigh"] = "rayleigh"
    snr_db: float = 10.0
    seed: Optional[int] = 12345
    doppler_hz: float = 30.0
    symbol_rate: float = 1e6
    block_fading: bool = False
    # NEW: average SNR calibration mode for Rayleigh
    snr_reference: Literal["tx", "rx"] = "rx"

@dataclass
class PilotConfig:
    preamble_len: int = 32
    pilot_len: int = 16
    pilot_every_n_symbols: int = 0  # not used

@dataclass
class SimulationConfig:
    app: AppConfig = field(default_factory=AppConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    mod: ModulationConfig = field(default_factory=ModulationConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
