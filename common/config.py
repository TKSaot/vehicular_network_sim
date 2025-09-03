# common/config.py
"""
Configuration dataclasses for the vehicular network simulation.
Now includes per-modality *decoder* options (moved from env-vars to config).
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

# ---------------- App-level decoder options ----------------
@dataclass
class SegDecoderConfig:
    # Out-of-range ID fallback at decoder
    fallback: Literal["uniform", "clamp", "mod"] = "uniform"
    seed: Optional[int] = None
    # Majority voting
    maj3x3: bool = False
    maj_iters: int = 1
    # Edge-assisted boundary protection
    use_edge_guidance: bool = False
    edge_guide_path: Optional[str] = None  # e.g., corrected edge PNG path

@dataclass
class EdgeDecoderConfig:
    # Denoiser: majority (3/5) or median (3/5), or none
    denoise: Literal["none", "maj3", "maj5", "median3", "median5"] = "none"
    iters: int = 1
    # For majority: threshold; default is ceil(k^2/2)
    thresh: Optional[int] = None
    # Preserve thin lines: avoid turning 1->0 if it would erase a ridge
    preserve_lines: bool = True

@dataclass
class DepthDecoderConfig:
    # Filter: median (3/5), simple bilateral(5), or none
    filt: Literal["none", "median3", "median5", "bilateral5"] = "median3"
    iters: int = 1
    # Bilateral parameters (5x5 kernel)
    sigma_s: float = 1.6   # spatial
    sigma_r: float = 12.0  # range (intensity)

@dataclass
class AppConfig:
    # modality: "text" | "edge" | "depth" | "segmentation"
    modality: Literal["text", "edge", "depth", "segmentation"] = "text"
    # For images: if True, enforce the expected channel layout per modality
    validate_image_mode: bool = True
    # For text decoding robustness: replace undecodable bytes
    text_encoding: str = "utf-8"
    text_errors: str = "replace"

    # --- Segmentation TX-side (optional semantic noise; default OFF) ---
    seg_strip_white: bool = True
    seg_white_thresh: int = 250
    seg_tx_noise_p: float = 0.0
    seg_tx_noise_seed: Optional[int] = None

    # --- Decoder options (moved from env to config) ---
    segdec: SegDecoderConfig = field(default_factory=SegDecoderConfig)
    edgedec: EdgeDecoderConfig = field(default_factory=EdgeDecoderConfig)
    depthdec: DepthDecoderConfig = field(default_factory=DepthDecoderConfig)

@dataclass
class LinkConfig:
    mtu_bytes: int = 1024
    interleaver_depth: int = 16
    fec_scheme: Literal["none", "repeat", "hamming74", "rs255_223"] = "hamming74"
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

@dataclass
class PilotConfig:
    preamble_len: int = 32
    pilot_len: int = 16
    pilot_every_n_symbols: int = 0  # unused (simple sim)

@dataclass
class SimulationConfig:
    app: AppConfig = field(default_factory=AppConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    mod: ModulationConfig = field(default_factory=ModulationConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
