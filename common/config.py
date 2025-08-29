# common/config.py
"""
Configuration dataclasses for the vehicular network simulation.
These allow you to customize parameters across the stack.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class AppConfig:
    # modality: "text" | "edge" | "depth" | "segmentation"
    modality: Literal["text", "edge", "depth", "segmentation"] = "text"
    # For images: if True, enforce the expected channel layout per modality
    validate_image_mode: bool = True
    # For text decoding robustness: replace undecodable bytes
    text_encoding: str = "utf-8"
    text_errors: str = "replace"  # "strict", "ignore", or "replace"

@dataclass
class LinkConfig:
    mtu_bytes: int = 1024             # payload bytes per DATA frame (before FEC/interleaving)
    interleaver_depth: int = 8        # block interleaver depth (>=1). 1 disables interleaving
    fec_scheme: Literal["none", "repeat", "hamming74", "rs255_223"] = "hamming74"
    repeat_k: int = 3                 # used if fec_scheme == "repeat"
    drop_bad_frames: bool = False     # if CRC fails: drop the frame (True) or keep corrupted payload (False)

    # Header robustness
    strong_header_protection: bool = True  # keep True
    header_copies: int = 7                 # repeat APP header frames (front) to ensure recovery at low SNR
    header_rep_k: int = 5                  # repetition factor applied to APP header frames after FEC
    force_output_on_hdr_fail: bool = True  # if header still fails, force using TX-known header to produce outputs

    verbose: bool = False             # print per-frame logs

@dataclass
class ModulationConfig:
    scheme: Literal["bpsk", "qpsk", "16qam"] = "qpsk"

@dataclass
class ChannelConfig:
    channel_type: Literal["awgn", "rayleigh"] = "rayleigh"
    snr_db: float = 10.0                 # Es/N0 in dB
    seed: Optional[int] = 12345          # RNG seed for reproducibility
    # Rayleigh-specific (simple time-varying AR(1) fading)
    doppler_hz: float = 30.0             # nominal Doppler (affects fading correlation)
    symbol_rate: float = 1e6             # symbols per second (for Doppler correlation)
    block_fading: bool = False           # if True, h is constant over a frame

@dataclass
class PilotConfig:
    preamble_len: int = 32               # BPSK preamble bits per frame (not used for sync in this sim)
    pilot_len: int = 16                  # known pilot symbols per frame for channel estimation (QPSK pilots)
    pilot_every_n_symbols: int = 0       # not used in this simplified sim (set 0 to disable mid-frame pilots)

@dataclass
class SimulationConfig:
    app: AppConfig = field(default_factory=AppConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    mod: ModulationConfig = field(default_factory=ModulationConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
