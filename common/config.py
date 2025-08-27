from dataclasses import dataclass, field
from typing import Optional, Literal

ModulationScheme = Literal["BPSK", "QPSK", "16QAM"]
ChannelType = Literal["awgn", "rayleigh", "rician"]
CodingScheme = Literal["none", "repetition3", "hamming74"]
ImageKind = Literal["depth", "edge", "seg"]  # 画像タイプ

@dataclass
class PHYConfig:
    modulation: ModulationScheme = "QPSK"
    snr_db: float = 8.0
    snr_type: Literal["EsN0", "EbN0"] = "EsN0"
    code_rate_override: Optional[float] = None
    soft_demapping: bool = False

@dataclass
class ChannelConfig:
    channel_type: str = "awgn"
    rician_K: float = 3.0
    time_selective: bool = False
    fading_model: str = "block"   # "block" | "gauss_markov"
    coherence_symbols: int = 32   # block-fading のブロック長
    rho: float = 0.95             # gauss_markov の相関

@dataclass
class LinkConfig:
    framing: Literal["length_crc32"] = "length_crc32"
    # 大きめ既定（約 5e6 bit = 625kB）
    max_frame_bits: int = 5_000_000
    interleaver: bool = True
    interleaver_seed: int = 2025
    coding: CodingScheme = "hamming74"

@dataclass
class UEPConfig:
    enabled: bool = False
    header_mod: ModulationScheme = "BPSK"
    header_coding: CodingScheme = "repetition3"
    header_interleaver: bool = True
    header_boost_db: float = 6.0
    header_max_frame_bits: int = 65_536
    header_repeats: int = 1

@dataclass
class AppConfig:
    app_type: Literal["text", "image"] = "text"
    text: str = "Hello vehicular world! こんにちは🚗"
    image_path: str = "examples/sample_image.png"
    # 0 または負値で元サイズ保持
    image_resize: int = 128
    # 保存時の最終サイズ（0 なら送信サイズをそのまま）
    image_save_size: int = 0
    image_kind: ImageKind = "depth"

@dataclass
class SimConfig:
    phy: PHYConfig = field(default_factory=PHYConfig)
    ch: ChannelConfig = field(default_factory=ChannelConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    app: AppConfig = field(default_factory=AppConfig)
    uep: UEPConfig = field(default_factory=UEPConfig)
    verbose: bool = True

def ebn0_to_esn0_db(ebn0_db: float, bits_per_sym: int, code_rate: float = 1.0) -> float:
    """Eb/N0[dB] -> Es/N0[dB]。Es = Eb * (bits_per_sym * code_rate)"""
    import math
    return ebn0_db + 10*math.log10(bits_per_sym * code_rate)
