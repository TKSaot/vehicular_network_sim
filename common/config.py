"""
Configuration dataclasses for the vehicular network simulation.
Now includes per-modality *decoder* options controlled from config.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

# ---------- Decoder options ----------
@dataclass
class SegDecoderConfig:
    # Out-of-range ID fallback at decoder
    fallback: Literal["uniform", "clamp", "mod"] = "uniform"
    # Post-filter strength
    # - "none": no smoothing
    # - "majority3": 3x3 モード（反復 iters）
    # - "majority5": 5x5 モード（反復 iters）
    # - "strong":   5x5 モード + 多数派合意（consensus，min_frac）
    mode: Literal["none", "majority3", "majority5", "strong"] = "strong"
    iters: int = 2
    # consensus：近傍内で同一ラベル率がこの閾値未満なら近傍モードに置換
    consensus_min_frac: float = 0.6
    # 乱数を使う処理の種（uniform 代替 ID など）
    seed: Optional[int] = 123

@dataclass
class EdgeDecoderConfig:
    # "none" | "maj3" | "maj5" | "median3" | "median5" |
    # "open3" | "open5" | "open3close3" | "open5close5"
    denoise: Literal["none","maj3","maj5","median3","median5","open3","open5","open3close3","open5close5"] = "open3close3"
    iters: int = 1
    # Majority のしきい値（未指定なら過半数）
    thresh: Optional[int] = None
    # 細線保護：1→0 で線が消えそうな画素は保持
    preserve_lines: bool = True

@dataclass
class DepthDecoderConfig:
    # "none" | "median3" | "median5" | "bilateral5" | "median5_bilateral5"
    filt: Literal["none","median3","median5","bilateral5","median5_bilateral5"] = "median5_bilateral5"
    iters: int = 1
    # bilateral(5x5)
    sigma_s: float = 1.6
    sigma_r: float = 12.0

# ---------- App / Link / PHY / Channel ----------
@dataclass
class AppConfig:
    modality: Literal["text", "edge", "depth", "segmentation"] = "text"
    validate_image_mode: bool = True
    text_encoding: str = "utf-8"
    text_errors: str = "replace"
    # 受信側デコーダ設定
    segdec: SegDecoderConfig = field(default_factory=SegDecoderConfig)
    edgedec: EdgeDecoderConfig = field(default_factory=EdgeDecoderConfig)
    depthdec: DepthDecoderConfig = field(default_factory=DepthDecoderConfig)

@dataclass
class LinkConfig:
    mtu_bytes: int = 1024
    interleaver_depth: int = 16
    # ★ 802.11p 風畳み込み符号のレート名も選択可能（既定は hamming74 のまま）
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

    # 送信前のバイトマッピング（フレーム化前の拡散）
    byte_mapping_scheme: Literal["none", "permute", "frame_block"] = "none"
    byte_mapping_seed: Optional[int] = None  # None → chan.seed を使用

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
    # （必要なら）snr_reference を channel 側でオプション引数に渡す

@dataclass
class PilotConfig:
    preamble_len: int = 32
    pilot_len: int = 16
    pilot_every_n_symbols: int = 0

@dataclass
class SimulationConfig:
    app: AppConfig = field(default_factory=AppConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    mod: ModulationConfig = field(default_factory=ModulationConfig)
    chan: ChannelConfig = field(default_factory=ChannelConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
