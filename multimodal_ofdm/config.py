from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal

Modal = Literal["text", "edge", "depth", "segmentation"]

@dataclass
class AppLayerConfig:
    # セグメンテーション復元の軽い後処理パラメータ（必要最小限）
    seg_white_thresh: int = 250
    seg_mode: Literal["none", "majority3", "majority5", "strong"] = "strong"
    seg_iters: int = 2
    seg_consensus_min_frac: float = 0.6
    seg_seed: int = 123

@dataclass
class LinkConfig:
    # FECは Hamming(7,4)，ブロック・インタリーブ固定（UEP/EEP で変更しない）
    mtu_bytes: int = 256
    interleaver_depth: int = 128
    header_rep_k: int = 5        # ヘッダ繰り返し
    header_boost_db: float = 6.0 # ヘッダのみ一定増幅（メタデータの保全用）

    # ★ 追加: バイトマッピング
    # none | permute（乱順）
    byte_mapping: Literal["none", "permute"] = "permute"
    # 乱順のシード（モダリティごとに派生させる）
    byte_seed: int = 12345

@dataclass
class OfdmConfig:
    n_fft: int = 512
    used_subcarriers: int = 480   # 端とDCは未使用
    cp_len: int = 64
    pilot_symbol_index: int = 0   # 先頭シンボルをパイロットに
    # 4モダリティに等分割（UEP/EEPでも固定，UEPは電力だけ変える）
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
    # 線形電力重み（平均電力は内部で正規化）．EEPなら全1．
    weights: Dict[Modal, float] = field(default_factory=lambda: {
        "text": 1.0, "edge": 1.0, "depth": 1.0, "segmentation": 1.0
    })

@dataclass
class Paths:
    # 既定はパッケージと同階層の examples/ を参照
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
