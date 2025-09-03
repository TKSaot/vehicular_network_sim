# configs/image_config.py
from __future__ import annotations
from typing import Literal
from common.config import (
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig,
    SegDecoderConfig, EdgeDecoderConfig, DepthDecoderConfig
)

INPUTS = {
    "edge": "examples/edge_00001_.png",
    "depth": "examples/depth_00001_.png",
    "segmentation": "examples/segmentation_00001_.png",
}
OUTPUT_ROOT = "outputs"
DEFAULT_MODALITY: Literal["edge","depth","segmentation"] = "segmentation"

def build_config(modality: Literal["edge","depth","segmentation"] = DEFAULT_MODALITY) -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(
            modality=modality,
            validate_image_mode=True,
            segdec=SegDecoderConfig(
                fallback="uniform",
                mode="strong",       # 5x5 モード + consensus（強め）
                iters=2,
                consensus_min_frac=0.6,
                seed=123
            ),
            edgedec=EdgeDecoderConfig(
                denoise="open3close3",  # pepper を強く抑える既定
                iters=1,
                thresh=None,
                preserve_lines=True
            ),
            depthdec=DepthDecoderConfig(
                filt="median5_bilateral5",  # 5x5メディアン→5x5バイラテラル
                iters=1,
                sigma_s=1.6,
                sigma_r=12.0
            ),
        ),
        link=LinkConfig(
            mtu_bytes=1024,
            interleaver_depth=16,
            fec_scheme="hamming74",
            repeat_k=3,
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            byte_mapping_scheme="permute",
            byte_mapping_seed=None,
        ),
        mod=ModulationConfig(scheme="qpsk"),
        chan=ChannelConfig(
            channel_type="rayleigh",
            snr_db=10.0,
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=16),
    )
