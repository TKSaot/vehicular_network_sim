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
    "text": "examples/sample.txt",
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
                mode="strong",
                iters=2,
                consensus_min_frac=0.6,
                seed=123
            ),
            edgedec=EdgeDecoderConfig(
                denoise="open3close3",
                iters=1,
                thresh=None,
                preserve_lines=True
            ),
            depthdec=DepthDecoderConfig(
                filt="median5_bilateral5",
                iters=1,
                sigma_s=1.6,
                sigma_r=12.0
            ),
        ),
        link=LinkConfig(
            mtu_bytes=256,
            interleaver_depth=128,
            fec_scheme="hamming74",
            repeat_k=1,                     # note: repeat_k is ignored for hamming74
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            byte_mapping_scheme="permute",
            byte_mapping_seed=None,
        ),
        mod=ModulationConfig(scheme="bpsk"),  # robust at very low SNR
        chan=ChannelConfig(
            channel_type="rayleigh",
            snr_db=10.0,         # override with --snr_db at runtime
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=32),  # longer pilot for stable equalization
    )
