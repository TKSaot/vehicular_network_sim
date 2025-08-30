# configs/image_config.py
from __future__ import annotations
from typing import Literal
from common.config import (
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig
)

INPUTS = {
    "edge": "examples/edge_00001_.png",
    "depth": "examples/depth_00001_.png",
    "segmentation": "examples/segmentation_00001_.png",
}
OUTPUT_ROOT = "outputs"
DEFAULT_MODALITY: Literal["edge","depth","segmentation"] = "depth"

def build_config(modality: Literal["edge","depth","segmentation"] = DEFAULT_MODALITY) -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(modality=modality, validate_image_mode=True),
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
            # NEW:
            byte_mapping_scheme="permute",   # try "permute" to fully randomize
            byte_mapping_seed=None,              # None -> use chan.seed
        ),
        mod=ModulationConfig(scheme="qpsk"),
        chan=ChannelConfig(
            channel_type="rayleigh",  # vehicular 既定
            snr_db=10.0,
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=16),
    )
