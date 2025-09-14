# configs/text_config.py
from __future__ import annotations
from common.config import (
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig
)

INPUT = "examples/sample.txt"
OUTPUT_ROOT = "outputs"

def build_config() -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(modality="text", validate_image_mode=False),
        link=LinkConfig(
            mtu_bytes=256,
            interleaver_depth=256,         # deeper to disperse burst errors
            fec_scheme="hamming74",
            repeat_k=1,                    # note: repeat_k is ignored for hamming74
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            byte_mapping_scheme="permute",
            byte_mapping_seed=None,
        ),
        mod=ModulationConfig(scheme="bpsk"),  # prioritize robustness for text
        chan=ChannelConfig(
            channel_type="rayleigh",
            snr_db=12.0,                   # override with --snr_db
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=32),
    )
