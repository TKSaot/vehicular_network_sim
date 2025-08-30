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
            mtu_bytes=1024,
            interleaver_depth=16,
            fec_scheme="hamming74",
            repeat_k=3,
            strong_header_protection=True,
            header_copies=7,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            # NEW (keep off for text):
            byte_mapping_scheme="none",
            byte_mapping_seed=None,
        ),
        mod=ModulationConfig(scheme="qpsk"),
        chan=ChannelConfig(
            channel_type="awgn",         # 初回安定実行
            snr_db=12.0,
            seed=12345,
            doppler_hz=50.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=PilotConfig(preamble_len=32, pilot_len=16),
    )
