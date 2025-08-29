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
            interleaver_depth=16,        # 深めでフェージング耐性向上
            fec_scheme="hamming74",      # 低SNRで耐性を高めるなら "repeat" + repeat_k を上げる
            repeat_k=3,
            strong_header_protection=True,
            header_copies=7,             # ヘッダ複製数
            header_rep_k=5,              # ヘッダに更なる繰返し
            force_output_on_hdr_fail=True,
            verbose=False,
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
