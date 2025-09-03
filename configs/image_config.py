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
DEFAULT_MODALITY: Literal["edge","depth","segmentation"] = "depth"

def build_config(modality: Literal["edge","depth","segmentation"] = DEFAULT_MODALITY) -> SimulationConfig:
    return SimulationConfig(
        app=AppConfig(
            modality=modality,
            validate_image_mode=True,
            # --- Segmentation TX-side (semantic noise OFF by default) ---
            seg_strip_white=True,
            seg_white_thresh=250,
            seg_tx_noise_p=0.0,
            seg_tx_noise_seed=None,
            # --- Decoder options moved from env ---
            segdec=SegDecoderConfig(
                fallback="uniform",      # out-of-range IDs → Uniform{0..K-1}
                seed=123,
                maj3x3=True,             # 3×3 majority
                maj_iters=2,             # repeat twice
                use_edge_guidance=True,  # use edge PNG to protect boundaries
                # NOTE: ここは「受信後の補正済みエッジ画像」を指すように適宜変更してください
                edge_guide_path="examples/edge_00001_.png",
            ),
            edgedec=EdgeDecoderConfig(
                denoise="maj3",          # 3×3 majority on {0,1}
                iters=1,
                thresh=None,             # default (過半数=5)
                preserve_lines=True,     # 細線を消さない保護ヒューリスティック
            ),
            depthdec=DepthDecoderConfig(
                filt="median3",          # or "median5" / "bilateral5"
                iters=1,
                sigma_s=1.6, sigma_r=12.0,
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
            byte_mapping_scheme="permute",   # try "permute" to fully randomize
            byte_mapping_seed=None,          # None -> use chan.seed
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
