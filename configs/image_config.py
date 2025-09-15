# configs/image_config.py
from __future__ import annotations
import os
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

def _eep_link() -> LinkConfig:
    # Low-SNR robust EEP baseline (BPSK + Hamming74 + deeper ILV + moderate pilots)
    return LinkConfig(
        mtu_bytes=256,
        interleaver_depth=128,
        fec_scheme="hamming74",
        repeat_k=1,                     # ignored for hamming74
        strong_header_protection=True,
        header_copies=7,
        header_rep_k=5,
        force_output_on_hdr_fail=True,
        verbose=False,
        byte_mapping_scheme="permute",
        byte_mapping_seed=None,
    )

def _mod_cfg() -> ModulationConfig:
    return ModulationConfig(scheme="bpsk")  # robust for low SNR

def _pilot_cfg() -> PilotConfig:
    return PilotConfig(preamble_len=32, pilot_len=32)

# ---------- UEP overrides (per modality) ----------
# Each entry returns (mtu_bytes, interleaver_depth, pilot_len, header_copies)
UEP_TABLE = {
    "S": {
        "text":        (128, 256, 48, 11),
        "edge":        (192, 192, 48,  9),
        "segmentation":(192, 192, 48,  9),
        "depth":       (288, 128, 24,  5),
    },
    "G": {
        "text":        (192, 192, 32,  7),
        "edge":        (224, 160, 32,  7),
        "segmentation":(192, 192, 40,  9),
        "depth":       (192, 256, 48,  9),
    },
    "B": {
        "text":        (160, 224, 40,  9),
        "edge":        (208, 176, 40,  8),
        "segmentation":(208, 176, 40,  8),
        "depth":       (272, 128, 24,  6),
    },
}

def build_config(modality: Literal["edge","depth","segmentation"] = DEFAULT_MODALITY) -> SimulationConfig:
    uep_mode = os.getenv("UEP_MODE", "off").strip().upper()
    link = _eep_link()
    mod  = _mod_cfg()
    pilot = _pilot_cfg()

    if uep_mode in UEP_TABLE:
        if modality not in UEP_TABLE[uep_mode]:
            raise ValueError(f"Unknown modality for UEP: {modality}")
        mtu, ilv, pil, hdr = UEP_TABLE[uep_mode][modality]
        link = LinkConfig(
            mtu_bytes=mtu,
            interleaver_depth=ilv,
            fec_scheme="hamming74",
            repeat_k=1,
            strong_header_protection=True,
            header_copies=hdr,
            header_rep_k=5,
            force_output_on_hdr_fail=True,
            verbose=False,
            byte_mapping_scheme="permute",
            byte_mapping_seed=None,
        )
        pilot = PilotConfig(preamble_len=32, pilot_len=pil)

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
        link=link,
        mod=mod,
        chan=ChannelConfig(
            channel_type="rayleigh",
            snr_db=10.0,   # override with --snr_db
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=pilot,
    )
