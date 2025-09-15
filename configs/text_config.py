# configs/text_config.py
from __future__ import annotations
import os
from common.config import (
    SimulationConfig, AppConfig, LinkConfig, ModulationConfig, ChannelConfig, PilotConfig
)

INPUT = "examples/sample.txt"
OUTPUT_ROOT = "outputs"

def _eep_link() -> LinkConfig:
    return LinkConfig(
        mtu_bytes=256,
        interleaver_depth=256,         # deeper for text to disperse bursts
        fec_scheme="hamming74",
        repeat_k=1,
        strong_header_protection=True,
        header_copies=7,
        header_rep_k=5,
        force_output_on_hdr_fail=True,
        verbose=False,
        byte_mapping_scheme="permute",
        byte_mapping_seed=None,
    )

def _mod_cfg() -> ModulationConfig:
    return ModulationConfig(scheme="bpsk")

def _pilot_cfg() -> PilotConfig:
    return PilotConfig(preamble_len=32, pilot_len=32)

UEP_TABLE = {
    "S": (128, 256, 48, 11),
    "G": (192, 192, 32,  7),
    "B": (160, 224, 40,  9),
}

def build_config() -> SimulationConfig:
    uep_mode = os.getenv("UEP_MODE", "off").strip().upper()
    link = _eep_link()
    mod  = _mod_cfg()
    pilot = _pilot_cfg()

    if uep_mode in UEP_TABLE:
        mtu, ilv, pil, hdr = UEP_TABLE[uep_mode]
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
        app=AppConfig(modality="text", validate_image_mode=False),
        link=link,
        mod=mod,
        chan=ChannelConfig(
            channel_type="rayleigh",
            snr_db=12.0,   # override with --snr_db
            seed=12345,
            doppler_hz=30.0,
            symbol_rate=1e6,
            block_fading=False,
        ),
        pilot=pilot,
    )
