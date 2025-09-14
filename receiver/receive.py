# receiver/receive.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from common.backend import to_xp
from common.config import SimulationConfig
from data_link_layer.encoding import (
    reverse_fec_and_deinterleave,
    reverse_fec_and_deinterleave_soft,
    reassemble_and_check,
)
from physical_layer.modulation import Modulator
from channel.channel_model import equalize

def recover_from_symbols(rx_symbols, tx_meta: Dict, cfg: SimulationConfig) -> Tuple[bytes, bytes, dict]:
    mod = Modulator(tx_meta["mod_scheme"])
    frames_bits: List[np.ndarray] = []
    frames_rel:  List[np.ndarray] = []
    est_channels: List[complex] = []

    z = to_xp(rx_symbols)  # GPU/CPU どちらでも OK

    for i, (start, end) in enumerate(tx_meta["frame_symbol_ranges"]):
        syms = z[start:end]
        pilot_len = len(tx_meta["pilots_tx"][i])
        preamble_len = cfg.pilot.preamble_len
        rx_pilot = syms[preamble_len:preamble_len+pilot_len]
        rx_data  = syms[preamble_len+pilot_len:]

        eq_data, h_hat = equalize(rx_data, tx_meta["pilots_tx"][i], rx_pilot)
        est_channels.append(h_hat)

        bits, rel = mod.demodulate_with_reliability(eq_data)  # ← LLR 近似
        frames_bits.append(bits.astype(np.uint8))
        frames_rel.append(rel.astype(np.float32))

    if cfg.link.fec_scheme.lower() == "hamming74":
        raw_frame_bytes = reverse_fec_and_deinterleave_soft(
            frames_bits, frames_rel,
            tx_meta["orig_bit_lengths"],
            fec_scheme=cfg.link.fec_scheme,
            repeat_k=cfg.link.repeat_k,
            interleaver_depth=cfg.link.interleaver_depth,
            strong_header=cfg.link.strong_header_protection,
            header_copies=cfg.link.header_copies,
            header_rep_k=cfg.link.header_rep_k
        )
    else:
        raw_frame_bytes = reverse_fec_and_deinterleave(
            frames_bits,
            tx_meta["orig_bit_lengths"],
            fec_scheme=cfg.link.fec_scheme,
            repeat_k=cfg.link.repeat_k,
            interleaver_depth=cfg.link.interleaver_depth,
            strong_header=cfg.link.strong_header_protection,
            header_copies=cfg.link.header_copies,
            header_rep_k=cfg.link.header_rep_k
        )

    app_hdr_bytes, payload_bytes, stats = reassemble_and_check(
        raw_frame_bytes,
        header_copies=cfg.link.header_copies,
        drop_bad=cfg.link.drop_bad_frames,
        verbose=cfg.link.verbose
    )
    stats["h_estimates"] = est_channels
    return app_hdr_bytes, payload_bytes, stats
