# receiver/receive.py
"""
Receiver pipeline: channel equalization -> demod -> deinterleave -> FEC decode -> CRC check -> reassemble -> application decode.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from common.config import SimulationConfig
from data_link_layer.encoding import reverse_fec_and_deinterleave, reassemble_and_check
from physical_layer.modulation import Modulator
from channel.channel_model import equalize

def recover_from_symbols(rx_symbols: np.ndarray, tx_meta: Dict, cfg: SimulationConfig) -> Tuple[bytes, bytes, dict]:
    """
    Split concatenated rx_symbols back into frames using tx_meta, run equalization and demodulation,
    then reverse interleaving/FEC and reassemble.
    Returns (app_header_bytes, payload_bytes, stats_dict)
    """
    mod = Modulator(tx_meta["mod_scheme"])
    frames_bits: List[np.ndarray] = []
    est_channels: List[complex] = []

    for i, (start, end) in enumerate(tx_meta["frame_symbol_ranges"]):
        syms = rx_symbols[start:end]
        pilot_len = len(tx_meta["pilots_tx"][i])
        preamble_len = cfg.pilot.preamble_len
        rx_pilot = syms[preamble_len:preamble_len+pilot_len]
        rx_data = syms[preamble_len+pilot_len:]

        # Equalize using pilots
        eq_data, h_hat = equalize(rx_data, tx_meta["pilots_tx"][i], rx_pilot)
        est_channels.append(h_hat)

        # Demod to bits
        bits = mod.demodulate(eq_data)
        frames_bits.append(bits)

    # Reverse interleaving and FEC per frame
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

    # Reassemble and CRC-check (header copies aware)
    app_hdr_bytes, payload_bytes, stats = reassemble_and_check(
        raw_frame_bytes,
        header_copies=cfg.link.header_copies,
        drop_bad=cfg.link.drop_bad_frames,
        verbose=cfg.link.verbose
    )

    # attach channel estimates (for logging)
    stats["h_estimates"] = est_channels
    return app_hdr_bytes, payload_bytes, stats
