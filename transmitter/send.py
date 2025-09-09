# transmitter/send.py
from __future__ import annotations
from typing import Dict, List, Tuple
from common.backend import xp, to_xp
from common.config import SimulationConfig
from common.utils import bytes_to_bits
from common.byte_mapping import map_bytes
from data_link_layer.encoding import segment_message, apply_fec_and_interleave
from physical_layer.modulation import Modulator, build_phy_frame

def build_transmission(app_header_bytes: bytes, payload_bytes: bytes, cfg: SimulationConfig):
    # Byte mapping before segmentation (DATA only)
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    mapped_payload = map_bytes(
        payload_bytes,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed
    )

    # 1) Link segmentation
    frames = segment_message(
        app_header_bytes,
        mapped_payload,
        mtu_bytes=cfg.link.mtu_bytes,
        header_copies=cfg.link.header_copies
    )

    # 2) Apply FEC + interleaving
    enc_bits_list, orig_bit_lengths = apply_fec_and_interleave(
        frames,
        fec_scheme=cfg.link.fec_scheme,
        repeat_k=cfg.link.repeat_k,
        interleaver_depth=cfg.link.interleaver_depth,
        strong_header=cfg.link.strong_header_protection,
        header_copies=cfg.link.header_copies,
        header_rep_k=cfg.link.header_rep_k
    )

    # 3) PHY: preamble + pilots + data symbols per frame (GPU if available)
    mod = Modulator(cfg.mod.scheme)
    frame_symbol_ranges: List[Tuple[int,int]] = []
    pilots_tx: List["xp.ndarray"] = []
    data_symbol_counts: List[int] = []
    all_syms: List["xp.ndarray"] = []

    cursor = 0
    for bits in enc_bits_list:
        syms, pilot, data_syms = build_phy_frame(bits, mod,
            preamble_len_bits=cfg.pilot.preamble_len,
            pilot_len_symbols=cfg.pilot.pilot_len)
        all_syms.append(syms)
        pilots_tx.append(pilot)
        data_symbol_counts.append(int(data_syms.size))
        frame_symbol_ranges.append((cursor, cursor+int(syms.size)))
        cursor += int(syms.size)

    tx_symbols = xp.concatenate(all_syms) if all_syms else xp.zeros(0, dtype=xp.complex128)
    tx_meta = {
        "mod_scheme": cfg.mod.scheme,
        "frame_symbol_ranges": frame_symbol_ranges,
        "data_symbol_counts": data_symbol_counts,
        "pilots_tx": pilots_tx,
        "orig_bit_lengths": orig_bit_lengths,
    }
    return tx_symbols, tx_meta
