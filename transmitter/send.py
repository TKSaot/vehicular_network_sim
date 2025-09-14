# transmitter/send.py
from __future__ import annotations
from typing import List, Tuple
import os

from common.backend import xp, to_xp
from common.config import SimulationConfig
from common.byte_mapping import map_bytes
from data_link_layer.encoding import segment_message, apply_fec_and_interleave
from physical_layer.modulation import Modulator, build_phy_frame

# tqdm はオプション（環境変数 TX_TQDM=1 の時だけ表示）
_USE_TQDM = str(os.getenv("TX_TQDM", "0")).strip().lower() in {"1","true","yes","y","on"}
if _USE_TQDM:
    try:
        from tqdm.auto import tqdm as _tqdm_tx
    except Exception:
        _USE_TQDM = False
        _tqdm_tx = None
else:
    _tqdm_tx = None

def build_transmission(app_header_bytes: bytes, payload_bytes: bytes, cfg: SimulationConfig):
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    mapped_payload = map_bytes(
        payload_bytes, mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme, seed=mapping_seed
    )

    frames = segment_message(
        app_header_bytes, mapped_payload,
        mtu_bytes=cfg.link.mtu_bytes, header_copies=cfg.link.header_copies
    )

    enc_bits_list, orig_bit_lengths = apply_fec_and_interleave(
        frames, fec_scheme=cfg.link.fec_scheme, repeat_k=cfg.link.repeat_k,
        interleaver_depth=cfg.link.interleaver_depth,
        strong_header_protection=cfg.link.strong_header_protection,
        header_copies=cfg.link.header_copies,
        header_rep_k=cfg.link.header_rep_k
    )

    mod = Modulator(cfg.mod.scheme)
    frame_symbol_ranges: List[Tuple[int,int]] = []
    pilots_tx: List = []
    data_symbol_counts: List[int] = []
    all_syms: List = []
    cursor = 0

    iterator = range(len(enc_bits_list))
    bar = None
    if _USE_TQDM and _tqdm_tx is not None:
        bar = _tqdm_tx(total=len(enc_bits_list), desc="TX build (frames)", unit="frm")

    for idx in iterator:
        bits = enc_bits_list[idx]
        syms, pilot, data_syms = build_phy_frame(
            bits, mod,
            preamble_len_bits=cfg.pilot.preamble_len,
            pilot_len_symbols=cfg.pilot.pilot_len
        )
        # backend 統一
        syms = to_xp(syms, dtype=xp.complex128)
        pilot = to_xp(pilot, dtype=xp.complex128)
        data_syms = to_xp(data_syms, dtype=xp.complex128)

        all_syms.append(syms)
        pilots_tx.append(pilot)
        data_symbol_counts.append(int(data_syms.size))
        L = int(syms.size)
        frame_symbol_ranges.append((cursor, cursor + L))
        cursor += L

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    tx_symbols = xp.concatenate(all_syms) if all_syms else xp.zeros(0, dtype=xp.complex128)
    tx_meta = {
        "mod_scheme": cfg.mod.scheme,
        "frame_symbol_ranges": frame_symbol_ranges,
        "data_symbol_counts": data_symbol_counts,
        "pilots_tx": pilots_tx,
        "orig_bit_lengths": orig_bit_lengths,
    }
    return tx_symbols, tx_meta
