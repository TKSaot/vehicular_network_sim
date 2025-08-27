import numpy as np
from typing import List, Tuple, Dict, Any

from common.config import SimConfig, ebn0_to_esn0_db
from data_link_layer.encoding import (
    bytes_to_bits, build_length_crc_frame, interleave
)
from data_link_layer.error_correction import get_code
from physical_layer.modulation import modulate, bits_per_symbol

# ---- ユーティリティ ----
def assemble_bitstream(frames: List[bytes]) -> np.ndarray:
    """複数フレームをそのまま連結してビット列に（非UEP用）"""
    if not frames:
        return np.zeros(0, dtype=np.uint8)
    bits = [bytes_to_bits(f) for f in frames]
    return np.concatenate(bits)

def _code_rate(name: str) -> float:
    n = name.lower()
    if n == "none": return 1.0
    if n == "repetition3": return 1.0/3.0
    if n == "hamming74": return 4.0/7.0
    raise ValueError("Unknown code: " + name)

# ---- 単一フレーム（小さいpayload向け） ----
def build_frames(payload: bytes, max_frame_bits: int) -> List[bytes]:
    frame = build_length_crc_frame(payload)
    need_bits = len(frame) * 8
    if need_bits > max_frame_bits:
        raise ValueError(
            f"Frame size {need_bits} bits exceeds max_frame_bits={max_frame_bits}. "
            f"Increase LinkConfig.max_frame_bits or reduce payload size."
        )
    return [frame]

# ---- 追加：フラグメンテーション（大きいpayload向け） ----
def _framing_overhead_bytes() -> int:
    return len(build_length_crc_frame(b""))  # 例: 8 bytes

def fragment_payload(payload: bytes, max_frame_bits: int) -> List[bytes]:
    ov = _framing_overhead_bytes()
    max_frame_bytes = max(1, max_frame_bits // 8)
    max_payload_bytes = max_frame_bytes - ov
    if max_payload_bytes <= 0:
        raise ValueError(
            f"max_frame_bits={max_frame_bits} is too small (overhead={ov} bytes)."
        )
    if len(payload) <= max_payload_bytes:
        return [payload]
    chunks: List[bytes] = []
    i = 0
    while i < len(payload):
        chunks.append(payload[i:i+max_payload_bytes])
        i += max_payload_bytes
    return chunks

def build_frames_from_chunks(chunks: List[bytes]) -> List[bytes]:
    return [build_length_crc_frame(c) for c in chunks]

# ---- 送信：変調・符号化・インターリーブ・EsN0 ----
def transmit_bits(cfg: SimConfig, tx_bits_uncoded: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    return transmit_bits_custom(
        snr_type=cfg.phy.snr_type,
        snr_db=cfg.phy.snr_db,
        modulation=cfg.phy.modulation,
        coding=cfg.link.coding,
        interleaver=cfg.link.interleaver,
        interleaver_seed=cfg.link.interleaver_seed,
        tx_bits_uncoded=tx_bits_uncoded,
        code_rate_override=cfg.phy.code_rate_override,
    )

def transmit_bits_custom(
    snr_type: str,
    snr_db: float,
    modulation: str,
    coding: str,
    interleaver: bool,
    interleaver_seed: int,
    tx_bits_uncoded: np.ndarray,
    code_rate_override: float | None = None,
    snr_boost_db: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    code = get_code(coding)
    coded = code.encode(tx_bits_uncoded)

    if interleaver:
        inter, perm = interleave(coded, interleaver_seed)
    else:
        inter, perm = coded, None

    syms = modulate(inter, modulation)

    bps = bits_per_symbol(modulation)
    rate = code_rate_override if (snr_type == "EbN0" and code_rate_override is not None) else _code_rate(coding)
    if snr_type == "EsN0":
        esn0_db = snr_db + snr_boost_db
    else:
        esn0_db = ebn0_to_esn0_db(snr_db, bps, rate) + snr_boost_db

    meta = {
        "bits_per_symbol": bps,
        "code_rate": rate,
        "esn0_db": esn0_db,
        "tx_bits_after_interleave": inter.copy(),
        "n_tx_after_interleave": len(inter),
        "n_tx_bits_uncoded": len(tx_bits_uncoded),
        "interleaver_perm": perm,
        "modulation": modulation,
        "coding": coding,
        "interleaver": interleaver,
    }
    return syms, meta
