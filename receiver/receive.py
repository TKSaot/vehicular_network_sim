import numpy as np
from typing import Dict, Any, List, Tuple
from data_link_layer.encoding import bits_to_bytes, parse_length_crc_frame, deinterleave
from data_link_layer.error_correction import get_code
from physical_layer.modulation import demod_hard

# 共通：受信→等化→復号→ビット列
def receive_bits_generic(y: np.ndarray, h: np.ndarray, meta: Dict[str, Any]):
    x = y / np.where(h == 0, 1e-12, h)  # 完全CSIで1タップ等化
    bits_int = demod_hard(x, meta["modulation"])
    if meta.get("interleaver") and meta.get("interleaver_perm") is not None:
        bits_deint = deinterleave(bits_int, meta["interleaver_perm"])
    else:
        bits_deint = bits_int
    code = get_code(meta["coding"])
    bits_dec = code.decode(bits_deint)
    # 送信側の未符号化長に切り詰め
    k = int(meta.get("n_tx_bits_uncoded", len(bits_dec)))
    bits_dec = bits_dec[:k].astype(np.uint8)
    rmeta = {
        "hard_pre_deintl": bits_int,
        "hard_post_deintl": bits_deint,
    }
    return bits_dec, rmeta

# 旧API互換（非UEPパス用）
def receive_bits(cfg, y, h, meta):
    return receive_bits_generic(y, h, meta)

# フレームを1つだけ解釈（先頭から）
def deframe_and_check(bits: np.ndarray) -> Tuple[List[bytes], List[bool]]:
    b = bits_to_bytes(bits)
    if len(b) < 8:
        return [b""], [False]
    # 先頭フレームだけを解釈
    L = int.from_bytes(b[0:4], 'big')
    if len(b) < 8 + L:
        return [b""], [False]
    payload, ok = parse_length_crc_frame(b[:8+L])
    return [payload], [bool(ok)]
