# data_link_layer/encoding.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from common.utils import (
    bytes_to_bits, bits_to_bytes, append_crc32, verify_and_strip_crc32,
    block_interleave, block_deinterleave
)
from data_link_layer.error_correction import make_fec, RepetitionFEC

FRAME_TYPE_APP_HDR = 0x01
FRAME_TYPE_DATA    = 0x02
HEADER_LEN = 8  # bytes

# ---------- Headers ----------
@dataclass
class LinkHeader:
    version: int = 1
    frame_type: int = FRAME_TYPE_DATA
    seq_no: int = 0
    total_frames: int = 1
    payload_len: int = 0
    def to_bytes(self) -> bytes:
        return bytes([self.version & 0xFF, self.frame_type & 0xFF]) + \
               int(self.seq_no).to_bytes(2,'big') + \
               int(self.total_frames).to_bytes(2,'big') + \
               int(self.payload_len).to_bytes(2,'big')
    @staticmethod
    def from_bytes(b: bytes) -> "LinkHeader":
        if len(b) < HEADER_LEN:
            raise ValueError("Header too short")
        ver = b[0]; ftype = b[1]
        seq = int.from_bytes(b[2:4],'big')
        tot = int.from_bytes(b[4:6],'big')
        plen = int.from_bytes(b[6:8],'big')
        return LinkHeader(ver, ftype, seq, tot, plen)

@dataclass
class LinkFrame:
    header: LinkHeader
    payload_with_crc: bytes
    def to_bytes(self) -> bytes:
        return self.header.to_bytes() + self.payload_with_crc
    @staticmethod
    def from_bytes_safe(b: bytes) -> "LinkFrame":
        hdr = LinkHeader.from_bytes(b[:HEADER_LEN])
        return LinkFrame(hdr, b[HEADER_LEN:])

# ---------- Packetization ----------
def segment_message(app_header: bytes, data: bytes, mtu_bytes: int, header_copies: int = 1) -> List[LinkFrame]:
    frames: List[LinkFrame] = []
    for c in range(max(1, int(header_copies))):
        hdr0 = LinkHeader(frame_type=FRAME_TYPE_APP_HDR, seq_no=c, total_frames=0, payload_len=len(app_header))
        frames.append(LinkFrame(header=hdr0, payload_with_crc=append_crc32(app_header)))
    total_data_frames = (len(data) + mtu_bytes - 1) // mtu_bytes
    for i in range(total_data_frames):
        chunk = data[i*mtu_bytes:(i+1)*mtu_bytes]
        hdr = LinkHeader(frame_type=FRAME_TYPE_DATA, seq_no=header_copies + i, total_frames=0, payload_len=len(chunk))
        frames.append(LinkFrame(header=hdr, payload_with_crc=append_crc32(chunk)))
    total_frames = header_copies + total_data_frames
    for fr in frames:
        fr.header.total_frames = total_frames
    return frames

# ---------- FEC apply / reverse (ハード版：既存) ----------
def apply_fec_and_interleave(frames: List[LinkFrame], fec_scheme: str, repeat_k: int,
                             interleaver_depth: int, strong_header: bool,
                             header_copies: int, header_rep_k: int) -> Tuple[List[np.ndarray], List[int]]:
    enc_frames: List[np.ndarray] = []
    orig_bit_lengths: List[int] = []
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1,int(header_rep_k))) if strong_header else None
    for idx, fr in enumerate(frames):
        bits = bytes_to_bits(fr.to_bytes())
        orig_bit_lengths.append(len(bits))
        enc = fec.encode(bits)
        if strong_header and header_rep is not None and idx < header_copies:
            enc = header_rep.encode(enc)
        inter = block_interleave(enc, interleaver_depth)
        enc_frames.append(inter)
    return enc_frames, orig_bit_lengths

def reverse_fec_and_deinterleave(encoded_frames: List[np.ndarray], original_bit_lengths: List[int],
                                 fec_scheme: str, repeat_k: int, interleaver_depth: int,
                                 strong_header: bool, header_copies: int, header_rep_k: int) -> List[bytes]:
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1,int(header_rep_k))) if strong_header else None
    out: List[bytes] = []
    for idx, enc in enumerate(encoded_frames):
        deinter = block_deinterleave(enc, interleaver_depth)
        dec = deinter
        if strong_header and header_rep is not None and idx < header_copies:
            dec = header_rep.decode(dec)
        dec = fec.decode(dec)
        Lbits = original_bit_lengths[idx]
        if len(dec) > Lbits: dec = dec[:Lbits]
        b = bits_to_bytes(dec)
        out.append(b[: (Lbits + 7)//8])
    return out

# ---------- Hamming(7,4) 用 Chase 風ソフト復号 ----------
def _block_deinterleave_generic(arr: np.ndarray, depth: int, original_len: Optional[int] = None) -> np.ndarray:
    depth = max(1, int(depth))
    arr = np.asarray(arr).reshape(-1)
    if depth == 1:
        return arr[:original_len] if original_len is not None else arr
    L = len(arr); cols = int(np.ceil(L / depth))
    pad = depth*cols - L
    if pad:
        padv = 0.0 if np.issubdtype(arr.dtype, np.floating) else 0
        arr = np.concatenate([arr, np.full(pad, padv, dtype=arr.dtype)])
    mat = arr.reshape(cols, depth).T
    out = mat.reshape(-1)
    return out[:original_len] if original_len is not None else out

def _ham74_encode(d4: np.ndarray) -> np.ndarray:
    d1,d2,d3,d4b = [d4[i] & 1 for i in range(4)]
    p1 = d1 ^ d2 ^ d4b
    p2 = d1 ^ d3 ^ d4b
    p3 = d2 ^ d3 ^ d4b
    return np.array([d1,d2,d3,d4b,p1,p2,p3], dtype=np.uint8)

def _ham74_correct_one(c7: np.ndarray) -> np.ndarray:
    d1,d2,d3,d4,p1,p2,p3 = [c7[i] & 1 for i in range(7)]
    s1 = d1 ^ d2 ^ d4 ^ p1
    s2 = d1 ^ d3 ^ d4 ^ p2
    s3 = d2 ^ d3 ^ d4 ^ p3
    synd = (s1 + (s2<<1) + (s3<<2)) & 0x7
    pos_map = np.array([0,5,6,1,7,2,3,4], dtype=np.uint8)  # 既存実装に整合
    pos = int(pos_map[synd])
    if pos != 0: c7[pos-1] ^= 1
    return c7

def _ham74_decode_chase(bits: np.ndarray, rel: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    r = np.asarray(rel, dtype=np.float32).reshape(-1)
    L = (len(b)//7)*7
    b = b[:L]; r = r[:L]
    out = []
    for i in range(0, L, 7):
        c = b[i:i+7].copy(); w = r[i:i+7].copy()
        idx = np.argsort(w)  # 低信頼→高信頼
        cand_sets = [[], [idx[0]]]
        if len(idx) > 1: cand_sets += [[idx[1]], [idx[0], idx[1]]]
        if len(idx) > 2: cand_sets += [[idx[2]]]

        best_metric = 1e18; best_d4 = c[:4]
        for flips in cand_sets:
            c_try = c.copy()
            for f in flips: c_try[f] ^= 1
            c_fix = _ham74_correct_one(c_try.copy())
            d4 = c_fix[:4].copy()
            cref = _ham74_encode(d4)
            mism = (cref ^ c_try).astype(np.float32)
            metric = float(np.sum(mism * (1.0 + (1.0/(w+1e-6)))))
            if metric < best_metric:
                best_metric = metric; best_d4 = d4
        out.append(best_d4)
    return np.concatenate(out, axis=0).astype(np.uint8)

def reverse_fec_and_deinterleave_soft(encoded_frames_bits: List[np.ndarray],
                                      encoded_frames_rel: List[np.ndarray],
                                      original_bit_lengths: List[int],
                                      fec_scheme: str, repeat_k: int, interleaver_depth: int,
                                      strong_header: bool, header_copies: int, header_rep_k: int
                                      ) -> List[bytes]:
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1,int(header_rep_k))) if strong_header else None
    out: List[bytes] = []
    for idx, (enc_b, enc_r) in enumerate(zip(encoded_frames_bits, encoded_frames_rel)):
        de_b = block_deinterleave(enc_b, interleaver_depth)
        de_r = _block_deinterleave_generic(enc_r, interleaver_depth)
        if strong_header and header_rep is not None and idx < header_copies:
            de_b = header_rep.decode(de_b)  # 信頼度は未使用
        if fec_scheme.lower() == "hamming74":
            dec = _ham74_decode_chase(de_b, de_r)
        elif hasattr(fec, "decode_soft"):
            dec = fec.decode_soft(de_b, de_r)   # ← NEW: conv_k7_* はここを通る
        else:
            dec = fec.decode(de_b)
        Lbits = original_bit_lengths[idx]
        if len(dec) > Lbits: dec = dec[:Lbits]
        b = bits_to_bytes(dec)
        out.append(b[: (Lbits + 7)//8])
    return out


# ---------- Reassembly ----------
def _majority_bytes(blobs: List[bytes]) -> bytes:
    if not blobs: return b""
    min_len = min(len(x) for x in blobs)
    if min_len == 0: return b""
    arr_bits = [bytes_to_bits(x[:min_len]) for x in blobs]
    M = np.stack(arr_bits, axis=0); s = np.sum(M, axis=0)
    out_bits = (s >= (M.shape[0]//2 + 1)).astype(np.uint8)
    from common.utils import bits_to_bytes
    return bits_to_bytes(out_bits)[:min_len]

def reassemble_and_check(frames_bytes: List[bytes], header_copies: int = 1,
                         drop_bad: bool = False, verbose: bool = False):
    n_frames = len(frames_bytes)
    H = max(1,int(header_copies)); H = min(H, n_frames)
    hdr_ok = False; hdr_major = False; hdr_payload = b""
    header_payload_crc_blobs: List[bytes] = []
    for i in range(H):
        b = frames_bytes[i]; payload_crc = b[HEADER_LEN:]; header_payload_crc_blobs.append(payload_crc)
        ok, payload = verify_and_strip_crc32(payload_crc)
        if ok and not hdr_ok: hdr_ok = True; hdr_payload = payload
    if not hdr_ok:
        voted = _majority_bytes(header_payload_crc_blobs)
        ok, payload = verify_and_strip_crc32(voted)
        if ok: hdr_ok = True; hdr_major = True; hdr_payload = payload
    data = bytearray(); n_bad = 0
    for i in range(H, n_frames):
        b = frames_bytes[i]; payload_crc = b[HEADER_LEN:]
        ok, payload = verify_and_strip_crc32(payload_crc)
        if not ok:
            n_bad += 1
            if drop_bad: payload = b""
            if verbose: print(f"[WARN] DATA frame {i-H}/{n_frames-H} CRC failed")
        data.extend(payload)
    stats = {
        "all_crc_ok": bool(hdr_ok and (n_bad == 0)),
        "app_header_crc_ok": bool(hdr_ok),
        "app_header_recovered_via_majority": bool(hdr_major),
        "n_bad_frames": int(n_bad),
        "n_frames": int(n_frames),
    }
    return hdr_payload, bytes(data), stats
