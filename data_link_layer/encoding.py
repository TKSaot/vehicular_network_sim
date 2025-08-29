# data_link_layer/encoding.py
"""
Link-layer framing: segmentation, headers, CRC, interleaving, and FEC application.

Frame types:
- 0x01: APP_HEADER  (this may be repeated 'header_copies' times up front)
- 0x02: DATA

Key robustness:
- Data reassembly ignores header.payload_len (which may be corrupted) and instead
  treats the whole tail (after 8B link header) as [payload || CRC32].
- APP header: multiple copies + FEC + Repetition(k), with bitwise-majority
  fallback to reconstruct a CRC-valid header at very low SNR.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from common.utils import bytes_to_bits, bits_to_bytes, append_crc32, verify_and_strip_crc32,\
                         block_interleave, block_deinterleave
from data_link_layer.error_correction import make_fec, RepetitionFEC

FRAME_TYPE_APP_HDR = 0x01
FRAME_TYPE_DATA    = 0x02

HEADER_LEN = 8  # bytes

@dataclass
class LinkHeader:
    version: int = 1
    frame_type: int = FRAME_TYPE_DATA
    seq_no: int = 0
    total_frames: int = 1
    payload_len: int = 0  # in bytes, before CRC

    def to_bytes(self) -> bytes:
        return bytes([self.version & 0xFF, self.frame_type & 0xFF]) + \
               int(self.seq_no).to_bytes(2, 'big') + \
               int(self.total_frames).to_bytes(2, 'big') + \
               int(self.payload_len).to_bytes(2, 'big')

    @staticmethod
    def from_bytes(b: bytes) -> 'LinkHeader':
        if len(b) < HEADER_LEN:
            raise ValueError("Header too short")
        ver = b[0]
        ftype = b[1]
        seq = int.from_bytes(b[2:4], 'big')
        tot = int.from_bytes(b[4:6], 'big')
        plen = int.from_bytes(b[6:8], 'big')
        return LinkHeader(version=ver, frame_type=ftype, seq_no=seq, total_frames=tot, payload_len=plen)

@dataclass
class LinkFrame:
    header: LinkHeader
    payload_with_crc: bytes  # payload || CRC32

    def to_bytes(self) -> bytes:
        return self.header.to_bytes() + self.payload_with_crc

    @staticmethod
    def from_bytes_unsafe(b: bytes) -> 'LinkFrame':
        """
        Parse using header.payload_len (unsafe if header corrupted).
        """
        hdr = LinkHeader.from_bytes(b[:HEADER_LEN])
        payload_with_crc = b[HEADER_LEN:HEADER_LEN + hdr.payload_len + 4]
        return LinkFrame(header=hdr, payload_with_crc=payload_with_crc)

    @staticmethod
    def from_bytes_safe(b: bytes) -> 'LinkFrame':
        """
        Safe parse that DOES NOT trust header.payload_len.
        It takes the entire tail after 8B header as [payload || CRC32].
        """
        hdr = LinkHeader.from_bytes(b[:HEADER_LEN])
        payload_with_crc = b[HEADER_LEN:]
        return LinkFrame(header=hdr, payload_with_crc=payload_with_crc)

# ------------- Packetization -------------
def segment_message(app_header: bytes, data: bytes, mtu_bytes: int, header_copies: int = 1) -> List[LinkFrame]:
    """
    First 'header_copies' frames carry the APP_HEADER (frame_type=APP_HDR).
    Following frames carry data. Each frame carries CRC over its payload (not including link header).
    """
    frames: List[LinkFrame] = []

    # Header frames
    for c in range(max(1, int(header_copies))):
        hdr0 = LinkHeader(frame_type=FRAME_TYPE_APP_HDR, seq_no=c, total_frames=0, payload_len=len(app_header))
        frames.append(LinkFrame(header=hdr0, payload_with_crc=append_crc32(app_header)))

    # Data frames
    total_data_frames = (len(data) + mtu_bytes - 1) // mtu_bytes
    for i in range(total_data_frames):
        chunk = data[i*mtu_bytes:(i+1)*mtu_bytes]
        # seq_no starts after header copies
        hdr = LinkHeader(frame_type=FRAME_TYPE_DATA, seq_no=header_copies + i, total_frames=0, payload_len=len(chunk))
        frames.append(LinkFrame(header=hdr, payload_with_crc=append_crc32(chunk)))

    # total_frames (for completeness; not relied on in RX)
    total_frames = header_copies + total_data_frames
    for fr in frames[:header_copies]:
        fr.header.total_frames = total_frames
    for fr in frames[header_copies:]:
        fr.header.total_frames = total_frames
    return frames

def apply_fec_and_interleave(frames: List[LinkFrame], fec_scheme: str, repeat_k: int,
                             interleaver_depth: int, strong_header: bool,
                             header_copies: int, header_rep_k: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert frames to bit arrays, apply FEC (+ header repetition if enabled) and interleaving per frame.
    Returns (list_of_bits, list_of_original_bit_lengths).
    """
    encoded_frames: List[np.ndarray] = []
    original_bit_lengths: List[int] = []

    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1, int(header_rep_k))) if strong_header else None

    for idx, fr in enumerate(frames):
        b = fr.to_bytes()
        bits = bytes_to_bits(b)
        original_bit_lengths.append(len(bits))

        enc = fec.encode(bits)
        if strong_header and header_rep is not None and idx < header_copies:
            enc = header_rep.encode(enc)

        inter = block_interleave(enc, interleaver_depth)
        encoded_frames.append(inter)

    return encoded_frames, original_bit_lengths

def reverse_fec_and_deinterleave(encoded_frames: List[np.ndarray], original_bit_lengths: List[int],
                                 fec_scheme: str, repeat_k: int, interleaver_depth: int,
                                 strong_header: bool, header_copies: int, header_rep_k: int) -> List[bytes]:
    """
    Reverse interleaving and FEC to recover raw frame bytes (without CRC verification).
    Header frames (first header_copies) are decoded with Repetition first.
    Returns list of frame bytes (may be corrupted).
    """
    fec = make_fec(fec_scheme, repeat_k=repeat_k)
    header_rep = RepetitionFEC(k=max(1, int(header_rep_k))) if strong_header else None

    out_bytes: List[bytes] = []
    for idx, enc in enumerate(encoded_frames):
        deinter = block_deinterleave(enc, interleaver_depth)
        dec = deinter
        if strong_header and header_rep is not None and idx < header_copies:
            dec = header_rep.decode(dec)
        dec = fec.decode(dec)

        Lbits = original_bit_lengths[idx]
        if len(dec) > Lbits:
            dec = dec[:Lbits]
        b = bits_to_bytes(dec)
        out_bytes.append(b[: (Lbits + 7)//8])
    return out_bytes

def _majority_bytes(blobs: List[bytes]) -> bytes:
    """
    Bitwise majority vote across same-length byte sequences.
    Returns the majority-voted bytes (length = min length of inputs).
    """
    if not blobs:
        return b""
    min_len = min(len(x) for x in blobs)
    if min_len == 0:
        return b""
    arr_bits = [bytes_to_bits(x[:min_len]) for x in blobs]
    M = np.stack(arr_bits, axis=0)
    s = np.sum(M, axis=0)
    out_bits = (s >= (M.shape[0]//2 + 1)).astype(np.uint8)
    return bits_to_bytes(out_bits)[:min_len]

def reassemble_and_check(frames_bytes: List[bytes], header_copies: int = 1,
                         drop_bad: bool = False, verbose: bool = False) -> Tuple[bytes, bytes, dict]:
    """
    Parse frames bytes, verify CRC per frame (payload CRC only).
    - First 'header_copies' frames are APP_HEADER. We try CRC individually; if all fail,
      we do bitwise-majority across their payload||CRC and verify again.
    - Data frames: ignore header.payload_len and slice tail safely as [payload||CRC32].

    Returns (app_header_bytes, data_bytes, stats_dict)
    stats:
      {
        "all_crc_ok": bool,
        "app_header_crc_ok": bool,
        "app_header_recovered_via_majority": bool,
        "n_bad_frames": int,  # DATAのみカウント（ヘッダ失敗は app_header_crc_ok に反映）
        "n_frames": int
      }
    """
    n_frames = len(frames_bytes)
    H = max(1, int(header_copies))
    H = min(H, n_frames)

    # --- APP HEADER(s) ---
    hdr_ok = False
    hdr_majority_used = False
    hdr_payload = b""

    header_payload_crc_blobs: List[bytes] = []
    for i in range(H):
        b = frames_bytes[i]
        # Safe: use full tail as payload||CRC
        payload_crc = b[HEADER_LEN:]
        header_payload_crc_blobs.append(payload_crc)
        ok, payload = verify_and_strip_crc32(payload_crc)
        if ok and not hdr_ok:
            hdr_ok = True
            hdr_payload = payload  # 16B

    if not hdr_ok:
        # majority across header copies
        voted = _majority_bytes(header_payload_crc_blobs)
        ok, payload = verify_and_strip_crc32(voted)
        if ok:
            hdr_ok = True
            hdr_majority_used = True
            hdr_payload = payload

    # --- DATA frames ---
    data = bytearray()
    n_bad = 0
    for i in range(H, n_frames):
        b = frames_bytes[i]
        # Safe tail slicing
        payload_crc = b[HEADER_LEN:]
        ok, payload = verify_and_strip_crc32(payload_crc)
        if not ok:
            n_bad += 1
            if drop_bad:
                payload = b""
            if verbose:
                print(f"[WARN] DATA frame {i-H}/{n_frames-H} CRC failed")
        data.extend(payload)

    stats = {
        "all_crc_ok": bool(hdr_ok and (n_bad == 0)),
        "app_header_crc_ok": bool(hdr_ok),
        "app_header_recovered_via_majority": bool(hdr_majority_used),
        "n_bad_frames": int(n_bad),
        "n_frames": int(n_frames),
    }
    return hdr_payload, bytes(data), stats
