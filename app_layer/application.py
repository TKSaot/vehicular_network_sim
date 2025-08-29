# app_layer/application.py
"""
Application-layer serializers for different modalities:
- text (.txt, UTF-8 by default)
- edge (binary image, PNG)
- depth (grayscale image, PNG)
- segmentation (RGB image, PNG)

We serialize *content*, not raw file bytes, to avoid container metadata corruption.
Images are loaded via Pillow into numpy arrays; text is handled as Unicode strings.
The application header encodes what is needed to reconstruct content at the receiver.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from PIL import Image

# ----------------- App Header -----------------
@dataclass
class AppHeader:
    version: int
    modality: Literal["text", "edge", "depth", "segmentation"]
    height: int = 0
    width: int = 0
    channels: int = 0
    bits_per_sample: int = 8
    payload_len_bytes: int = 0   # number of data bytes that follow

    def to_bytes(self) -> bytes:
        # Fixed 16-byte header
        mod_code = {"text":0, "edge":1, "depth":2, "segmentation":3}[self.modality]
        b = bytearray(16)
        b[0] = self.version & 0xFF
        b[1] = mod_code & 0xFF
        b[2:4] = int(self.height).to_bytes(2, 'big')
        b[4:6] = int(self.width).to_bytes(2, 'big')
        b[6] = self.channels & 0xFF
        b[7] = self.bits_per_sample & 0xFF
        b[8:12] = int(self.payload_len_bytes).to_bytes(4, 'big')
        return bytes(b)

    @staticmethod
    def from_bytes(b: bytes) -> 'AppHeader':
        if len(b) < 16:
            raise ValueError(f"AppHeader bytes too short: {len(b)} < 16 (likely header corruption).")
        version = b[0]
        code = b[1]
        mapping = {0:"text",1:"edge",2:"depth",3:"segmentation"}
        if code not in mapping:
            raise ValueError(f"Invalid modality code in AppHeader: {code} (likely header CRC failure / severe channel errors).")
        modality = mapping[code]
        height = int.from_bytes(b[2:4], 'big')
        width  = int.from_bytes(b[4:6], 'big')
        channels = b[6]
        bps = b[7]
        payload_len = int.from_bytes(b[8:12], 'big')
        return AppHeader(version=version, modality=modality, height=height, width=width,
                         channels=channels, bits_per_sample=bps, payload_len_bytes=payload_len)

# ----------------- Serialization -----------------
def load_text_as_bytes(path: str, encoding: str="utf-8") -> bytes:
    with open(path, "r", encoding=encoding) as f:
        txt = f.read()
    return txt.encode(encoding)

def text_bytes_to_string(b: bytes, encoding: str="utf-8", errors: str="replace") -> str:
    return b.decode(encoding, errors=errors)

def load_image_to_array(path: str, modality: str, validate_mode: bool = True) -> np.ndarray:
    im = Image.open(path)
    if modality == "edge":
        if validate_mode and im.mode not in ("L","1"):
            im = im.convert("L")
        arr = np.array(im)
        if arr.ndim == 3:
            arr = np.array(im.convert("L"))
        arr = (arr >= 128).astype(np.uint8) * 255
        return arr
    elif modality == "depth":
        if validate_mode and im.mode != "L":
            im = im.convert("L")
        arr = np.array(im)
        return arr.astype(np.uint8)
    elif modality == "segmentation":
        if validate_mode and im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.array(im)
        return arr.astype(np.uint8)
    else:
        raise ValueError("Unsupported modality for image")

def serialize_content(modality: str, content_path: str, text_encoding: str="utf-8", validate_image_mode: bool=True) -> tuple[AppHeader, bytes]:
    if modality == "text":
        data = load_text_as_bytes(content_path, encoding=text_encoding)
        hdr = AppHeader(version=1, modality="text", height=0, width=0, channels=0,
                        bits_per_sample=8, payload_len_bytes=len(data))
        return hdr, data

    arr = load_image_to_array(content_path, modality=modality, validate_mode=validate_image_mode)
    if modality in ("edge","depth"):
        h, w = arr.shape
        ch = 1
        payload = arr.reshape(-1).tobytes()
    elif modality == "segmentation":
        h, w, ch = arr.shape
        payload = arr.reshape(-1).tobytes()
    else:
        raise ValueError("Unknown modality")

    hdr = AppHeader(version=1, modality=modality, height=h, width=w, channels=ch,
                    bits_per_sample=8, payload_len_bytes=len(payload))
    return hdr, payload

def _reshape_bytes_safe(payload_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Robust reshape:
    - truncate if too long
    - zero-pad if too short
    Always returns a uint8 array with the requested shape.
    """
    n_expected = 1
    for s in shape: n_expected *= s
    b = np.frombuffer(payload_bytes, dtype=np.uint8, count=min(len(payload_bytes), n_expected))
    if b.size < n_expected:
        b = np.pad(b, (0, n_expected - b.size), mode="constant", constant_values=0)
    else:
        b = b[:n_expected]
    return b.reshape(shape)

def deserialize_content(hdr: AppHeader, payload_bytes: bytes, text_encoding: str="utf-8", text_errors: str="replace") -> tuple[str, np.ndarray]:
    """
    Returns (text_str, image_array). Only one is relevant per modality.
    Uses robust reshape for images to guarantee output even when payload length mismatches.
    """
    if hdr.modality == "text":
        s = text_bytes_to_string(payload_bytes, encoding=text_encoding, errors=text_errors)
        return s, np.array([], dtype=np.uint8)

    if hdr.modality in ("edge","depth"):
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width))
        if hdr.modality == "edge":
            arr = (arr >= 128).astype(np.uint8)*255
        return "", arr

    if hdr.modality == "segmentation":
        arr = _reshape_bytes_safe(payload_bytes, (hdr.height, hdr.width, hdr.channels))
        return "", arr

    raise ValueError("Unknown modality in header")

def save_output(hdr: AppHeader, text_str: str, img_arr: np.ndarray, out_path: str):
    if hdr.modality == "text":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_str)
        return

    from PIL import Image
    if hdr.modality in ("edge","depth"):
        im = Image.fromarray(img_arr.astype(np.uint8), mode="L")
        im.save(out_path, format="PNG")
    elif hdr.modality == "segmentation":
        im = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
        im.save(out_path, format="PNG")
    else:
        raise ValueError("Unsupported modality")
