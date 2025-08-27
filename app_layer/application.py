import io, json
from typing import Tuple
from PIL import Image

# 2B magic | 2B hlen | header-json | body
def _make_app_header_bytes(magic2: bytes, header_dict: dict) -> bytes:
    hdr = json.dumps(header_dict, ensure_ascii=False).encode("utf-8")
    assert len(magic2) == 2
    hlen = len(hdr).to_bytes(2, 'big')
    return magic2 + hlen + hdr

def _parse_app_payload(payload: bytes) -> Tuple[str, dict, bytes]:
    if len(payload) < 4:
        return "", {}, b""
    magic = payload[0:2].decode(errors="ignore")
    hlen = int.from_bytes(payload[2:4], 'big')
    if len(payload) < 4 + hlen:
        return magic, {}, b""
    header = json.loads(payload[4:4+hlen].decode("utf-8"))
    body = payload[4+hlen:]
    return magic, header, body

# ---------- 非UEP（PNG圧縮ペイロード） ----------
def prepare_image_payload_png(path: str, resize: int = 128) -> bytes:
    img = Image.open(path).convert("L")
    if resize and resize > 0:
        img = img.resize((resize, resize))
    bio = io.BytesIO(); img.save(bio, format="PNG")
    img_bytes = bio.getvalue()
    header = {"fmt": "PNG", "w": img.width, "h": img.height}
    return _make_app_header_bytes(b"IM", header) + img_bytes

def parse_image_payload_png(payload: bytes, out_path: str) -> None:
    magic, header, body = _parse_app_payload(payload)
    if magic != "IM" or header.get("fmt") != "PNG":
        raise ValueError("Not a PNG payload")
    with open(out_path, "wb") as f:
        f.write(body)

# ---------- UEP：タイプ別RAW生成（resize<=0 なら元サイズ保持） ----------
def prepare_image_header_and_body_raw(path: str, resize: int, kind: str) -> Tuple[bytes, bytes]:
    kind = kind.lower()
    if kind == "seg":
        img = Image.open(path).convert("RGB")
    else:
        img = Image.open(path).convert("L")
    if resize and resize > 0:
        img = img.resize((resize, resize))

    if kind == "depth":
        raw = img.tobytes()
        header = {"fmt": "RAW8_GRAY", "w": img.width, "h": img.height, "kind": "depth", "channels": 1}
        return _make_app_header_bytes(b"IM", header), raw

    if kind == "edge":
        img = img.point(lambda p: 255 if p >= 128 else 0, mode="L")
        raw = img.tobytes()
        header = {"fmt": "RAW8_EDGE", "w": img.width, "h": img.height, "kind": "edge", "channels": 1}
        return _make_app_header_bytes(b"IM", header), raw

    if kind == "seg":
        raw = img.tobytes()  # RGB
        header = {"fmt": "RAW8_RGB", "w": img.width, "h": img.height, "kind": "seg", "channels": 3}
        return _make_app_header_bytes(b"IM", header), raw

    raise ValueError("Unknown image kind: " + kind)

# ---------- RAW→PIL（PNG保存や後処理に使う） ----------
def pil_from_raw(fmt: str, w: int, h: int, raw: bytes) -> Image.Image:
    need = w * h
    if fmt == "RAW8_GRAY":
        buf = raw[:need].ljust(need, b"\x00")
        return Image.frombytes("L", (w, h), buf)
    if fmt == "RAW8_EDGE":
        buf = raw[:need].ljust(need, b"\x00")
        im = Image.frombytes("L", (w, h), buf)
        return im.point(lambda p: 255 if p >= 128 else 0).convert("1")
    if fmt == "RAW8_RGB":
        need_rgb = w * h * 3
        buf = raw[:need_rgb].ljust(need_rgb, b"\x00")
        return Image.frombytes("RGB", (w, h), buf)
    buf = raw[:need].ljust(need, b"\x00")
    return Image.frombytes("L", (w, h), buf)

# ---------- テキスト ----------
def prepare_text_header_and_body(text: str) -> Tuple[bytes, bytes]:
    by = text.encode("utf-8")
    header = {"encoding": "utf-8", "nbytes": len(by), "fmt": "TXT"}
    header_bytes = _make_app_header_bytes(b"TX", header)
    return header_bytes, by

def parse_text_payload(payload: bytes) -> str:
    magic, header, body = _parse_app_payload(payload)
    if magic != "TX":
        raise ValueError("Not a text payload")
    enc = header.get("encoding", "utf-8")
    return body.decode(enc, errors="strict")
