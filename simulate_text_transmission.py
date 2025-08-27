import os, argparse, json
import numpy as np
from typing import Optional, Dict, Any

from common.config import SimConfig, PHYConfig, ChannelConfig, LinkConfig, AppConfig
from transmitter.send import build_frames, assemble_bitstream, transmit_bits
from receiver.receive import receive_bits, deframe_and_check
from channel.channel_model import ChannelModel
from app_layer.application import prepare_text_header_and_body, parse_text_payload
from data_link_layer.encoding import bytes_to_bits

def _tag(cfg: SimConfig, crc_ok: bool) -> str:
    ch = cfg.ch.channel_type.lower()
    mod = cfg.phy.modulation.upper()
    cod = cfg.link.coding.lower()
    intl = "intl1" if cfg.link.interleaver else "intl0"
    snr = f"{int(round(cfg.phy.snr_db))}dB"
    crc = "crc1" if crc_ok else "crc0"
    return f"{ch}_{mod}_{cod}_{intl}_snr{snr}_{crc}"

def _load_json_if_exists(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _default_config_path() -> str:
    return os.path.join("configs", "sim_text.json")

def _make_cfg(d: Dict[str,Any]) -> SimConfig:
    phy = PHYConfig(modulation=d.get("mod","QPSK"), snr_db=float(d.get("snr_db",8.0)))
    ch = ChannelConfig(channel_type=d.get("channel","awgn"))
    link = LinkConfig(coding=d.get("coding","hamming74"), interleaver=bool(d.get("interleaver",True)))
    app = AppConfig(app_type="text", text=d.get("text","Hello vehicular world! こんにちは🚗"))
    return SimConfig(phy=phy, ch=ch, link=link, app=app)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--text_file", type=str, default=None)
    p.add_argument("--snr_db", type=float, default=None)
    p.add_argument("--mod", type=str, default=None)
    p.add_argument("--channel", type=str, default=None, choices=["awgn","rayleigh","rician"])
    p.add_argument("--coding", type=str, default=None, choices=["none","repetition3","hamming74"])
    p.add_argument("--interleaver", type=lambda s: s.lower()!="false", default=None)
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    d = _load_json_if_exists(args.config or _default_config_path())
    cfg = _make_cfg(d)
    if args.text is not None: cfg.app.text = args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            cfg.app.text = f.read()
    if args.snr_db is not None: cfg.phy.snr_db = float(args.snr_db)
    if args.mod: cfg.phy.modulation = args.mod
    if args.channel: cfg.ch.channel_type = args.channel
    if args.coding: cfg.link.coding = args.coding
    if args.interleaver is not None: cfg.link.interleaver = bool(args.interleaver)
    outdir = args.outdir or d.get("outdir","outputs"); os.makedirs(outdir, exist_ok=True)

    # 1フレームで送る（テキストは小さい想定）
    hbytes, bbytes = prepare_text_header_and_body(cfg.app.text)
    payload = hbytes + bbytes
    frames = build_frames(payload, cfg.link.max_frame_bits)
    tx_bits = assemble_bitstream(frames)

    from channel.channel_model import ChannelModel
    ch = ChannelModel(cfg.ch)
    sy, meta = transmit_bits(cfg, tx_bits)
    y, h = ch.transmit(sy, meta["esn0_db"])
    bits_dec, _ = receive_bits(cfg, y, h, meta)
    rec, oks = deframe_and_check(bits_dec[:meta["n_tx_bits_uncoded"]])
    ok = oks[0]
    text_rx = ""
    if ok and len(rec[0])>0:
        text_rx = parse_text_payload(rec[0])

    tag = _tag(cfg, ok)
    out_path = os.path.join(outdir, f"text_{tag}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text_rx if ok else "")
    print(json.dumps({
        "crc_ok": ok,
        "out": out_path,
        "mod": cfg.phy.modulation,
        "coding": cfg.link.coding,
        "channel": cfg.ch.channel_type,
        "snr_db": cfg.phy.snr_db
    }, ensure_ascii=False, indent=2))
    print("Text saved:", out_path)

if __name__ == "__main__":
    main()
