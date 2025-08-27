import os, argparse, json
import numpy as np
from typing import Optional, Dict, Any
from PIL import Image

from common.config import SimConfig, PHYConfig, ChannelConfig, LinkConfig, AppConfig, UEPConfig
from transmitter.send import (
    build_frames, assemble_bitstream, transmit_bits, transmit_bits_custom,
    fragment_payload, build_frames_from_chunks
)
from receiver.receive import receive_bits_generic, deframe_and_check
from channel.channel_model import ChannelModel
from app_layer.application import (
    prepare_image_payload_png,
    prepare_image_header_and_body_raw,
    parse_image_payload_png,
    pil_from_raw
)
from app_layer.postprocess import postprocess_image
from data_link_layer.encoding import bytes_to_bits, bits_to_bytes

def _fmt_snr(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}dB"
    s = f"{x:.1f}".replace(".", "p")
    return f"{s}dB"

def _tag_from_cfg(cfg: SimConfig, crc_ok: bool) -> str:
    ch = cfg.ch.channel_type.lower()
    mod = cfg.phy.modulation.upper()
    cod = cfg.link.coding.lower()
    intl = "intl1" if cfg.link.interleaver else "intl0"
    uep = "uep1" if cfg.uep.enabled else "uep0"
    snr = _fmt_snr(cfg.phy.snr_db)
    crc = "crc1" if crc_ok else "crc0"
    ts = ""
    if getattr(cfg.ch, "time_selective", False):
        if getattr(cfg.ch, "fading_model", "block") == "block":
            ts = f"_ts{int(getattr(cfg.ch, 'coherence_symbols', 0) or 0)}"
        else:
            rho = str(getattr(cfg.ch, "rho", 0.95)).replace(".", "p")
            ts = f"_tsGM{rho}"
    kind = getattr(cfg.app, "image_kind", "depth")
    return f"{ch}_{mod}_{cod}_{intl}_{uep}{ts}_{kind}_snr{snr}_{crc}"

def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def _load_json_if_exists(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _default_config_path() -> str:
    return os.path.join("configs", "sim_image.json")

def _get_cfg_from_dict(d: Dict[str, Any]) -> SimConfig:
    phy = PHYConfig(
        modulation=d.get("mod", d.get("modulation", "QPSK")),
        snr_db=float(d.get("snr_db", 8.0)),
        snr_type=d.get("snr_type", "EsN0"),
        code_rate_override=d.get("code_rate_override", None),
    )
    ch = ChannelConfig(
        channel_type=d.get("channel", d.get("channel_type", "awgn")),
        rician_K=float(d.get("rician_K", 3.0)),
        time_selective=bool(d.get("time_selective", False)),
        fading_model=d.get("fading_model", "block"),
        coherence_symbols=int(d.get("coherence_symbols", 32)),
        rho=float(d.get("rho", 0.95)),
    )
    link = LinkConfig(
        coding=d.get("coding", "hamming74"),
        interleaver=bool(d.get("interleaver", True)),
        interleaver_seed=int(d.get("interleaver_seed", 2025)),
        max_frame_bits=int(d.get("max_frame_bits", 5_000_000)),
    )
    uep_dict = d.get("uep", {})
    uep = UEPConfig(
        enabled=bool(uep_dict.get("enabled", False)),
        header_mod=uep_dict.get("header_mod", "BPSK"),
        header_coding=uep_dict.get("header_coding", "repetition3"),
        header_interleaver=bool(uep_dict.get("header_interleaver", True)),
        header_boost_db=float(uep_dict.get("header_boost_db", 6.0)),
        header_max_frame_bits=int(uep_dict.get("header_max_frame_bits", 65_536)),
        header_repeats=int(uep_dict.get("header_repeats", 1)),
    )
    app = AppConfig(
        app_type="image",
        image_path=d.get("image", "examples/sample_image.png"),
        image_resize=int(d.get("resize", 128)),
        image_save_size=int(d.get("save_size", 0)),
        image_kind=d.get("image_kind", "depth"),
    )
    return SimConfig(phy=phy, ch=ch, link=link, app=app, uep=uep)

def _merge_cli_over_cfg(cfg: SimConfig, args) -> SimConfig:
    if args.mod:         cfg.phy.modulation = args.mod
    if args.snr_db is not None: cfg.phy.snr_db = float(args.snr_db)
    if args.channel:     cfg.ch.channel_type = args.channel
    if args.coding:      cfg.link.coding = args.coding
    if args.interleaver is not None: cfg.link.interleaver = bool(args.interleaver)
    if args.image:       cfg.app.image_path = args.image
    if args.resize is not None:
        if isinstance(args.resize, str) and args.resize.lower() == "orig":
            cfg.app.image_resize = 0
        else:
            cfg.app.image_resize = int(args.resize)
    if args.save_size is not None: cfg.app.image_save_size = int(args.save_size)
    if args.uep is not None: cfg.uep.enabled = bool(args.uep)
    if args.kind:        cfg.app.image_kind = args.kind
    if args.header_mod: cfg.uep.header_mod = args.header_mod
    if args.header_coding: cfg.uep.header_coding = args.header_coding
    if args.header_interleaver is not None: cfg.uep.header_interleaver = bool(args.header_interleaver)
    if args.header_boost_db is not None: cfg.uep.header_boost_db = float(args.header_boost_db)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--mod", type=str, default=None)
    parser.add_argument("--channel", type=str, default=None, choices=["awgn","rayleigh","rician"])
    parser.add_argument("--coding", type=str, default=None, choices=["none","repetition3","hamming74"])
    parser.add_argument("--interleaver", type=lambda s: s.lower()!="false", default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--resize", type=str, default=None)   # "orig" で元サイズ
    parser.add_argument("--save_size", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--time_selective", action="store_true")
    parser.add_argument("--fading_model", choices=["block","gauss_markov"], default=None)
    parser.add_argument("--coherence_symbols", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--rician_K", type=float, default=None)
    # UEP
    parser.add_argument("--uep", action="store_true")
    parser.add_argument("--no-uep", dest="uep", action="store_false")
    parser.set_defaults(uep=None)
    parser.add_argument("--header_mod", type=str, choices=["BPSK","QPSK","16QAM"], default=None)
    parser.add_argument("--header_coding", type=str, choices=["none","repetition3","hamming74"], default=None)
    parser.add_argument("--header_interleaver", type=lambda s: s.lower()!="false", default=None)
    parser.add_argument("--header_boost_db", type=float, default=None)
    parser.add_argument("--kind", type=str, choices=["depth","edge","seg"], default=None)
    parser.add_argument("--pp", action="store_true")
    parser.add_argument("--pp_strength", type=str, default="auto", choices=["auto","light","medium","strong"])
    parser.add_argument("--pp_window", type=int, default=3)

    args = parser.parse_args()

    cfg_dict = _load_json_if_exists(args.config or _default_config_path())
    cfg = _get_cfg_from_dict(cfg_dict)
    if args.time_selective: cfg.ch.time_selective = True
    if args.fading_model:   cfg.ch.fading_model = args.fading_model
    if args.coherence_symbols is not None: cfg.ch.coherence_symbols = int(args.coherence_symbols)
    if args.rho is not None: cfg.ch.rho = float(args.rho)
    if args.rician_K is not None: cfg.ch.rician_K = float(args.rician_K)

    cfg = _merge_cli_over_cfg(cfg, args)
    outdir = args.outdir or cfg_dict.get("outdir", "outputs")
    os.makedirs(outdir, exist_ok=True)

    ch = ChannelModel(cfg.ch)

    # ===== UEP: ヘッダ強保護 + ボディ多フレーム =====
    if cfg.uep.enabled:
        header_bytes, body_bytes = prepare_image_header_and_body_raw(
            cfg.app.image_path, cfg.app.image_resize, cfg.app.image_kind
        )

        # --- header（強保護＋ブースト）
        from data_link_layer.encoding import bytes_to_bits
        header_frame = build_frames(header_bytes, cfg.uep.header_max_frame_bits)[0]
        header_bits_unc = bytes_to_bits(header_frame)
        sy_h, meta_h = transmit_bits_custom(
            snr_type=cfg.phy.snr_type, snr_db=cfg.phy.snr_db,
            modulation=cfg.uep.header_mod, coding=cfg.uep.header_coding,
            interleaver=cfg.uep.header_interleaver, interleaver_seed=2025,
            tx_bits_uncoded=header_bits_unc, code_rate_override=cfg.phy.code_rate_override,
            snr_boost_db=cfg.uep.header_boost_db,
        )
        y_h, h_h = ch.transmit(sy_h, meta_h["esn0_db"])
        bits_h_dec, _ = receive_bits_generic(y_h, h_h, meta_h)
        recH, oksH = deframe_and_check(bits_h_dec)
        header_ok = oksH[0]

        # --- header 復元 or フォールバック（W,H を正しく保持）
        if header_ok and len(recH[0]) >= 6:
            try:
                hp = recH[0]
                hlen  = int.from_bytes(hp[2:4], 'big')
                hdrj  = json.loads(hp[4:4+hlen].decode("utf-8"))
                w = int(hdrj.get("w")); h = int(hdrj.get("h"))
                fmt = hdrj.get("fmt")
            except Exception:
                src_w, src_h = Image.open(cfg.app.image_path).size
                if cfg.app.image_resize and cfg.app.image_resize > 0:
                    w = h = int(cfg.app.image_resize)
                else:
                    w, h = src_w, src_h
                fmt = {"depth":"RAW8_GRAY","edge":"RAW8_EDGE","seg":"RAW8_RGB"}[cfg.app.image_kind]
        else:
            src_w, src_h = Image.open(cfg.app.image_path).size
            if cfg.app.image_resize and cfg.app.image_resize > 0:
                w = h = int(cfg.app.image_resize)
            else:
                w, h = src_w, src_h
            fmt = {"depth":"RAW8_GRAY","edge":"RAW8_EDGE","seg":"RAW8_RGB"}[cfg.app.image_kind]

        # --- body をフラグメント送信
        chunks = fragment_payload(body_bytes, cfg.link.max_frame_bits)
        frames = build_frames_from_chunks(chunks)

        from data_link_layer.encoding import bits_to_bytes
        recv_body = bytearray()
        all_crc_ok = True
        for chunk, frame in zip(chunks, frames):
            bits_unc = bytes_to_bits(frame)
            sy_b, meta_b = transmit_bits_custom(
                snr_type=cfg.phy.snr_type, snr_db=cfg.phy.snr_db,
                modulation=cfg.phy.modulation, coding=cfg.link.coding,
                interleaver=cfg.link.interleaver, interleaver_seed=cfg.link.interleaver_seed,
                tx_bits_uncoded=bits_unc, code_rate_override=cfg.phy.code_rate_override,
                snr_boost_db=0.0,
            )
            y_b, h_b = ch.transmit(sy_b, meta_b["esn0_db"])
            bits_b_dec, _ = receive_bits_generic(y_b, h_b, meta_b)
            recB, oksB = deframe_and_check(bits_b_dec)
            ok = oksB[0]
            all_crc_ok = all_crc_ok and ok
            if ok:
                recv_body.extend(recB[0])
            else:
                raw = bits_to_bytes(bits_b_dec)
                need = len(chunk)
                if len(raw) >= need + 8:
                    recv_body.extend(raw[4:4+need])
                else:
                    recv_body.extend(raw[:need].ljust(need, b"\x00"))

        # RAW -> 画像
        img = pil_from_raw(fmt, w, h, bytes(recv_body))

        # アプリ層誤り隠蔽（任意）
        if args.pp:
            img = postprocess_image(img, cfg.app.image_kind,
                                    strength=args.pp_strength, window=args.pp_window)

        # 保存サイズ（任意）
        save_size = int(cfg.app.image_save_size or 0)
        if args.save_size is not None:
            save_size = int(args.save_size)
        if save_size and save_size > 0 and (img.width != save_size or img.height != save_size):
            resample = Image.NEAREST if cfg.app.image_kind in ("edge","seg") else Image.BILINEAR
            img = img.resize((save_size, save_size), resample=resample)

        tag = _tag_from_cfg(cfg, crc_ok=(header_ok and all_crc_ok))
        out_png = _unique_path(os.path.join(outdir, f"image_{tag}.png"))
        img.save(out_png, format="PNG")

        print(json.dumps({
            "uep": True,
            "header_ok": header_ok,
            "body_all_crc_ok": all_crc_ok,
            "n_body_frames": len(frames),
            "saved_png": out_png,
            "meta": {
                "channel": cfg.ch.channel_type,
                "snr_db": cfg.phy.snr_db,
                "kind": cfg.app.image_kind,
                "header": {"mod": cfg.uep.header_mod, "coding": cfg.uep.header_coding, "boost_db": cfg.uep.header_boost_db},
                "body":   {"mod": cfg.phy.modulation,   "coding": cfg.link.coding}
            }
        }, ensure_ascii=False, indent=2))
        print("PNG saved:", out_png)
        return

    # ===== 非UEP：PNGペイロード1フレーム =====
    payload = prepare_image_payload_png(cfg.app.image_path, cfg.app.image_resize)
    frames = build_frames(payload, cfg.link.max_frame_bits)
    tx_bits_uncoded = assemble_bitstream(frames)
    sy, meta = transmit_bits(cfg, tx_bits_uncoded)
    y, h = ch.transmit(sy, meta["esn0_db"])
    from receiver.receive import receive_bits as _recv_old
    bits_dec, rmeta = _recv_old(cfg, y, h, meta)
    rec, oks = deframe_and_check(bits_dec[:meta["n_tx_bits_uncoded"]])
    ok = oks[0]
    tag = _tag_from_cfg(cfg, ok)
    out_png = _unique_path(os.path.join(outdir, f"image_{tag}.png"))
    if ok and len(rec[0]) > 0:
        parse_image_payload_png(rec[0], out_png)
        # 保存サイズ変更（任意）
        if cfg.app.image_save_size and cfg.app.image_save_size > 0:
            im = Image.open(out_png)
            resample = Image.NEAREST if cfg.app.image_kind in ("edge", "seg") else Image.BILINEAR
            im = im.resize((cfg.app.image_save_size, cfg.app.image_save_size), resample=resample)
            im.save(out_png, format="PNG")
        print("Recovered image saved to:", out_png)
    else:
        print("CRC failed (non-UEP). Try higher SNR or enable --uep.")

if __name__ == "__main__":
    main()
