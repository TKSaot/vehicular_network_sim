# common/run_utils.py
from __future__ import annotations
import os, json, re, datetime
from dataclasses import asdict

def _sanitize(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z_.-]+', '-', str(s)).strip('-')

def make_output_dir(cfg, modality: str, input_path: str, output_root: str = "outputs") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ch = cfg.chan.channel_type
    doppler_tag = f"_fd{int(round(cfg.chan.doppler_hz))}" if ch == "rayleigh" else ""
    snr_tag = f"snr{int(round(cfg.chan.snr_db))}"
    fec = cfg.link.fec_scheme
    if fec == "repeat":
        fec = f"repeat{cfg.link.repeat_k}"
    hdr_tag = f"_hdr{cfg.link.header_copies}xR{cfg.link.header_rep_k}"
    # NEW
    map_tag = {"none":"mapNone", "permute":"mapPerm", "frame_block":"mapFB"}.get(cfg.link.byte_mapping_scheme, "map?")
    if cfg.link.byte_mapping_scheme != "none":
        seed_tag = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
        map_tag += f"{seed_tag}"

    folder = (
        f"{ts}__{modality}__{ch}{doppler_tag}_{snr_tag}"
        f"__{cfg.mod.scheme}__{fec}{hdr_tag}_{map_tag}"
        f"_ilv{cfg.link.interleaver_depth}"
        f"_mtu{cfg.link.mtu_bytes}"
        f"_seed{cfg.chan.seed}"
    )
    base = os.path.join(output_root, _sanitize(folder))
    os.makedirs(base, exist_ok=True)

    meta = asdict(cfg)
    meta.update({
        "modality": modality,
        "input_file": input_path,
        "output_dir": base,
    })
    with open(os.path.join(base, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return base

def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
