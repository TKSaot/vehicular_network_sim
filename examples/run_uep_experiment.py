# examples/run_uep_experiment.py
"""
Run EEP vs UEP experiments with equal airtime (equal total symbols) across modalities.

- Scenarios:
    * eep        : Equal Error Protection (uniform MCS for all modalities)
    * uep_edge   : Edge-prioritized protection
    * uep_depth  : Depth-prioritized protection
    * all        : Run all three above

- SNR sweep: default "1,4,8,12"
- Channel:  rayleigh (default) or awgn
- Outputs:
    outputs/experiments/<scenario>/snr_<XdB>/<modality>/(received.png|received.txt, rx_stats.json)
    outputs/experiments/<scenario>/summary_<scenario>.csv (append rows per SNR)

Notes:
- Keeps the #2 baseline intact. Only orchestrates per-modality configs from outside.
- If a requested FEC scheme is not supported in your local tree, it falls back to "hamming74".
- Equal airtime is matched to EEP's total symbol count within a default tolerance of 1%.
"""

from __future__ import annotations
import os, sys, csv, json, argparse
import copy
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project utilities (already present in your tree)
from common.config import SimulationConfig
from common.utils import set_seed, psnr
from common.run_utils import write_json
from common.byte_mapping import unmap_bytes

from app_layer.application import (
    serialize_content, AppHeader, deserialize_content, save_output
)
from transmitter.send import build_transmission
from channel.channel_model import awgn_channel, rayleigh_fading
from receiver.receive import recover_from_symbols

# Default inputs (we add 'text' path here via image_config)
from configs import image_config as CFG

# ---------------------- Helpers ----------------------

MODALITIES = ["edge", "depth", "segmentation", "text"]
BPS = {"bpsk": 1, "qpsk": 2, "16qam": 4}

class MCS:
    def __init__(self, mod: str = "qpsk", fec: str = "hamming74", ilv: int = 16):
        self.mod = mod
        self.fec = fec
        self.ilv = ilv
    def copy(self): return MCS(self.mod, self.fec, self.ilv)
    def as_dict(self): return {"mod": self.mod, "fec": self.fec, "interleaver_depth": self.ilv}

def _ensure_input_path(modality: str, path: str | None) -> str:
    if path is not None and os.path.isfile(path):
        return path
    if modality in CFG.INPUTS and os.path.isfile(CFG.INPUTS[modality]):
        return CFG.INPUTS[modality]
    # final fallbacks for robustness
    fallback = {
        "text": "examples/sample.txt",
        "edge": "examples/edge_00001_.png",
        "depth": "examples/depth_00001_.png",
        "segmentation": "examples/segmentation_00001_.png",
    }[modality]
    if os.path.isfile(fallback):
        return fallback
    # allow /mnt/data paths if user placed samples there
    mnt = {
        "text": "/mnt/data/sample.txt",
        "edge": "/mnt/data/edge_aachen_72_00001_.png",
        "depth": "/mnt/data/depth_aachen_72_00001_.png",
        "segmentation": "/mnt/data/segmentation_aachen_72_00001_.png",
    }[modality]
    if os.path.isfile(mnt):
        return mnt
    raise FileNotFoundError(f"No input found for modality={modality}")

def _apply_mcs_to_cfg(cfg: SimulationConfig, mcs: MCS, drop_bad_frames: bool) -> None:
    cfg.mod.scheme = mcs.mod
    cfg.link.fec_scheme = mcs.fec
    cfg.link.interleaver_depth = int(mcs.ilv)
    cfg.link.drop_bad_frames = bool(drop_bad_frames)
    # keep mapping & header protection as in your tree
    if cfg.link.byte_mapping_seed is None:
        cfg.link.byte_mapping_seed = cfg.chan.seed

def _estimate_symbols_for_modality(modality: str, input_path: str,
                                   cfg: SimulationConfig) -> tuple[int, AppHeader, bytes]:
    """
    Build a transmission once and return exact symbol length, header bytes.
    """
    # Serialize
    hdr, payload = serialize_content(modality, input_path, app_cfg=cfg.app,
                                     text_encoding="utf-8", validate_image_mode=True)
    hdr_bytes = hdr.to_bytes()
    # Try building; if FEC unsupported, fallback to hamming74
    try:
        tx_syms, _ = build_transmission(hdr_bytes, payload, cfg)
    except Exception as e:
        # FEC unknown or not implemented in local tree
        if "Unknown FEC scheme" in str(e) or "rs255_223" in str(e) or "conv_k7" in str(e):
            old = cfg.link.fec_scheme
            cfg.link.fec_scheme = "hamming74"
            tx_syms, _ = build_transmission(hdr_bytes, payload, cfg)
            print(f"[WARN] FEC '{old}' unsupported here. Fell back to 'hamming74' for {modality}.")
        else:
            raise
    return int(len(tx_syms)), hdr, payload

def _run_single_transmission(modality: str, input_path: str, cfg: SimulationConfig,
                             channel: str, snr_db: float, out_dir: str,
                             prebuilt_hdr: AppHeader | None = None,
                             prebuilt_payload: bytes | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Prepare header & payload
    if prebuilt_hdr is None or prebuilt_payload is None:
        hdr, payload = serialize_content(modality, input_path, app_cfg=cfg.app,
                                         text_encoding="utf-8", validate_image_mode=True)
    else:
        hdr, payload = prebuilt_hdr, prebuilt_payload
    hdr_bytes = hdr.to_bytes()

    # TX
    tx_syms, tx_meta = build_transmission(hdr_bytes, payload, cfg)

    # Channel
    if channel == "awgn":
        rx_syms = awgn_channel(tx_syms, snr_db, seed=cfg.chan.seed)
    else:
        rx_syms, _ = rayleigh_fading(
            tx_syms, snr_db, seed=cfg.chan.seed,
            doppler_hz=cfg.chan.doppler_hz, symbol_rate=cfg.chan.symbol_rate,
            block_fading=cfg.chan.block_fading,  # snr_reference="rx" by default in your channel model
        )

    # RX
    rx_app_hdr_b, rx_payload_b, stats = recover_from_symbols(rx_syms, tx_meta, cfg)

    # Try parse header; if failed and force_output_on_hdr_fail, fall back to TX hdr
    try:
        rx_hdr = AppHeader.from_bytes(rx_app_hdr_b)
    except Exception:
        rx_hdr = hdr

    # Byte unmapping
    mapping_seed = cfg.link.byte_mapping_seed if cfg.link.byte_mapping_seed is not None else cfg.chan.seed
    rx_payload_b = unmap_bytes(
        rx_payload_b,
        mtu_bytes=cfg.link.mtu_bytes,
        scheme=cfg.link.byte_mapping_scheme,
        seed=mapping_seed,
        original_len=rx_hdr.payload_len_bytes,
    )

    # Deserialize & Save
    text_str, img_arr = deserialize_content(rx_hdr, rx_payload_b, app_cfg=cfg.app, text_encoding="utf-8")
    if modality == "text":
        out_path = os.path.join(out_dir, "received.txt")
        save_output(rx_hdr, text_str, img_arr, out_path)
        psnr_db = None
    else:
        out_path = os.path.join(out_dir, "received.png")
        save_output(rx_hdr, text_str, img_arr, out_path)
        # PSNR quick check (for your internal reference; ComfyUI will handle FID/LPIPS)
        try:
            im_orig = Image.open(input_path)
            if modality in ("edge", "depth"): im_orig = im_orig.convert("L")
            elif modality == "segmentation": im_orig = im_orig.convert("RGB")
            arr_orig = np.array(im_orig)
            if arr_orig.shape != img_arr.shape:
                arr_orig = arr_orig[:img_arr.shape[0], :img_arr.shape[1], ...]
            psnr_db = float(psnr(arr_orig.astype(np.uint8), img_arr.astype(np.uint8), data_range=255))
        except Exception:
            psnr_db = None

    # Write stats json with extra fields
    rec = {
        "modality": modality,
        "fec": cfg.link.fec_scheme,
        "modulation": cfg.mod.scheme,
        "interleaver_depth": cfg.link.interleaver_depth,
        "snr_db": snr_db,
        "channel": channel,
        "output_path": out_path,
        "psnr_db": psnr_db,
    }
    # merge rx stats
    for k, v in stats.items():
        if isinstance(v, (np.bool_, bool)):
            rec[k] = bool(v)
        elif isinstance(v, (np.integer, int)):
            rec[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            rec[k] = float(v)
        else:
            try:
                rec[k] = int(v)
            except Exception:
                pass

    write_json(os.path.join(out_dir, "rx_stats.json"), rec)
    return rec

# ---------------------- Scenario & Scheduler ----------------------

def _baseline_eep_mcs() -> dict:
    # EEP を “少しだけ”堅牢化：QPSK + Conv(K=7) R=2/3 + interleaver 32
    return {m: MCS(mod="qpsk", fec="conv_k7_r23", ilv=32) for m in MODALITIES}

def _uep_edge_initial() -> dict:
    return {
        "edge": MCS(mod="bpsk", fec="conv_k7_r12", ilv=32),
        "depth": MCS(mod="qpsk", fec="conv_k7_r23", ilv=24),
        "segmentation": MCS(mod="qpsk", fec="conv_k7_r34", ilv=16),
        "text": MCS(mod="qpsk", fec="conv_k7_r12", ilv=32),
    }

def _uep_depth_initial() -> dict:
    return {
        "depth": MCS(mod="bpsk", fec="conv_k7_r12", ilv=32),
        "edge": MCS(mod="qpsk", fec="conv_k7_r23", ilv=24),
        "segmentation": MCS(mod="qpsk", fec="conv_k7_r34", ilv=16),
        "text": MCS(mod="qpsk", fec="conv_k7_r12", ilv=32),
    }

def _next_lighter(mcs: MCS) -> MCS | None:
    # lighten = consume fewer symbols (higher throughput): try FEC 2/3->3/4, 1/2->2/3->3/4; then modulation BPSK->QPSK->16QAM
    order_fec = ["conv_k7_r12", "conv_k7_r23", "conv_k7_r34", "hamming74"]  # treat hamming ~ light-ish
    order_mod = ["bpsk", "qpsk", "16qam"]
    m = mcs.copy()
    # fec step up
    if m.fec in order_fec:
        idx = order_fec.index(m.fec)
        if idx + 1 < len(order_fec):
            m.fec = order_fec[idx + 1]; return m
    # modulation step up
    if m.mod in order_mod:
        im = order_mod.index(m.mod)
        if im + 1 < len(order_mod):
            m.mod = order_mod[im + 1]; return m
    return None

def _next_stronger(mcs: MCS) -> MCS | None:
    # stronger = more redundancy (more symbols): 3/4->2/3->1/2; 16QAM->QPSK->BPSK
    order_fec = ["hamming74", "conv_k7_r34", "conv_k7_r23", "conv_k7_r12"]
    order_mod = ["16qam", "qpsk", "bpsk"]
    m = mcs.copy()
    if m.fec in order_fec:
        idx = order_fec.index(m.fec)
        if idx + 1 < len(order_fec):
            m.fec = order_fec[idx + 1]; return m
    if m.mod in order_mod:
        im = order_mod.index(m.mod)
        if im + 1 < len(order_mod):
            m.mod = order_mod[im + 1]; return m
    return None

def _build_cfg(modality: str, mcs: MCS, chan_type: str, snr_db: float, seed: int,
               drop_bad_frames: bool) -> SimulationConfig:
    cfg: SimulationConfig = CFG.build_config(modality=modality)
    cfg.chan.channel_type = chan_type
    cfg.chan.snr_db = float(snr_db)
    cfg.chan.seed = int(seed)
    _apply_mcs_to_cfg(cfg, mcs, drop_bad_frames=drop_bad_frames)
    return cfg

def _equal_airtime_adjust(target_total_syms: int,
                          init_mcs: dict,
                          inputs: dict,
                          chan_type: str,
                          snr_db: float,
                          seed: int,
                          drop_bad_frames: bool,
                          priority: str) -> tuple[dict, dict]:
    """
    Adjust non-priority modalities' MCS so that sum(symbols) ~= target_total_syms (±1%).
    Returns (final_mcs_dict, prebuilt_cache) where prebuilt_cache carries (hdr, payload, sym_len) for reuse.
    """
    # Prebuild headers/payloads once per modality (independent of MCS)
    prebuilt = {}
    # We'll estimate symbols per modality for current MCS by actually building transmissions.
    def estimate_for_mod(m, mcs_obj):
        cfg = _build_cfg(m, mcs_obj, chan_type, snr_db, seed, drop_bad_frames)
        if m in prebuilt and prebuilt[m].get("payload") is not None and prebuilt[m].get("hdr") is not None:
            # reuse payload if available
            hdr = prebuilt[m]["hdr"]; payload = prebuilt[m]["payload"]
            hdr_b = hdr.to_bytes()
            try:
                tx_syms, _ = build_transmission(hdr_b, payload, cfg)
            except Exception as e:
                if "Unknown FEC scheme" in str(e) or "rs255_223" in str(e) or "conv_k7" in str(e):
                    old = cfg.link.fec_scheme
                    cfg.link.fec_scheme = "hamming74"
                    tx_syms, _ = build_transmission(hdr_b, payload, cfg)
                    print(f"[WARN] FEC '{old}' unsupported. Fallback to 'hamming74' for {m}.")
                else:
                    raise
            return len(tx_syms), hdr, payload
        else:
            # serialize fresh
            cfg_serialize = CFG.build_config(modality=m)  # app-layer settings only
            inp = inputs[m]
            hdr, payload = serialize_content(m, inp, app_cfg=cfg_serialize.app, text_encoding="utf-8", validate_image_mode=True)
            prebuilt[m] = {"hdr": hdr, "payload": payload}
            hdr_b = hdr.to_bytes()
            try:
                tx_syms, _ = build_transmission(hdr_b, payload, cfg)
            except Exception as e:
                if "Unknown FEC scheme" in str(e) or "rs255_223" in str(e) or "conv_k7" in str(e):
                    old = cfg.link.fec_scheme
                    cfg.link.fec_scheme = "hamming74"
                    tx_syms, _ = build_transmission(hdr_b, payload, cfg)
                    print(f"[WARN] FEC '{old}' unsupported. Fallback to 'hamming74' for {m}.")
                else:
                    raise
            return len(tx_syms), hdr, payload

    mcs = {m: init_mcs[m].copy() for m in MODALITIES}
    # initial estimate
    sym_len = {}
    total = 0
    for m in MODALITIES:
        n, hdr, payload = estimate_for_mod(m, mcs[m])
        sym_len[m] = n; total += n
        prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}

    tol = int(max(1, round(target_total_syms * 0.01)))  # ±1% tolerance
    max_iter = 20

    # Which modalities to adjust first?
    if priority == "edge":
        lighten_order = ["segmentation", "depth", "text"]  # do not lighten 'edge'
        strengthen_order = ["segmentation", "depth", "text"]  # fill budget here first
    elif priority == "depth":
        lighten_order = ["segmentation", "edge", "text"]
        strengthen_order = ["segmentation", "edge", "text"]
    else:
        lighten_order = ["segmentation", "depth", "edge", "text"]
        strengthen_order = ["segmentation", "depth", "edge", "text"]

    it = 0
    while abs(total - target_total_syms) > tol and it < max_iter:
        it += 1
        if total > target_total_syms:
            # Too heavy → lighten non-priority modalities
            changed = False
            for m in lighten_order:
                if m == priority: 
                    continue
                cand = _next_lighter(mcs[m])
                if cand is None: 
                    continue
                # try this step
                mcs[m] = cand
                n, hdr, payload = estimate_for_mod(m, mcs[m])
                total = sum(sym_len[x] if x != m else n for x in MODALITIES)
                sym_len[m] = n
                prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}
                changed = True
                if abs(total - target_total_syms) <= tol:
                    break
            if not changed:
                break  # cannot lighten further
        else:
            # Too light → strengthen non-priority modalities to fill budget
            changed = False
            for m in strengthen_order:
                if m == priority:
                    continue
                cand = _next_stronger(mcs[m])
                if cand is None:
                    continue
                mcs[m] = cand
                n, hdr, payload = estimate_for_mod(m, mcs[m])
                total = sum(sym_len[x] if x != m else n for x in MODALITIES)
                sym_len[m] = n
                prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}
                changed = True
                if abs(total - target_total_syms) <= tol:
                    break
            if not changed:
                break

    return mcs, prebuilt

# ---------------------- Main runner ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, choices=["eep", "uep_edge", "uep_depth", "all"], default="all")
    ap.add_argument("--snrs", type=str, default="1,4,8,12")
    ap.add_argument("--channel", type=str, choices=["awgn","rayleigh"], default="rayleigh")
    ap.add_argument("--output_root", type=str, default="outputs/experiments")
    ap.add_argument("--drop_bad_frames", type=int, choices=[0,1], default=1)
    # Optional: custom inputs
    ap.add_argument("--edge", type=str, default=None)
    ap.add_argument("--depth", type=str, default=None)
    ap.add_argument("--seg", type=str, default=None)
    ap.add_argument("--text", type=str, default=None)
    args = ap.parse_args()

    snr_list = [float(s) for s in args.snrs.split(",") if s.strip()]
    drop_bad = bool(args.drop_bad_frames)

    # Resolve inputs
    inputs = {
        "edge": _ensure_input_path("edge", args.edge),
        "depth": _ensure_input_path("depth", args.depth),
        "segmentation": _ensure_input_path("segmentation", args.seg),
        "text": _ensure_input_path("text", args.text),
    }

    scenarios = []
    if args.scenario == "all":
        scenarios = ["eep", "uep_edge", "uep_depth"]
    else:
        scenarios = [args.scenario]

    # Seeds: fix per (snr, modality) so EEP/UEP are comparable
    base_seed = 12345

    for scen in scenarios:
        out_root = os.path.join(args.output_root, scen)
        os.makedirs(out_root, exist_ok=True)
        summary_csv = os.path.join(out_root, f"summary_{scen}.csv")
        if not os.path.isfile(summary_csv):
            with open(summary_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "scenario","snr_db","modality","fec","modulation","interleaver_depth",
                    "symbols_tx","frames","bad_frames","all_crc_ok","psnr_db","output_path"
                ])

        # 1) Baseline EEP total symbols (target for equal airtime)
        eep_mcs = _baseline_eep_mcs()
        # estimate once (snr doesn't affect symbol count)
        eep_sym_total = 0
        eep_prebuilt = {}
        for m in MODALITIES:
            cfg = _build_cfg(m, eep_mcs[m], args.channel, snr_db=snr_list[0], seed=base_seed, drop_bad_frames=drop_bad)
            n, hdr, payload = _estimate_symbols_for_modality(m, inputs[m], cfg)
            eep_sym_total += n
            eep_prebuilt[m] = {"hdr": hdr, "payload": payload, "sym_len": n}

        # 2) Scenario initial MCS
        if scen == "eep":
            final_mcs = eep_mcs
            prebuilt = eep_prebuilt
            priority = "none"
        elif scen == "uep_edge":
            init = _uep_edge_initial()
            final_mcs, prebuilt = _equal_airtime_adjust(
                target_total_syms=eep_sym_total, init_mcs=init, inputs=inputs,
                chan_type=args.channel, snr_db=snr_list[0], seed=base_seed,
                drop_bad_frames=drop_bad, priority="edge"
            )
            priority = "edge"
        else:  # uep_depth
            init = _uep_depth_initial()
            final_mcs, prebuilt = _equal_airtime_adjust(
                target_total_syms=eep_sym_total, init_mcs=init, inputs=inputs,
                chan_type=args.channel, snr_db=snr_list[0], seed=base_seed,
                drop_bad_frames=drop_bad, priority="depth"
            )
            priority = "depth"

        # Report equal-airtime check (one-line print)
        total_syms = sum(prebuilt[m]["sym_len"] for m in MODALITIES) if scen != "eep" else eep_sym_total
        diff_pct = 100.0 * (total_syms - eep_sym_total) / max(1, eep_sym_total)
        print(f"[{scen}] Equal-airtime: total_syms={total_syms}, baseline={eep_sym_total}, diff={diff_pct:+.2f}%")

        # 3) Run transmissions for each SNR
        for snr_db in snr_list:
            snr_dir = os.path.join(out_root, f"snr_{int(snr_db)}dB")
            os.makedirs(snr_dir, exist_ok=True)
            # stable per (snr, modality)
            records = []
            for idx, m in enumerate(MODALITIES):
                # build cfg for this modality
                seed = base_seed + int(snr_db) * 100 + idx
                cfg = _build_cfg(m, final_mcs[m], args.channel, snr_db=snr_db, seed=seed, drop_bad_frames=drop_bad)

                # make output dir
                mdir = os.path.join(snr_dir, m)
                os.makedirs(mdir, exist_ok=True)

                # run
                rec = _run_single_transmission(
                    modality=m,
                    input_path=inputs[m],
                    cfg=cfg,
                    channel=args.channel,
                    snr_db=snr_db,
                    out_dir=mdir,
                    prebuilt_hdr=prebuilt[m]["hdr"] if m in prebuilt else None,
                    prebuilt_payload=prebuilt[m]["payload"] if m in prebuilt else None,
                )
                # add actual symbols used (recalculate here for exactness)
                # (Symbol count depends only on link/mod, not on channel)
                cfg_count = _build_cfg(m, final_mcs[m], args.channel, snr_db=snr_db, seed=seed, drop_bad_frames=drop_bad)
                n_sym, _, _ = _estimate_symbols_for_modality(m, inputs[m], cfg_count)
                rec["symbols_tx"] = int(n_sym)

                # write per-modality meta
                meta = {
                    "scenario": scen, "snr_db": snr_db, "modality": m,
                    "mcs": final_mcs[m].as_dict(), "symbols_tx": int(n_sym),
                    "priority": priority
                }
                write_json(os.path.join(mdir, "meta.json"), meta)

                # to summary.csv
                row = [
                    scen, snr_db, m, rec.get("fec"), rec.get("modulation"), rec.get("interleaver_depth"),
                    int(rec.get("symbols_tx", n_sym)),
                    int(rec.get("n_frames", rec.get("frames", 0))),
                    int(rec.get("n_bad_frames", rec.get("bad_frames", 0))),
                    bool(rec.get("all_crc_ok", False)),
                    (None if rec.get("psnr_db") is None else float(rec["psnr_db"])),
                    rec.get("output_path", ""),
                ]
                records.append(row)

            # append to scenario summary CSV
            with open(summary_csv, "a", newline="") as f:
                w = csv.writer(f); [w.writerow(r) for r in records]

    print("Done. See outputs under:", os.path.abspath(args.output_root))

if __name__ == "__main__":
    main()
