
# Multimodal UEP/EEP OFDM Simulator

This package simulates transmission of **four modalities** — `text`, `edge`, `depth`, and `segmentation` — over an OFDM link with Rayleigh or AWGN channels.
Only the **power allocation per modality** differs between UEP and EEP cases; all other link parameters remain identical.

Key features:
- BPSK, Hamming(7,4) FEC, block interleaver
- OFDM with pilots (first OFDM symbol), per-subcarrier equalization
- Per-modality fixed resource mapping (disjoint sub-carrier groups), **UEP via power scaling**
- Robust, CRC-protected—and power boosted—**application headers**
- Segmentation: RGB↔ID mapping preserved; white boundary suppression; palette carried in-process (no corruption)

## Quick start

```bash
python -m multimodal_ofdm.run_multimodal_ofdm --snr_db 10 --channel rayleigh --mode eep
python -m multimodal_ofdm.run_multimodal_ofdm --snr_db 10 --channel rayleigh --mode uep --power "text=3,edge=1,depth=1,segmentation=2"
```

Inputs default to the sample images/text placed alongside this notebook/session (`/mnt/data`).  
Outputs (PNG/TXT and JSON report) are written under `outputs/` with a timestamped folder.

## Notes
- Only **power ratios** change between EEP and UEP. We keep identical MTU, interleaver depth, pilots, and FEC for every modality.
- Headers are always transmitted with a fixed high power boost (not part of UEP/EPP comparison) to guarantee metadata integrity.
- The segmentation palette/ID mapping logic is adapted from your original `まとめ.py`.
