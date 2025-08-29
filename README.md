# Vehicular Network Simulation (Python)

This reference implementation simulates a **vehicular communication link** end‑to‑end, from **application** (text/images) through a **link layer** (framing, CRC, interleaving, FEC) down to a simplified **physical layer** (BPSK/QPSK/16‑QAM) and **wireless channels** (AWGN or Rayleigh with Doppler). It is tailored for **multimodal semantic communication** experiments with the following modalities:

- **Text** (`.txt`, UTF‑8)
- **Edge** images (binary, PNG)
- **Depth** images (grayscale, PNG)
- **Segmentation** images (RGB, PNG)

**Key design choice to avoid metadata corruption:** we serialize *content* (text characters, pixel arrays) rather than raw container bytes. The receiver always reconstructs valid `.txt`/`.png` files even if there are residual bit errors (which manifest as character substitutions or pixel value errors), so files remain openable.

---

## Project layout

```
vehicular_network_sim/
├── app_layer/
│   └── application.py         # content (de)serialization per modality
├── channel/
│   └── channel_model.py       # AWGN, Rayleigh, equalization helper
├── common/
│   ├── config.py              # dataclasses for configuration (app/link/phy/channel)
│   └── utils.py               # bits/bytes, CRC-32, interleaver, PSNR
├── data_link_layer/
│   ├── encoding.py            # framing, CRC, FEC + interleaving (tx/rx)
│   └── error_correction.py    # None, Repetition(k), Hamming(7,4), RS(255,223)*
├── physical_layer/
│   └── modulation.py          # BPSK/QPSK/16QAM mappers + PHY frame (preamble+pilots)
├── receiver/
│   └── receive.py             # equalize, demod, decode, reassemble
├── transmitter/
│   └── send.py                # packetize and build tx symbols
└── examples/
    ├── simulate_image_transmission.py
    └── simulate_text_transmission.py
```

\* RS(255,223) requires `reedsolo` (optional). If not installed, the code falls back gracefully.

---

## Install

Python 3.9+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -U numpy pillow
# Optional FEC:
pip install reedsolo
```

> No obscure packages are required. If you later want LDPC/Viterbi/OFDM, you can slot them in at the link/phy layers.

---

## Run (Text)

```bash
python examples/simulate_text_transmission.py \
  --input examples/sample.txt \  # provide your own .txt if not present
  --snr_db 8 \
  --channel rayleigh \
  --modulation qpsk \
  --fec hamming74 \
  --interleaver 8 \
  --mtu 1024 \
  --doppler_hz 50 \
  --symbol_rate 1e6 \
  --output received_text.txt
```

The script prints frame‑level stats and an approximate **Character Error Rate (CER)**. The output is always a valid `.txt` file thanks to robust decoding (`errors="replace"`).

---

## Run (Images)

```bash
python examples/simulate_image_transmission.py \
  --modality depth \
  --input path/to/your_depth.png \
  --snr_db 10 \
  --channel rayleigh \
  --modulation qpsk \
  --fec hamming74 \
  --interleaver 8 \
  --mtu 1500 \
  --doppler_hz 30 \
  --symbol_rate 1e6 \
  --output received.png
```

The script prints per‑frame stats and **PSNR** of the reconstructed image vs. the original. Output is always a valid `.png` file; visual degradation reflects residual errors after FEC/CRC/interleaving.

---

## Layer choices (brief rationale)

- **Application**: serialize *semantics* (strings/pixels), not container bytes. Prevents header/metadata corruption; makes degradation analyzable (CER/PSNR) across modalities.
- **Link**: simple framing with **CRC‑32** per frame; **block interleaver**; pluggable **FEC** (None, Repetition(k), **Hamming(7,4)**, **RS(255,223)**). The **app header** is protected more strongly (repetition+Hamming by default) because losing shape/text metadata would compromise decoding.
- **PHY**: **BPSK/QPSK/16‑QAM** hard‑decision demodulation. Each frame has a **preamble** (for visibility) and **pilots** for **one‑tap channel equalization**.
- **Channel**: **AWGN** or **Rayleigh fading** with optional **Doppler** via an AR(1) approximation. `block_fading` can emulate slow fading (e.g., per‑frame constant channel).

These defaults emulate DSRC/ITS‑G5 / C‑V2X flavors at a high level without reproducing their full stacks, keeping the code concise and hackable for research.

---

## Customize

All knobs are in `common/config.py` and exposed via CLI:

- `--channel {awgn,rayleigh}`
- `--snr_db`, `--doppler_hz`, `--symbol_rate`, `--block_fading`
- `--modulation {bpsk,qpsk,16qam}`
- `--fec {none,repeat,hamming74,rs255_223}` + `--repeat_k`
- `--interleaver`, `--mtu`
- Header protection and frame dropping behavior are in `LinkConfig`.

---

## Notes & Extensions

- **Unequal error protection (UEP)**: by default, the **APP header** is encoded with Hamming + Repetition in addition to the selected FEC. You can extend UEP to protect the first N bytes of image payload (e.g., critical segments) similarly.
- **Metrics**: We include PSNR for images. You can plug in SSIM/LPIPS if desired.
- **OFDM/802.11p**: Add an OFDM mapper around the current symbol stream; pilots/preamble are already factored, so hooking in FFT/IFFT and guard intervals is straightforward.
- **Semantic codecs**: For multimodal semantic comms, swap the "raw pixel bytes" with compact semantic embeddings and transmit those instead—this code will still work as the transport.

---

## Troubleshooting

- If you select `rs255_223` but do not have `reedsolo` installed, you'll see a warning and the code will fall back to `NoFEC`.
- The simulator assumes **perfect frame timing** (receiver knows frame boundaries). Preamble is included for realism but not used for sync in this reference version.
- For very low SNR with `drop_bad_frames=True`, reassembled files may be truncated (by design). With `drop_bad_frames=False`, errors pass through and are visible in content.

---

## License

MIT (for research/teaching use). Contributions welcome.
