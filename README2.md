# Vehicular Multi‑Modal Transmission (UEP vs EEP)

> **Diffusion‑ready simulation pipeline** for transmitting multi‑modal data (Edge, Depth, Segmentation, Text) over noisy channels, comparing **UEP** (Unequal Error Protection) vs **EEP** (Equal Error Protection) under an **equal airtime** (Equal‑Transmit‑Symbols, ETS) constraint. Designed to produce **received PNG/TXT** artifacts that can be fed into a diffusion model.

---

## ✨ Key Features

* **Multi‑modal TX/RX**: Edge (binary), Depth (8‑bit), Segmentation (ID↔RGB), Text (.txt)
* **Channels**: AWGN and Rayleigh fading (flat‑fading baseline; OFDM extension is easy to add)
* **FEC & Modulation**: Convolutional codes with puncturing (R∈{1/2, 2/3, 3/4}), Hamming(7,4), Repetition; BPSK/QPSK/16QAM
* **Interleaver (ILV)**: block interleaving with configurable depth
* **ETS (Equal‑Transmit‑Symbols)**: UEP is auto‑tuned to match EEP’s total symbol budget (±1%)
* **Receiver post‑processing**: light, modality‑specific error suppression (morphology for Edge, median+bilateral for Depth, voting for Seg)
* **CRC policy for diffusion**: keep CRC‑failed frames (`drop_bad_frames=0`) to avoid black screens and preserve “pepper” structure
* **Batch runner**: parallel jobs, modality filtering, robust error logging (per‑modality `error.log`)

---

## 🗂 Repository Layout (relevant parts)

```
app_layer/
  application.py        # save_output(...) and modality‑level utilities (hardened I/O)
common/                 # configs, utils, byte mapping, run utilities
configs/
  image_config.py       # default MCS/MTU etc. for image runs
examples/
  run_uep_experiment.py # robust runner (parallel, modality‑filtered, ETS logging)
  simulate_image_transmission.py  # (legacy/simple entry point)
receiver/, transmitter/, channel/, data_link_layer/, physical_layer/  # stack
outputs/                # auto‑generated artifacts (received.png/txt, rx_stats.json, meta.json)
```

> **Note**: `examples/run_uep_experiment.py` includes a single **TODO hook** inside `_run_single_transmission(...)`. Replace that line with your project’s actual TX/RX function that returns `(rx_hdr, text_str, img_arr, stats, mcs, symbols_tx)`.

---

## ✅ Prerequisites

* Python 3.10+
* `pip install -r requirements.txt` (Pillow, numpy, etc.)
* **WSL users**: Prefer working under WSL’s ext4 (e.g., `~/work/...`) instead of `/mnt/c/...` for speed and fewer I/O races.

---

## 🔧 Installation & Environment

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt

# Make packages visible
touch app_layer/__init__.py examples/__init__.py

# Option A: run scripts directly (absolute imports)
export PYTHONPATH=$(pwd)

# (Optional) keep each Python process single‑threaded for better parallel scaling
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

---

## 🚀 Quick Start

**Run EEP at SNR=1,12 (Rayleigh), keeping corrupted frames**

```bash
PYTHONPATH=$(pwd) python -u examples/run_uep_experiment.py \
  --scenario eep \
  --snrs 1,12 \
  --channel rayleigh \
  --drop_bad_frames 0 \
  --modalities edge,depth,segmentation,text \
  --fail_fast 0 \
  --jobs 4 \
  --output_root "$(pwd)/outputs/experiments"
```

**Run all three scenarios (EEP, UEP‑Edge, UEP‑Depth) in parallel**

```bash
for s in eep uep_edge uep_depth; do
  PYTHONPATH=$(pwd) python -u examples/run_uep_experiment.py \
    --scenario $s --snrs 1,4,8,12 --channel rayleigh \
    --drop_bad_frames 0 --jobs 8 \
    --output_root "$(pwd)/outputs/experiments_$s" &
done
wait
```

**Filter by modality** (e.g., Edge only): `--modalities edge`

Artifacts appear under:

```
outputs/experiments_<scenario>/snr_<XdB>/<modality>/
  received.png | received.txt
  rx_stats.json  # per‑trial stats (bad frames, PSNR, etc.)
  meta.json      # MCS, symbol count, scenario info
```

---

## ⚖️ Equal‑Transmit‑Symbols (ETS) in practice

1. The runner first **measures** EEP’s total symbols $S_{\mathrm{EEP}}=\sum_m S_m$ with overhead (preamble/pilots/padding) included.
2. UEP’s per‑modality **MCS** $(b,r,\text{ILV})$ is greedily adjusted until $\sum_m S_m\in[S_{\mathrm{EEP}}(1\!\pm\!0.01)]$.
3. The ETS attainment and symbol counts are printed/logged; captions should include this (e.g., `ΣS=12.85M, +0.7%`).

> **Tip**: At very low SNR, slightly **strengthen EEP** (e.g., QPSK+R=2/3, ILV=32; MTU=512) so comparisons remain meaningful; UEP will be auto‑matched to the new $S_{\mathrm{EEP}}$.

---

## 📡 Channels

* **AWGN**: $y_k=x_k+n_k$, $n_k\sim\mathcal{CN}(0,N_0)$. Constellations normalized to **$E_s=1$** ⇒ $\mathrm{SNR}=1/N_0$.
* **Rayleigh fading** (flat baseline): $y_k=h_kx_k+n_k$, $h_k\sim\mathcal{CN}(0,1)$. Average SNR = $E_s/N_0$; instantaneous SNR = $|h_k|^2\gamma$.

> OFDM (per‑subcarrier LS/ZF) can be enabled by swapping in the OFDM branch; count pilots/preamble/CP in ETS.

---

## 🧩 Modality‑specific suppression (RX)

* **Edge**: 3×3 **Opening** (erosion→dilation) then **Closing** (dilation→erosion), one pass, with thin‑line safeguard.
* **Depth**: **median 5×5** then **bilateral** (σs≈1.6, σr≈12), edge‑preserving smoothing.
* **Segmentation**: palette projection to nearest ID; **3×3 majority voting ×2** with consensus α=0.6.
* **Text**: keep bytes even on CRC failure (retain readable fragments for LLM correction later).

---

## 🧪 Runner CLI (common flags)

```
--scenario {eep,uep_edge,uep_depth,all}
--snrs 1,4,8,12
--channel {awgn,rayleigh}
--drop_bad_frames {0,1}
--modalities edge,depth,segmentation,text
--fail_fast {0,1}
--jobs <N>             # parallel (snr × modality)
--output_root <path>
```

**Notes**

* `--fail_fast 0` ensures one modality’s failure doesn’t abort others; check `<...>/error.log` per modality.
* Set `PYTHONPATH=$(pwd)` or run as a module (`python -m examples.run_uep_experiment`) with proper `__init__.py`.

---

## 🧱 Hardening & TODO hook

* `app_layer/application.py :: save_output(...)` is **hardened**: creates parent dirs, routes `.txt` to text, coerces image shape to valid `L/RGB` before saving.
* `examples/run_uep_experiment.py :: _run_single_transmission(...)` contains one **TODO**. Replace the placeholder with your TX/RX function returning:

  ```python
  (rx_hdr, text_str, img_arr, stats, mcs, symbols_tx)
  ```

  where `rx_hdr.content_type in {"text","edge","depth","segmentation"}`; `stats` has `psnr_db, n_bad_frames, n_frames, ...`; `mcs` has `mod, fec, interleaver_depth`.

---

## 🧷 Troubleshooting

* **`ModuleNotFoundError: app_layer`** → add `app_layer/__init__.py` and run with `PYTHONPATH=$(pwd)` from repo root, or `python -m examples.run_uep_experiment`.
* **PIL `tile cannot extend outside image`** → ensure you use the **hardened `save_output(...)`** (coerces array shapes; routes text to `.txt`).
* **No results at SNR=12** → a crash in another modality likely stopped the batch; re‑run with `--fail_fast 0` and check per‑modality `error.log`.
* **Slow on `/mnt/c`** → run under WSL ext4 (e.g., `~/work/...`) and use `--jobs` with `OMP_NUM_THREADS=1` to leverage CPU cores without oversubscription.

---

## 🔭 Roadmap

* OFDM path with per‑subcarrier LS/LMMSE + MMSE EQ; ETS counting CP/pilots explicitly
* Soft‑decision FEC (LLR) and LDPC/Polar options (same ETS)
* Auto‑captions with ETS attainment in generated figures
* Batch evaluators for FID/LPIPS/PSNR with 95% CIs (bootstrap)

---

## 📄 License

TBD (insert your project’s license).

## 🙌 Acknowledgements

Thanks to contributors and upstream libraries (NumPy, Pillow, etc.).
