# Patched files & how to run

- `config.py`: sets fair baseline repetition k=1 for all modalities.
- `presets.py`: adds SNR-specific starting presets for (none) and (rxstrong) RX profiles.
- `run_multimodal_ofdm.py`: 
  - adds `--ecc-profile` (`none` or `rxstrong`) and explicit `--edge-denoise / --depth-denoise / --seg-mode` overrides,
  - new output directory structure: `outputs/<channel>/<RXec...__link...>/snr<dB>/<POWER_TAG>__N<fft>CP<cp>/<timestamp>/...`,
  - supports batched power files (dict-of-dicts or list of dicts).
- `run_sweep_ofdm.py`: batch runner to scale selected modalities around a base preset.

## Example commands (from the package root)

Replace `<PKG>` below with the package folder name that contains these modules (i.e., the folder where `config.py` lives). If your folder is `mmofdm`, use `python -m mmofdm`.

### 1) Single run (Rayleigh, SNR=6 dB, RX profile=rxstrong, preset=snr6_rxstrong_base)
python -m <PKG>.run_multimodal_ofdm --snr_db 6 --channel rayleigh --mode uep --power-preset snr6_rxstrong_base --ecc-profile rxstrong --payload-rep text=1,edge=1,depth=1,segmentation=1 --tag exp1

### 2) AWGN at 1/6/12 dB with recommended starting presets (RX none)
# SNR=1
python -m <PKG>.run_multimodal_ofdm --snr_db 1 --channel awgn --mode uep --power-preset snr1_none_base --ecc-profile none --payload-rep text=1,edge=1,depth=1,segmentation=1
# SNR=6
python -m <PKG>.run_multimodal_ofdm --snr_db 6 --channel awgn --mode uep --power-preset snr6_none_base --ecc-profile none --payload-rep text=1,edge=1,depth=1,segmentation=1
# SNR=12
python -m <PKG>.run_multimodal_ofdm --snr_db 12 --channel awgn --mode uep --power-preset snr12_none_base --ecc-profile none --payload-rep text=1,edge=1,depth=1,segmentation=1

### 3) Rayleigh at 1/6/12 dB with recommended starting presets (RX strong)
# SNR=1
python -m <PKG>.run_multimodal_ofdm --snr_db 1 --channel rayleigh --mode uep --power-preset snr1_rxstrong_base --ecc-profile rxstrong --payload-rep text=1,edge=1,depth=1,segmentation=1
# SNR=6
python -m <PKG>.run_multimodal_ofdm --snr_db 6 --channel rayleigh --mode uep --power-preset snr6_rxstrong_base --ecc-profile rxstrong --payload-rep text=1,edge=1,depth=1,segmentation=1
# SNR=12
python -m <PKG>.run_multimodal_ofdm --snr_db 12 --channel rayleigh --mode uep --power-preset snr12_rxstrong_base --ecc-profile rxstrong --payload-rep text=1,edge=1,depth=1,segmentation=1

### 4) Local sweep around a base (scale edge+depth by 0.8..1.2 at SNR=6, Rayleigh, RX strong)
python -m <PKG>.run_sweep_ofdm --base-preset snr6_rxstrong_base --targets edge,depth --scales 0.8,0.9,1.0,1.1,1.2 --snr-list 6 --channel rayleigh --ecc-profile rxstrong --tag sweep_v1

### 5) Batch via power-file (dict-of-dicts)
# power.json content example:
# {
#   "v1": {"text": 0.6, "edge": 2.6, "depth": 2.2, "segmentation": 0.6},
#   "v2": {"text": 0.6, "edge": 2.8, "depth": 2.6, "segmentation": 0.0}
# }
python -m <PKG>.run_multimodal_ofdm --snr_db 1 --channel rayleigh --mode uep --power-file power.json --ecc-profile none --payload-rep text=1,edge=1,depth=1,segmentation=1 --tag sweep_from_file

