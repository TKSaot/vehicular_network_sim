#!/usr/bin/env bash
set -euo pipefail

SNR=${SNR:-10}
CHANNEL=${CHANNEL:-rayleigh}
TAG=${TAG:-"sweep"}
CONCURRENCY=${CONCURRENCY:-2}  # increase carefully; higher uses more RAM/VRAM

gen_cmds() {
  echo "python -m multimodal_ofdm.run_multimodal_ofdm --snr_db $SNR --channel $CHANNEL --mode eep --tag $TAG"
  for P in text edge edge_depth segmentation; do
    echo "python -m multimodal_ofdm.run_multimodal_ofdm --snr_db $SNR --channel $CHANNEL --mode uep --power-preset $P --tag $TAG"
  done
}

# Parallelize with xargs (GNU parallel also works)
gen_cmds | xargs -I CMD -P "$CONCURRENCY" bash -lc "CMD"
