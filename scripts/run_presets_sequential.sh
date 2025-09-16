#!/usr/bin/env bash
set -euo pipefail

SNR=${SNR:-10}
CHANNEL=${CHANNEL:-rayleigh}
TAG=${TAG:-"sweep"}
PRESETS=( eep text edge edge_depth segmentation )

for P in "${PRESETS[@]}"; do
  echo ">>> Running preset: $P"
  if [[ "$P" == "eep" ]]; then
    python -m multimodal_ofdm.run_multimodal_ofdm --snr_db "$SNR" --channel "$CHANNEL" --mode eep --tag "$TAG"
  else
    python -m multimodal_ofdm.run_multimodal_ofdm --snr_db "$SNR" --channel "$CHANNEL" \
      --mode uep --power-preset "$P" --tag "$TAG"
  fi
done
