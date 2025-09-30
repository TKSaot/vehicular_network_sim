#!/bin/bash
PKG=multimodal_ofdm
CHAN=rayleigh
EXDIR=examples
FAIR='text=1,edge=1,depth=1,segmentation=1'

# === SNR=1 dB ===
for preset in eep geom_pair_x2 text_edge_x2 text_depth_x2 low_seg boost_edge_x2 boost_depth_x2; do
  python -m $PKG --channel $CHAN --snr_db 1 --power-preset "$preset" \
    --examples-dir "$EXDIR" --payload-rep "$FAIR" \
    --tag "ec=none,snr=1,preset=$preset"
done

for P in \
  "text=0.6,edge=1.6,depth=1.6,segmentation=0.2" \
  "text=0.7,edge=1.8,depth=1.3,segmentation=0.2" \
  "text=0.5,edge=1.5,depth=1.7,segmentation=0.3" \
; do
  python -m $PKG --channel $CHAN --snr_db 1 --mode uep --power "$P" \
    --examples-dir "$EXDIR" --payload-rep "$FAIR" \
    --tag "ec=none,snr=1,power"
done

# === SNR=6 dB ===
for preset in eep geom_pair_x2 text_edge_x2 text_depth_x2 boost_depth_x2; do
  python -m $PKG --channel $CHAN --snr_db 6 --power-preset "$preset" \
    --examples-dir "$EXDIR" --payload-rep "$FAIR" \
    --tag "ec=none,snr=6,preset=$preset"
done

for P in \
  "text=0.9,edge=1.4,depth=1.4,segmentation=0.3" \
  "text=1.0,edge=1.3,depth=1.3,segmentation=0.4" \
  "text=1.1,edge=1.2,depth=1.2,segmentation=0.5" \
; do
  python -m $PKG --channel $CHAN --snr_db 6 --mode uep --power "$P" \
    --examples-dir "$EXDIR" --payload-rep "$FAIR" \
    --tag "ec=none,snr=6,power"
done

# === SNR=12 dB ===
for preset in eep boost_text_x2 geom_pair_x2 low_seg; do
  python -m $PKG --channel $CHAN --snr_db 12 --power-preset "$preset" \
    --examples-dir "$EXDIR" --payload-rep "$FAIR" \
    --tag "ec=none,snr=12,preset=$preset"
done

for P in \
  "text=1.2,edge=1.0,depth=1.0,segmentation=0.8" \
  "text=1.3,edge=1.1,depth=1.0,segmentation=0.6" \
  "text=1.1,edge=1.1,depth=1.1,segmentation=0.7" \
; do
  python -m $PKG --channel $CHAN --snr_db 12 --mode uep --power "$P" \
    --examples-dir "$EXDIR" --payload-rep "$FAIR" \
    --tag "ec=none,snr=12,power"
done
