#!/bin/bash
# Retrains YOLO on the grouped, leakage-free split (data/yolo_grouped).
# Same recipe as the deployed model (runs/moult_bench/yolo11n_moulting_bench_taskaware/args.yaml).
# Run on the GPU workstation from vision/scripts/pipeline/.
set -e

REPO=$(cd ../../.. && pwd)/vision

yolo task=detect mode=train \
  model=yolo11n.pt \
  data="$REPO/data/yolo_grouped/data.yaml" \
  epochs=300 imgsz=1280 batch=-1 device=0 \
  project="$REPO/scripts/training/runs/moult_bench_grouped" \
  name=yolo11n_grouped_split seed=42 verbose=True \
  optimizer=AdamW cos_lr=True weight_decay=0.0005 \
  mosaic=0.50 mixup=0.05 copy_paste=0.05 close_mosaic=20 rect=False \
  hsv_h=0.015 hsv_s=0.60 hsv_v=0.20 translate=0.12 scale=0.60 degrees=0.0 shear=0.0 \
  label_smoothing=0.005 box=7.5 cls=0.30 dfl=2.0 \
  cache=True deterministic=True patience=60 workers=8 plots=True warmup_epochs=5 \
  lr0=0.0025 lrf=0.01

cp "$REPO/scripts/training/runs/moult_bench_grouped/yolo11n_grouped_split/weights/best.pt" \
   "$REPO/models/yolo_detect_GROUPED_SPLIT.pt"

echo "saved -> $REPO/models/yolo_detect_GROUPED_SPLIT.pt"
