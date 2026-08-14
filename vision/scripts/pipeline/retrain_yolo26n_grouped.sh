#!/bin/bash
# Retrains YOLO26n on the same grouped split as YOLO11n (data/yolo_grouped),
# using Ultralytics' own official small-dataset guidance (<1000 images):
# lighter augmentation, lower lr0, fewer epochs with early-stop patience.
# https://docs.ultralytics.com/guides/yolo26-training-recipe/
#
# YOLO26n is worth trying specifically because of STAL (Small-Target-Aware
# Label Assignment), built for exactly the failure mode we measured on
# YOLO11n: 92% of end-to-end errors were "wrong branch" (missed detections),
# not misclassification.
#
# Run on the GPU workstation from vision/scripts/pipeline/.
set -e

REPO=$(cd ../../.. && pwd)/vision

yolo task=detect mode=train \
  model=yolo26n.pt \
  data="$REPO/data/yolo_grouped/data.yaml" \
  epochs=50 patience=20 imgsz=1280 batch=-1 device=0 \
  project="$REPO/scripts/training/runs/moult_bench_grouped" \
  name=yolo26n_grouped_split seed=42 verbose=True \
  optimizer=auto lr0=0.001 \
  mosaic=0.5 mixup=0.0 copy_paste=0.0 \
  cache=True deterministic=True workers=8 plots=True

cp "$REPO/scripts/training/runs/moult_bench_grouped/yolo26n_grouped_split/weights/best.pt" \
   "$REPO/models/yolo26_detect_GROUPED_SPLIT.pt"

echo "saved -> $REPO/models/yolo26_detect_GROUPED_SPLIT.pt"

# If this still underperforms, the next thing to try (not run by default
# here) is freezing the backbone to fight overfitting on so few images:
#   freeze=10
# Add it to the yolo command above and re-run with a fresh --name.
