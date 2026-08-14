"""Sanity check before building the role-assignment classifier: does merging
organism+exuviae into one class actually recover recall? Compares raw
detection recall (any box IoU>=0.5 vs each GT entity, ignoring class label)
between the single-class model and the 2-class GROUPED_SPLIT model, on the
same val images. If single-class recall isn't clearly better, the
"single-class YOLO + XGBoost role classifier" idea isn't worth building out.

Usage:
    cd vision
    python scripts/pipeline/singleclass_recall_check.py \
        --model_2class ../models/yolo_detect_GROUPED_SPLIT.pt \
        --model_1class ../models/yolo11n_detect_SINGLECLASS.pt
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
SPLIT_CSV = BASE_DIR / "data" / "master_split.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"

_cli = argparse.ArgumentParser()
_cli.add_argument("--model_2class", default=str(BASE_DIR / "models" / "yolo_detect_GROUPED_SPLIT.pt"))
_cli.add_argument("--model_1class", default=str(BASE_DIR / "models" / "yolo11n_detect_SINGLECLASS.pt"))
_cli.add_argument("--conf", type=float, default=0.35)
_args = _cli.parse_args()

IOU_OK = 0.5


def _area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1, ix2, iy2 = max(x1, X1), max(y1, Y1), min(x2, X2), min(y2, Y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (_area(*b1) + _area(*b2) - inter + 1e-9)


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def recall_any_class(model, val_rows):
    hit_org = hit_exu = 0
    for _, row in val_rows.iterrows():
        img_path = IMAGE_ROOT / row["stage"] / row["filename"]
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")
        r = model.predict(pil, imgsz=1280, conf=_args.conf, verbose=False)[0]
        boxes = [[float(v) for v in b] for b in r.boxes.xyxy.cpu().numpy()] if r.boxes is not None else []

        gt_o, gt_e = gt_box(row, "organism"), gt_box(row, "exuviae")
        if gt_o and any(_iou(b, gt_o) >= IOU_OK for b in boxes):
            hit_org += 1
        if gt_e and any(_iou(b, gt_e) >= IOU_OK for b in boxes):
            hit_exu += 1
    n = len(val_rows)
    return hit_org / n, hit_exu / n


def main():
    df = pd.read_csv(FEATURES_CSV)
    split = pd.read_csv(SPLIT_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")].copy()
    both = both.merge(split, on="observation_id", how="inner")
    val_rows = both[both["split"] == "val"].reset_index(drop=True)
    print(f"n={len(val_rows)} val images (both-subset)\n")

    print("2-class model:", _args.model_2class)
    m2 = YOLO(_args.model_2class)
    r2_org, r2_exu = recall_any_class(m2, val_rows)
    print(f"  organism recall (IoU>=0.5) = {r2_org:.3f}   exuviae recall = {r2_exu:.3f}")

    print("\n1-class model:", _args.model_1class)
    m1 = YOLO(_args.model_1class)
    r1_org, r1_exu = recall_any_class(m1, val_rows)
    print(f"  organism recall (IoU>=0.5) = {r1_org:.3f}   exuviae recall = {r1_exu:.3f}")

    print(f"\nDelta: organism {r1_org-r2_org:+.3f}   exuviae {r1_exu-r2_exu:+.3f}")
    print("If both deltas are small/negative, the single-class idea isn't paying off --")
    print("skip building the role-assignment classifier. If clearly positive, worth pursuing.")


if __name__ == "__main__":
    main()
