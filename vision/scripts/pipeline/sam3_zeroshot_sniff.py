"""Zero-shot sniff test, SAM3.1 version: same idea and same 20-image sample
as yoloe_zeroshot_sniff.py, so the two are directly comparable. No training.

Prerequisite: request access to gated weights at huggingface.co/facebook/sam3,
download sam3.pt, place at vision/models/sam3.pt.

Usage:
    cd vision
    python scripts/pipeline/sam3_zeroshot_sniff.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"
SAM3_WEIGHTS = BASE_DIR / "models" / "sam3.pt"

N_SAMPLE = 20
SEED = 42  # same sample as yoloe_zeroshot_sniff.py

TEXT_ORGANISM = "insects"
TEXT_EXUVIAE = ["shell", "husk", "empty shell", "insect shell", "empty insect shell",
                "molted exoskeleton", "shed skin", "exuvia"]


def _iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1, ix2, iy2 = max(x1, X1), max(y1, Y1), min(x2, X2), min(y2, Y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    return inter / (a1 + a2 - inter + 1e-9)


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def best_box(predictor, pil_img, text):
    predictor.set_image(np.array(pil_img))
    r = predictor(text=[text] if isinstance(text, str) else text)
    res = r[0] if isinstance(r, (list, tuple)) else r
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    return [float(v) for v in xyxy[conf.argmax()]]


def main():
    if not SAM3_WEIGHTS.exists():
        raise SystemExit(f"Missing {SAM3_WEIGHTS} -- request access at huggingface.co/facebook/sam3 first.")

    df = pd.read_csv(FEATURES_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")]
    sample = both.sample(n=min(N_SAMPLE, len(both)), random_state=SEED)

    predictor = SAM3SemanticPredictor(overrides={"conf": 0.25, "task": "segment", "mode": "predict",
                                                  "model": str(SAM3_WEIGHTS), "quantize": 16, "verbose": False})

    print(f"n={len(sample)} images, zero-shot, text='{TEXT_ORGANISM}' (organism) / {TEXT_EXUVIAE} (exuviae)\n")
    ious_org, ious_exu = [], []
    hit_org = hit_exu = 0

    for _, row in sample.iterrows():
        img_path = IMAGE_ROOT / row["stage"] / row["filename"]
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")
        gt_o, gt_e = gt_box(row, "organism"), gt_box(row, "exuviae")

        det_o = best_box(predictor, pil, TEXT_ORGANISM)
        det_e = best_box(predictor, pil, TEXT_EXUVIAE)

        iou_o = _iou(det_o, gt_o) if (det_o and gt_o) else 0.0
        iou_e = _iou(det_e, gt_e) if (det_e and gt_e) else 0.0
        ious_org.append(iou_o)
        ious_exu.append(iou_e)
        hit_org += iou_o >= 0.5
        hit_exu += iou_e >= 0.5
        print(f"{row['filename']}: organism IoU={iou_o:.2f}  exuviae IoU={iou_e:.2f}  "
              f"(found: org={det_o is not None} exu={det_e is not None})")

    n = len(ious_org)
    print(f"\n=== SUMMARY (n={n}) ===")
    print(f"organism: mean IoU={sum(ious_org)/n:.3f}  hit-rate (IoU>=0.5)={hit_org}/{n}")
    print(f"exuviae:  mean IoU={sum(ious_exu)/n:.3f}  hit-rate (IoU>=0.5)={hit_exu}/{n}")
    print("\nCompare directly against yoloe_zeroshot_sniff.py's output -- same 20 images, same seed.")


if __name__ == "__main__":
    main()
