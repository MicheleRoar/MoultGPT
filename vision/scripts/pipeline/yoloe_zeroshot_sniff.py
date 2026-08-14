"""Zero-shot sniff test: does YOLOE find organism/exuviae with NO fine-tuning
at all, just text prompts? No training, no GPU strictly required (small
model, ~20 images). If this looks promising we invest more; if it's garbage
we drop it in five minutes instead of an afternoon.

Auto-downloads yoloe-11s-seg.pt on first run (~30MB, open weights, no gating
unlike SAM3).

Usage:
    cd vision
    python scripts/pipeline/yoloe_zeroshot_sniff.py
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"

N_SAMPLE = 20
SEED = 42

TEXT_ORGANISM = ["arthropod", "insect", "spider", "crab", "crustacean"]
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


def main():
    df = pd.read_csv(FEATURES_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")]
    sample = both.sample(n=min(N_SAMPLE, len(both)), random_state=SEED)

    model = YOLO("yoloe-11s-seg.pt")

    print(f"n={len(sample)} images, zero-shot, no fine-tuning\n")
    ious_org, ious_exu = [], []
    hit_org = hit_exu = 0

    for _, row in sample.iterrows():
        img_path = IMAGE_ROOT / row["stage"] / row["filename"]
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")
        gt_o, gt_e = gt_box(row, "organism"), gt_box(row, "exuviae")

        model.set_classes(TEXT_ORGANISM)
        r_o = model.predict(pil, conf=0.15, verbose=False)[0]
        best_o = None
        if r_o.boxes is not None and len(r_o.boxes) > 0:
            i = r_o.boxes.conf.argmax()
            best_o = [float(v) for v in r_o.boxes.xyxy[i].cpu().numpy()]

        model.set_classes(TEXT_EXUVIAE)
        r_e = model.predict(pil, conf=0.15, verbose=False)[0]
        best_e = None
        if r_e.boxes is not None and len(r_e.boxes) > 0:
            i = r_e.boxes.conf.argmax()
            best_e = [float(v) for v in r_e.boxes.xyxy[i].cpu().numpy()]

        iou_o = _iou(best_o, gt_o) if (best_o and gt_o) else 0.0
        iou_e = _iou(best_e, gt_e) if (best_e and gt_e) else 0.0
        ious_org.append(iou_o)
        ious_exu.append(iou_e)
        hit_org += iou_o >= 0.5
        hit_exu += iou_e >= 0.5
        print(f"{row['filename']}: organism IoU={iou_o:.2f}  exuviae IoU={iou_e:.2f}  "
              f"(found: org={best_o is not None} exu={best_e is not None})")

    n = len(ious_org)
    print(f"\n=== SUMMARY (n={n}) ===")
    print(f"organism: mean IoU={sum(ious_org)/n:.3f}  hit-rate (IoU>=0.5)={hit_org}/{n}")
    print(f"exuviae:  mean IoU={sum(ious_exu)/n:.3f}  hit-rate (IoU>=0.5)={hit_exu}/{n}")
    print("\nRule of thumb: hit-rate below ~0.3-0.4 on this quick check means zero-shot")
    print("isn't worth pursuing further without fine-tuning YOLOE on your own data.")


if __name__ == "__main__":
    main()
