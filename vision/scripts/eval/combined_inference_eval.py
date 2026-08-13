#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end (detector-in-the-loop) evaluation of the deployed YOLO + XGBoost
stage-classification pipeline, using ONLY existing, already-trained models
(no retraining here — see scripts/training/ for that).

What this does, for every image in the "both organism+exuvia annotated"
subset of data/annotated_features.csv (624 images):
  1. Runs the real YOLO detector (models/yolo_detect.pt) on the raw image
     file, exactly as vision/backend/app.py does (same imgsz/conf/iou,
     same cross-class soft-NMS, same "best detection per class" logic).
  2. Applies the same decision tree described in the paper (Section 3.6):
       - only exuviae detected  -> "exuviae"
       - only organism detected -> "post-moult"
       - both detected          -> XGBoost classifier
       - neither detected       -> no prediction (counted as an error)
  3. For the "both detected" case, builds the SAME 13 features the
     classifier was actually trained on (see build_annotated_features.py /
     scripts/training/train_xgboost.py), which are RAW PIXEL coordinates
     (top-left corner, not centroid) and RAW 0-255 color means.

     IMPORTANT: this is deliberately NOT the same feature computation as
     vision/backend/app.py's build_features(). That function normalizes
     coordinates to [0,1] and divides color means by 255 -- which does not
     match how the classifier was trained (raw pixel scale, 0-255 colors).
     Feeding app.py's normalized features into this classifier is a real
     train/serve skew bug in the live demo; this script does NOT reproduce
     that bug, so the numbers below reflect the classifier's true
     capability, not the (likely degraded) behaviour of the current
     /predict_image endpoint. Worth fixing in app.py separately.

  4. Compares the predicted stage to the true annotated stage.
  5. For every error, also computes IoU between YOLO's predicted box and
     the manually-annotated ground-truth box for organism/exuviae, so you
     can tell whether an error traces back to detection (YOLO missed an
     entity, or its box barely overlaps the true one) or to classification
     (both entities detected accurately, but the classifier picked the
     wrong stage).

Caveat to keep in mind when reading the results: we don't have a record of
which images were in the ORIGINAL OFFICIAL_SPLIT validation set (that
split's script/notebook could not be located -- see conversation), so this
script evaluates on the full 624-image "both" pool, which likely overlaps
with the classifier's own training data. Treat this as "how the deployed
combination behaves across the corpus" rather than a clean generalization
number. A separate, leakage-free retrain+eval (grouped by observation_id)
was already done and gave val_macro_f1 = 0.719 on a proper 121-image
held-out split -- useful as a cross-check ceiling for how much the numbers
here might be inflated by the classifier having seen some of these exact
images during its own training.

Usage:
    cd vision
    python scripts/eval/combined_inference_eval.py

Requires the same environment vision/backend/app.py already runs in
(torch, ultralytics, xgboost, pandas, scikit-learn, pillow).

Output:
    scripts/results/combined_inference_eval.csv  -- one row per image
    Console summary: accuracy, macro F1, confusion matrix, and an
    error-attribution breakdown.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageStat
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

# ───────────────────────── Config ─────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # vision/

FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"

YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo_detect.pt"
# The classifier whose numbers are reported in the paper (Table 5),
# NOT vision/backend/app.py's current default (models/xgboost_stage.pkl),
# which is a different, separately-trained file.
CLF_MODEL_PATH = BASE_DIR / "models" / "xgboost_stage_OFFICIAL_SPLIT_13feat.pkl"
CLF_ENCODER_PATH = BASE_DIR / "models" / "label_encoder_OFFICIAL_SPLIT_13feat.pkl"

OUT_CSV = BASE_DIR / "scripts" / "results" / "combined_inference_eval.csv"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 1024
IOU_CROSS_NMS = 0.50

TRAIN_FEATURES = [
    "box_overlap", "dist_centroids", "x_organism", "y_organism",
    "x_exuviae", "y_exuviae", "h_exuviae",
    "org_mean_g", "org_mean_gray",
    "taxon_group_Crustacea", "taxon_group_Hexapoda",
    "taxon_group_Chelicerata", "taxon_group_Myriapoda",
]
VALID_TAXON_GROUPS = ["Chelicerata", "Crustacea", "Hexapoda", "Myriapoda"]


# ───────────────────────── Geometry helpers (mirrors app.py) ─────────────────────────
def _area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_boxes(b1, b2):
    if not b1 or not b2:
        return 0.0
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _area(*b1) + _area(*b2) - inter + 1e-9
    return float(inter / union)


def _filter_cross_overlap(dets_by_cls, thr=0.5):
    orgs = dets_by_cls.get("organism", [])
    exus = dets_by_cls.get("exuviae", [])
    keep_orgs, keep_exus = [], []
    for o in orgs:
        if any(_iou_boxes(o["box"], e["box"]) > thr and e["conf"] > o["conf"] for e in exus):
            continue
        keep_orgs.append(o)
    for e in exus:
        if any(_iou_boxes(e["box"], o["box"]) > thr and o["conf"] > e["conf"] for o in orgs):
            continue
        keep_exus.append(e)
    dets_by_cls["organism"] = keep_orgs
    dets_by_cls["exuviae"] = keep_exus


def bucket(name_or_id, names):
    if isinstance(names, dict):
        label = names.get(int(name_or_id), str(name_or_id))
    else:
        try:
            label = names[int(name_or_id)]
        except Exception:
            label = str(name_or_id)
    return "exuviae" if "exuv" in str(label).lower() else "organism"


# ───────────────────────── Feature builder matching TRAINING scale ─────────────────────────
def build_training_scale_features(box_o_xyxy, box_e_xyxy, pil_img, taxon_group):
    """
    Reproduces the exact feature definitions used by
    scripts/training/build_annotated_features.py + train_xgboost.py:
    RAW pixel coordinates (top-left corner, not centroid, not normalized),
    RAW 0-255 color means (not divided by 255).
    """
    f = {n: -1.0 for n in TRAIN_FEATURES}

    if box_o_xyxy:
        f["x_organism"] = float(box_o_xyxy[0])
        f["y_organism"] = float(box_o_xyxy[1])
    if box_e_xyxy:
        f["x_exuviae"] = float(box_e_xyxy[0])
        f["y_exuviae"] = float(box_e_xyxy[1])
        f["h_exuviae"] = float(box_e_xyxy[3] - box_e_xyxy[1])

    f["box_overlap"] = _iou_boxes(box_o_xyxy, box_e_xyxy) if (box_o_xyxy and box_e_xyxy) else 0.0

    if box_o_xyxy and box_e_xyxy:
        cx_o = (box_o_xyxy[0] + box_o_xyxy[2]) / 2.0
        cy_o = (box_o_xyxy[1] + box_o_xyxy[3]) / 2.0
        cx_e = (box_e_xyxy[0] + box_e_xyxy[2]) / 2.0
        cy_e = (box_e_xyxy[1] + box_e_xyxy[3]) / 2.0
        f["dist_centroids"] = float(math.hypot(cx_o - cx_e, cy_o - cy_e))

    # Organism crop color stats, RAW 0-255 scale (no /255)
    try:
        if box_o_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in box_o_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            if x2 > x1 and y2 > y1:
                crop = pil_img.crop((x1, y1, x2, y2))
                stat_rgb = ImageStat.Stat(crop)
                mean_g = stat_rgb.mean[1] if len(stat_rgb.mean) >= 2 else 0.0
                gray = crop.convert("L")
                mean_gray = ImageStat.Stat(gray).mean[0]
                f["org_mean_g"] = float(mean_g)
                f["org_mean_gray"] = float(mean_gray)
    except Exception:
        pass

    for g in VALID_TAXON_GROUPS:
        f[f"taxon_group_{g}"] = 1.0 if taxon_group == g else 0.0

    for k, v in f.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            f[k] = -1.0
    return f


def build_image_path(stage: str, filename: str) -> Path:
    return IMAGE_ROOT / stage / filename


# ───────────────────────── Main ─────────────────────────
def main():
    print(f"[INFO] Loading YOLO from {YOLO_MODEL_PATH}")
    yolo = YOLO(str(YOLO_MODEL_PATH))
    names = yolo.names if hasattr(yolo, "names") else yolo.model.names

    print(f"[INFO] Loading classifier from {CLF_MODEL_PATH}")
    clf = joblib.load(CLF_MODEL_PATH)
    label_encoder = None
    if CLF_ENCODER_PATH.exists():
        label_encoder = joblib.load(CLF_ENCODER_PATH)

    df = pd.read_csv(FEATURES_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1)].copy()
    both = both[both["stage"] != "pre-moult"].reset_index(drop=True)
    print(f"[INFO] Evaluating on {len(both)} images (both organism+exuvia annotated)")

    rows = []
    for i, row in both.iterrows():
        img_path = build_image_path(row["stage"], row["filename"])
        if not img_path.exists():
            print(f"[WARN] missing image, skipping: {img_path}")
            continue

        pil = Image.open(img_path).convert("RGB")

        res = yolo.predict(pil, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        r = res[0]
        dets_by_cls = {"organism": [], "exuviae": []}
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                dets_by_cls[bucket(c, names)].append(
                    {"box": [float(x1), float(y1), float(x2), float(y2)], "conf": float(p)}
                )
        _filter_cross_overlap(dets_by_cls, IOU_CROSS_NMS)

        best = {"organism": None, "exuviae": None}
        best_conf = {"organism": -1.0, "exuviae": -1.0}
        for k in ("organism", "exuviae"):
            for d in dets_by_cls[k]:
                if d["conf"] > best_conf[k]:
                    best_conf[k] = d["conf"]
                    best[k] = d["box"]

        has_org = best["organism"] is not None
        has_exu = best["exuviae"] is not None

        if has_exu and not has_org:
            pred_stage = "exuviae"
            branch = "rule_only_exuviae"
        elif has_org and not has_exu:
            pred_stage = "post-moult"
            branch = "rule_only_organism"
        elif has_org and has_exu:
            feat = build_training_scale_features(best["organism"], best["exuviae"], pil, row["taxon_group"])
            order = list(getattr(clf, "feature_names_in_", TRAIN_FEATURES))
            # NOTE: the OFFICIAL_SPLIT model is a sklearn Pipeline whose
            # ColumnTransformer selects columns by name, so it needs a
            # DataFrame here -- a bare numpy array raises "Specifying the
            # columns using strings is only supported for dataframes."
            X = pd.DataFrame([[float(feat.get(n, -1.0)) for n in order]], columns=order)
            pred_idx = int(clf.predict(X)[0])
            if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
                pred_stage = str(label_encoder.inverse_transform([pred_idx])[0])
            elif hasattr(clf, "classes_"):
                pred_stage = str(clf.classes_[pred_idx])
            else:
                pred_stage = ["post-moult", "moulting", "exuviae"][pred_idx]
            branch = "classifier"
        else:
            pred_stage = None
            branch = "no_detection"

        # Ground-truth boxes (xywh -> xyxy) for detection-error attribution
        def gt_box(prefix):
            x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
            if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
                return None
            return [float(x), float(y), float(x + w), float(y + h)]

        gt_org = gt_box("organism")
        gt_exu = gt_box("exuviae")
        iou_org = _iou_boxes(best["organism"], gt_org) if (best["organism"] and gt_org) else (0.0 if gt_org else None)
        iou_exu = _iou_boxes(best["exuviae"], gt_exu) if (best["exuviae"] and gt_exu) else (0.0 if gt_exu else None)

        correct = (pred_stage == row["stage"])
        rows.append({
            "filename": row["filename"],
            "observation_id": row["observation_id"],
            "true_stage": row["stage"],
            "pred_stage": pred_stage,
            "correct": correct,
            "branch": branch,
            "has_org_detected": has_org,
            "has_exu_detected": has_exu,
            "iou_organism_vs_gt": iou_org,
            "iou_exuviae_vs_gt": iou_exu,
        })

        if (i + 1) % 50 == 0:
            print(f"[INFO] {i+1}/{len(both)} processed")

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[✓] Saved per-image results to {OUT_CSV}")

    # ───────────── Summary ─────────────
    valid = out_df[out_df["pred_stage"].notna()]
    n_no_detection = (out_df["pred_stage"].isna()).sum()

    acc = accuracy_score(valid["true_stage"], valid["pred_stage"])
    bal_acc = balanced_accuracy_score(valid["true_stage"], valid["pred_stage"])
    macro_f1 = f1_score(valid["true_stage"], valid["pred_stage"], average="macro")

    print("\n=== COMBINED DETECTOR-IN-THE-LOOP RESULTS ===")
    print(f"n images: {len(out_df)}  (no-detection cases excluded from accuracy: {n_no_detection})")
    print(f"accuracy: {acc:.3f}")
    print(f"balanced_accuracy: {bal_acc:.3f}")
    print(f"macro_f1: {macro_f1:.3f}")
    labels = sorted(set(valid["true_stage"]) | set(valid["pred_stage"]))
    print("\nconfusion matrix", labels)
    print(confusion_matrix(valid["true_stage"], valid["pred_stage"], labels=labels))

    # ───────────── Error attribution ─────────────
    errors = valid[~valid["correct"]].copy()
    IOU_OK = 0.5  # "detection roughly correct" threshold

    def attribute(r):
        det_ok_org = (r["iou_organism_vs_gt"] is not None) and (r["iou_organism_vs_gt"] >= IOU_OK)
        det_ok_exu = (r["iou_exuviae_vs_gt"] is not None) and (r["iou_exuviae_vs_gt"] >= IOU_OK)
        if r["branch"] != "classifier":
            return "detection (wrong branch: single/no entity detected, should have been both)"
        if det_ok_org and det_ok_exu:
            return "classification (both boxes accurate, XGBoost picked wrong stage)"
        return "detection (organism/exuvia box inaccurate, IoU < 0.5 vs ground truth)"

    if len(errors):
        errors["error_cause"] = errors.apply(attribute, axis=1)
        print(f"\n=== ERROR ATTRIBUTION ({len(errors)} errors) ===")
        print(errors["error_cause"].value_counts())
        errors.to_csv(OUT_CSV.with_name("combined_inference_eval_errors.csv"), index=False)
        print(f"[✓] Saved error breakdown to {OUT_CSV.with_name('combined_inference_eval_errors.csv')}")
    else:
        print("\nNo errors — nothing to attribute.")


if __name__ == "__main__":
    main()
