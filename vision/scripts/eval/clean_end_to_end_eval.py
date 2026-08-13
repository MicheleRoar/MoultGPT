#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The definitive, apples-to-apples version of the detector-in-the-loop
evaluation. Fixes two issues found in earlier passes:

1. combined_inference_eval.py used the OFFICIAL_SPLIT classifier as-is,
   whose original train/val membership is unknown (the script/notebook
   that produced it could not be located) -- so the 624-image eval it ran
   on likely included images the classifier had already seen in training.

2. Its reported macro F1 (0.473) was distorted by treating "exuviae" as a
   third scoreable class, when in the "both organism+exuvia annotated"
   subset the true label is NEVER "exuviae" by construction -- it only
   ever shows up as a wrong prediction when YOLO misses the organism.

This script instead:
  a. Rebuilds a train/validation split on the "both" subset, GROUPED by
     observation_id (StratifiedGroupKFold, 5 folds, fold 0 = validation,
     same as the separate retrain done earlier -- SEED=42, so this should
     reproduce that same ~121-image held-out set deterministically).
  b. Retrains XGBoost on the training fold only, using the same
     hyperparameters as the OFFICIAL_SPLIT model (Table 5), on the
     RAW-pixel-scale 13 features it was actually trained on.
  c. Evaluates this SAME classifier TWICE on the SAME held-out images, so
     the only thing that changes between the two numbers is the box
     source:
       - "oracle":    manually-annotated boxes (upper bound, comparable to
                      Table 5's methodology)
       - "detector":  real YOLO detections on the raw image (what a user
                      actually gets)
  d. Scores both consistently as a clean 2-class problem (moulting vs
     post-moult), logging "exuviae"-branch predictions (detection failure:
     YOLO missed the organism) as a distinct, always-wrong outcome instead
     of a third class that distorts macro F1.
  e. Attributes every detector-mode error to detection or classification,
     using IoU against the ground-truth boxes.

Usage:
    cd vision
    python scripts/eval/clean_end_to_end_eval.py

Output: console summary + scripts/results/clean_end_to_end_eval.csv
(one row per held-out image, oracle and detector predictions side by side).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageStat
from ultralytics import YOLO
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

# ───────────────────────── Config ─────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # vision/

FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo_detect.pt"
OUT_CSV = BASE_DIR / "scripts" / "results" / "clean_end_to_end_eval.csv"

SEED = 42
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 1024
IOU_CROSS_NMS = 0.50
IOU_OK = 0.5  # "detection roughly correct" threshold for error attribution

TRAIN_FEATURES = [
    "box_overlap", "dist_centroids", "x_organism", "y_organism",
    "x_exuviae", "y_exuviae", "h_exuviae",
    "org_mean_g", "org_mean_gray",
    "taxon_group_Crustacea", "taxon_group_Hexapoda",
    "taxon_group_Chelicerata", "taxon_group_Myriapoda",
]
VALID_TAXON_GROUPS = ["Chelicerata", "Crustacea", "Hexapoda", "Myriapoda"]
BEST_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=1.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=SEED, tree_method="hist", n_jobs=-1,
)


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


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def build_image_path(stage: str, filename: str) -> Path:
    return IMAGE_ROOT / stage / filename


# ───────────────────────── Feature builder (training scale) ─────────────────────────
def build_training_scale_features(box_o_xyxy, box_e_xyxy, pil_img, taxon_group):
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
        cx_o, cy_o = (box_o_xyxy[0] + box_o_xyxy[2]) / 2.0, (box_o_xyxy[1] + box_o_xyxy[3]) / 2.0
        cx_e, cy_e = (box_e_xyxy[0] + box_e_xyxy[2]) / 2.0, (box_e_xyxy[1] + box_e_xyxy[3]) / 2.0
        f["dist_centroids"] = float(math.hypot(cx_o - cx_e, cy_o - cy_e))
    try:
        if box_o_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in box_o_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            if x2 > x1 and y2 > y1:
                crop = pil_img.crop((x1, y1, x2, y2))
                stat_rgb = ImageStat.Stat(crop)
                f["org_mean_g"] = float(stat_rgb.mean[1] if len(stat_rgb.mean) >= 2 else 0.0)
                f["org_mean_gray"] = float(ImageStat.Stat(crop.convert("L")).mean[0])
    except Exception:
        pass
    for g in VALID_TAXON_GROUPS:
        f[f"taxon_group_{g}"] = 1.0 if taxon_group == g else 0.0
    for k, v in f.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            f[k] = -1.0
    return f


def classify(clf, feat):
    X = pd.DataFrame([[float(feat.get(n, -1.0)) for n in TRAIN_FEATURES]], columns=TRAIN_FEATURES)
    idx = int(clf.predict(X)[0])
    return ["moulting", "post-moult"][idx]  # class order fixed by sorted(['moulting','post-moult'])


def decide(best_org, best_exu, clf, pil_img, taxon_group):
    """Full pipeline decision for one image given detected (or oracle) boxes."""
    has_org, has_exu = best_org is not None, best_exu is not None
    if has_exu and not has_org:
        return "exuviae", "rule_only_exuviae"
    if has_org and not has_exu:
        return "post-moult", "rule_only_organism"
    if has_org and has_exu:
        feat = build_training_scale_features(best_org, best_exu, pil_img, taxon_group)
        return classify(clf, feat), "classifier"
    return None, "no_detection"


# ───────────────────────── Main ─────────────────────────
def main():
    print("[1/4] Loading data and building the grouped, leakage-free split...")
    df = pd.read_csv(FEATURES_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1)].copy()
    both = both[both["stage"] != "pre-moult"].reset_index(drop=True)

    classes = sorted(both["stage"].unique())  # ['moulting', 'post-moult']
    assert classes == ["moulting", "post-moult"], f"unexpected classes: {classes}"

    X_all = both[TRAIN_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(-1)
    y_all = np.array([classes.index(v) for v in both["stage"].values])
    groups = both["observation_id"].astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = list(sgkf.split(X_all, y_all, groups))[0]
    assert set(groups[train_idx]) & set(groups[val_idx]) == set(), "observation leakage between train/val!"
    print(f"    train_n={len(train_idx)}  val_n={len(val_idx)}  (no observation overlap, confirmed)")

    print("[2/4] Retraining XGBoost on the training fold (raw-pixel features)...")
    clf = XGBClassifier(**BEST_PARAMS)
    clf.fit(X_all.iloc[train_idx], y_all[train_idx])

    val_rows = both.iloc[val_idx].reset_index(drop=True)

    print("[3/4] Loading YOLO and running both oracle-box and real-detector inference...")
    yolo = YOLO(str(YOLO_MODEL_PATH))
    names = yolo.names if hasattr(yolo, "names") else yolo.model.names

    results = []
    for i, row in val_rows.iterrows():
        img_path = build_image_path(row["stage"], row["filename"])
        if not img_path.exists():
            print(f"[WARN] missing image, skipping: {img_path}")
            continue
        pil = Image.open(img_path).convert("RGB")

        # --- oracle: manually annotated boxes ---
        oracle_org, oracle_exu = gt_box(row, "organism"), gt_box(row, "exuviae")
        pred_oracle, branch_oracle = decide(oracle_org, oracle_exu, clf, pil, row["taxon_group"])

        # --- detector: real YOLO boxes ---
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
        pred_det, branch_det = decide(best["organism"], best["exuviae"], clf, pil, row["taxon_group"])

        iou_org = _iou_boxes(best["organism"], oracle_org) if (best["organism"] and oracle_org) else (0.0 if oracle_org else None)
        iou_exu = _iou_boxes(best["exuviae"], oracle_exu) if (best["exuviae"] and oracle_exu) else (0.0 if oracle_exu else None)

        results.append({
            "filename": row["filename"], "observation_id": row["observation_id"],
            "true_stage": row["stage"],
            "pred_oracle": pred_oracle, "branch_oracle": branch_oracle,
            "pred_detector": pred_det, "branch_detector": branch_det,
            "iou_organism_vs_gt": iou_org, "iou_exuviae_vs_gt": iou_exu,
        })

    out = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[✓] Saved per-image results to {OUT_CSV}")

    print("\n[4/4] Scoring (2-class: moulting vs post-moult; any 'exuviae' prediction here is always wrong)")

    def score(pred_col):
        d = out.copy()
        d["correct"] = d["true_stage"] == d[pred_col]
        acc = d["correct"].mean()

        rows = []
        for cls_name in ["moulting", "post-moult"]:
            tp = ((d["true_stage"] == cls_name) & (d[pred_col] == cls_name)).sum()
            fn = ((d["true_stage"] == cls_name) & (d[pred_col] != cls_name)).sum()
            fp = ((d["true_stage"] != cls_name) & (d[pred_col] == cls_name)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append((cls_name, prec, rec, f1))
        macro_f1 = float(np.mean([r[3] for r in rows]))
        return acc, macro_f1, rows

    for label, col in [("ORACLE (manual boxes, same classifier)", "pred_oracle"),
                        ("DETECTOR (real YOLO boxes, same classifier)", "pred_detector")]:
        acc, macro_f1, rows = score(col)
        print(f"\n=== {label} ===")
        print(f"accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")
        for cls_name, prec, rec, f1 in rows:
            print(f"  {cls_name}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

    # Error attribution, detector mode only
    d = out.copy()
    d["correct_det"] = d["true_stage"] == d["pred_detector"]
    errors = d[~d["correct_det"]].copy()

    def attribute(r):
        if r["branch_detector"] != "classifier":
            return "detection (wrong branch: single/no entity detected, should have been both)"
        det_ok_org = (r["iou_organism_vs_gt"] is not None) and (r["iou_organism_vs_gt"] >= IOU_OK)
        det_ok_exu = (r["iou_exuviae_vs_gt"] is not None) and (r["iou_exuviae_vs_gt"] >= IOU_OK)
        if det_ok_org and det_ok_exu:
            return "classification (both boxes accurate, XGBoost picked wrong stage)"
        return "detection (organism/exuvia box inaccurate, IoU < 0.5 vs ground truth)"

    if len(errors):
        errors["error_cause"] = errors.apply(attribute, axis=1)
        print(f"\n=== ERROR ATTRIBUTION, detector mode ({len(errors)} errors / {len(d)} images) ===")
        print(errors["error_cause"].value_counts())
    else:
        print("\nNo detector-mode errors.")


if __name__ == "__main__":
    main()
