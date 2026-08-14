"""Replaces the hardcoded rule_only_organism / rule_only_exuviae branches
(decide() always returning "post-moult" / "exuviae") with the XGBoost
classifier itself, made robust to partial detections via training-time
masking augmentation.

Why: final_end_to_end_eval.py / sahi_tiled_eval.py restrict the eval subset
to has_organism_box & has_exuviae_box & stage != pre-moult, so true_stage is
always "moulting" or "post-moult" -- the rule_only_exuviae branch predicting
literal "exuviae" is *guaranteed* wrong on this subset (0% accuracy is not a
bug, see decide() in the source scripts). rule_only_organism always
predicting "post-moult" is similarly a constant-output shortcut, not a
learned decision. Both branches together cover ~60% of images (71/125 with
SAHI) so this is the main remaining lever.

Training-time augmentation: every "both boxes" training row is cloned into
3 variants -- full (both boxes), organism-masked (exuviae features zeroed
out, organism_detected=0), exuviae-masked (organism features zeroed out,
exuviae_detected=0) -- with the same true label. This teaches the
classifier to decide from partial evidence instead of being evaluated
out-of-distribution (it would otherwise never see a masked-feature input at
train time even though real detector output is full of them).

no_detection is left as-is (no signal to route through a classifier without
building genuinely new global-image features -- out of scope for today).

Usage:
    cd vision
    python scripts/pipeline/unified_classifier_eval.py --model ../models/yolo_detect_GROUPED_SPLIT.pt --backend standard
    python scripts/pipeline/unified_classifier_eval.py --model ../models/yolo_detect_GROUPED_SPLIT.pt --backend sahi
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from xgboost import XGBClassifier
# ultralytics / sahi are imported lazily inside main(), depending on
# --backend, so the augmentation/feature logic can be exercised (e.g. in
# tests) without pulling in the heavier detection stack.

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
SPLIT_CSV = BASE_DIR / "data" / "master_split.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"

_cli = argparse.ArgumentParser()
_cli.add_argument("--model", default=str(BASE_DIR / "models" / "yolo_detect_GROUPED_SPLIT.pt"))
_cli.add_argument("--backend", choices=["standard", "sahi"], default="standard")
_cli.add_argument("--conf", type=float, default=0.35, help="Conf threshold (standard backend).")
_cli.add_argument("--iou", type=float, default=0.45, help="NMS IoU (standard backend).")
_cli.add_argument("--imgsz", type=int, default=1024, help="Inference imgsz (standard backend).")
_cli.add_argument("--slice", type=int, default=640, help="Slice height/width (sahi backend).")
_cli.add_argument("--overlap", type=float, default=0.2, help="Slice overlap (sahi backend).")
_cli.add_argument("--sahi_conf", type=float, default=0.25, help="Conf threshold (sahi backend).")
_cli.add_argument("--device", default="cpu", help="Device for sahi backend.")
_cli.add_argument("--mask_frac", type=float, default=1.0,
                   help="Fraction of train rows to also add as masked (organism-only / exuviae-only) variants.")
_cli.add_argument("--seed", type=int, default=42)
_args = _cli.parse_args()

YOLO_MODEL_PATH = Path(_args.model)
OUT_CSV = BASE_DIR / "scripts" / "results" / f"unified_classifier_eval_{YOLO_MODEL_PATH.stem}_{_args.backend}.csv"

SEED = _args.seed
IOU_CROSS_NMS = 0.50
IOU_OK = 0.5

# Base geometric/color features (same as final_end_to_end_eval.py) plus
# explicit missingness indicators so the model can condition on which box
# is actually present, not just infer it from sentinel values.
TRAIN_FEATURES = [
    "box_overlap", "dist_centroids", "x_organism", "y_organism",
    "x_exuviae", "y_exuviae", "h_exuviae",
    "org_mean_g", "org_mean_gray",
    "taxon_group_Crustacea", "taxon_group_Hexapoda",
    "taxon_group_Chelicerata", "taxon_group_Myriapoda",
    "organism_detected", "exuviae_detected",
]
VALID_TAXON_GROUPS = ["Chelicerata", "Crustacea", "Hexapoda", "Myriapoda"]
BEST_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=1.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=SEED, tree_method="hist", n_jobs=-1,
)


def _area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_boxes(b1, b2):
    if not b1 or not b2:
        return 0.0
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1, ix2, iy2 = max(x1, X1), max(y1, Y1), min(x2, X2), min(y2, Y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(*b1) + _area(*b2) - inter + 1e-9
    return float(inter / union)


def _filter_cross_overlap(dets, thr=0.5):
    orgs, exus = dets.get("organism", []), dets.get("exuviae", [])
    keep_orgs = [o for o in orgs if not any(_iou_boxes(o["box"], e["box"]) > thr and e["conf"] > o["conf"] for e in exus)]
    keep_exus = [e for e in exus if not any(_iou_boxes(e["box"], o["box"]) > thr and o["conf"] > e["conf"] for o in orgs)]
    dets["organism"], dets["exuviae"] = keep_orgs, keep_exus


def bucket(cls_id, names):
    label = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(names[int(cls_id)])
    return "exuviae" if "exuv" in label.lower() else "organism"


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def build_features(box_o, box_e, pil_img, taxon_group):
    """Same schema as final_end_to_end_eval.py's build_training_scale_features,
    plus explicit organism_detected / exuviae_detected flags."""
    f = {n: -1.0 for n in TRAIN_FEATURES}
    f["organism_detected"] = 1.0 if box_o else 0.0
    f["exuviae_detected"] = 1.0 if box_e else 0.0
    if box_o:
        f["x_organism"], f["y_organism"] = box_o[0], box_o[1]
    if box_e:
        f["x_exuviae"], f["y_exuviae"], f["h_exuviae"] = box_e[0], box_e[1], box_e[3] - box_e[1]
    f["box_overlap"] = _iou_boxes(box_o, box_e) if (box_o and box_e) else 0.0
    if box_o and box_e:
        cx_o, cy_o = (box_o[0] + box_o[2]) / 2, (box_o[1] + box_o[3]) / 2
        cx_e, cy_e = (box_e[0] + box_e[2]) / 2, (box_e[1] + box_e[3]) / 2
        f["dist_centroids"] = math.hypot(cx_o - cx_e, cy_o - cy_e)
    try:
        if box_o and pil_img is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box_o]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            if x2 > x1 and y2 > y1:
                crop = pil_img.crop((x1, y1, x2, y2))
                f["org_mean_g"] = ImageStat.Stat(crop).mean[1]
                f["org_mean_gray"] = ImageStat.Stat(crop.convert("L")).mean[0]
    except Exception:
        pass
    for g in VALID_TAXON_GROUPS:
        f[f"taxon_group_{g}"] = 1.0 if taxon_group == g else 0.0
    return {k: (-1.0 if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v) for k, v in f.items()}


def classify(clf, feat):
    X = pd.DataFrame([[float(feat.get(n, -1.0)) for n in TRAIN_FEATURES]], columns=TRAIN_FEATURES)
    return ["moulting", "post-moult"][int(clf.predict(X)[0])]


def decide(best_org, best_exu, clf, pil_img, taxon_group):
    """Unlike the rule-based decide(), always defers to the classifier as
    long as at least one box is present. Only true no-detection falls
    through with no prediction."""
    if best_org is None and best_exu is None:
        return None, "no_detection"
    branch = "classifier" if (best_org is not None and best_exu is not None) else (
        "classifier_organism_only" if best_org is not None else "classifier_exuviae_only")
    return classify(clf, build_features(best_org, best_exu, pil_img, taxon_group)), branch


def build_augmented_training_set(train_rows, rng):
    """For each ground-truth (both-boxes) training row, add masked variants
    so the classifier sees organism-only and exuviae-only inputs at train
    time, not just at eval time."""
    X_rows, y_rows = [], []
    classes = ["moulting", "post-moult"]
    for _, row in train_rows.iterrows():
        box_o, box_e = gt_box(row, "organism"), gt_box(row, "exuviae")
        pil = None
        img_path = IMAGE_ROOT / row["stage"] / row["filename"]
        if img_path.exists():
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                pil = None
        label = classes.index(row["stage"])

        # full (both boxes) -- always included
        X_rows.append(build_features(box_o, box_e, pil, row["taxon_group"]))
        y_rows.append(label)

        if rng.random() < _args.mask_frac:
            # organism-masked (simulates "only exuviae detected")
            X_rows.append(build_features(None, box_e, pil, row["taxon_group"]))
            y_rows.append(label)
            # exuviae-masked (simulates "only organism detected")
            X_rows.append(build_features(box_o, None, pil, row["taxon_group"]))
            y_rows.append(label)

    X = pd.DataFrame(X_rows, columns=TRAIN_FEATURES)
    y = np.array(y_rows)
    return X, y


def detect_standard(yolo, names, pil):
    r = yolo.predict(pil, imgsz=_args.imgsz, conf=_args.conf, iou=_args.iou, verbose=False)[0]
    dets = {"organism": [], "exuviae": []}
    if r.boxes is not None:
        for (x1, y1, x2, y2), c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            dets[bucket(c, names)].append({"box": [float(x1), float(y1), float(x2), float(y2)], "conf": float(p)})
    _filter_cross_overlap(dets, IOU_CROSS_NMS)
    best, best_conf = {"organism": None, "exuviae": None}, {"organism": -1.0, "exuviae": -1.0}
    for k in best:
        for d in dets[k]:
            if d["conf"] > best_conf[k]:
                best_conf[k], best[k] = d["conf"], d["box"]
    return best["organism"], best["exuviae"]


def detect_sahi(detection_model, img_path):
    from sahi.predict import get_sliced_prediction
    r = get_sliced_prediction(
        str(img_path), detection_model,
        slice_height=_args.slice, slice_width=_args.slice,
        overlap_height_ratio=_args.overlap, overlap_width_ratio=_args.overlap,
        verbose=0,
    )
    dets = {"organism": [], "exuviae": []}
    for p in r.object_prediction_list:
        name = "exuviae" if "exuv" in p.category.name.lower() else "organism"
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        dets[name].append({"box": [float(x1), float(y1), float(x2), float(y2)], "conf": float(p.score.value)})
    _filter_cross_overlap(dets, IOU_CROSS_NMS)
    best, best_conf = {"organism": None, "exuviae": None}, {"organism": -1.0, "exuviae": -1.0}
    for k in best:
        for d in dets[k]:
            if d["conf"] > best_conf[k]:
                best_conf[k], best[k] = d["conf"], d["box"]
    return best["organism"], best["exuviae"]


def score(df, pred_col):
    d = df.copy()
    acc = (d["true_stage"] == d[pred_col]).mean()
    rows = []
    for c in ["moulting", "post-moult"]:
        tp = ((d["true_stage"] == c) & (d[pred_col] == c)).sum()
        fn = ((d["true_stage"] == c) & (d[pred_col] != c)).sum()
        fp = ((d["true_stage"] != c) & (d[pred_col] == c)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append((c, p, r, f1))
    return acc, float(np.mean([r[3] for r in rows])), rows


def main():
    print(f"detector: {YOLO_MODEL_PATH.name}  backend={_args.backend}")
    df = pd.read_csv(FEATURES_CSV)
    split = pd.read_csv(SPLIT_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")].copy()
    both = both.merge(split, on="observation_id", how="inner").reset_index(drop=True)

    train_rows = both[both["split"] == "train"].reset_index(drop=True)
    val_rows = both[both["split"] == "val"].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    X_train, y_train = build_augmented_training_set(train_rows, rng)
    print(f"classifier train_n={len(train_rows)} (base) -> {len(X_train)} (augmented)  val_n={len(val_rows)}")

    clf = XGBClassifier(**BEST_PARAMS)
    clf.fit(X_train, y_train)

    results = []
    if _args.backend == "standard":
        from ultralytics import YOLO
        yolo = YOLO(str(YOLO_MODEL_PATH))
        names = yolo.names if hasattr(yolo, "names") else yolo.model.names
        for _, row in val_rows.iterrows():
            img_path = IMAGE_ROOT / row["stage"] / row["filename"]
            if not img_path.exists():
                continue
            pil = Image.open(img_path).convert("RGB")
            best_org, best_exu = detect_standard(yolo, names, pil)
            pred_det, branch_det = decide(best_org, best_exu, clf, pil, row["taxon_group"])
            results.append({"filename": row["filename"], "true_stage": row["stage"],
                             "pred_detector": pred_det, "branch_detector": branch_det})
    else:
        from sahi import AutoDetectionModel
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(YOLO_MODEL_PATH),
            confidence_threshold=_args.sahi_conf, device=_args.device,
        )
        for _, row in val_rows.iterrows():
            img_path = IMAGE_ROOT / row["stage"] / row["filename"]
            if not img_path.exists():
                continue
            pil = Image.open(img_path).convert("RGB")
            best_org, best_exu = detect_sahi(detection_model, img_path)
            pred_det, branch_det = decide(best_org, best_exu, clf, pil, row["taxon_group"])
            results.append({"filename": row["filename"], "true_stage": row["stage"],
                             "pred_detector": pred_det, "branch_detector": branch_det})

    out = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    acc, macro_f1, rows = score(out, "pred_detector")
    print(f"\n=== UNIFIED CLASSIFIER DETECTOR ({_args.backend}) — n={len(out)} ===  accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")
    for c, p, r, f1 in rows:
        print(f"  {c}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

    print("\n=== branch distribution + per-branch accuracy ===")
    d = out.copy()
    d["correct"] = d["true_stage"] == d["pred_detector"]
    print(d.groupby("branch_detector").agg(n=("correct", "size"), accuracy=("correct", "mean")).round(3))
    print("\nCompare directly against final_end_to_end_eval.py / sahi_tiled_eval.py's DETECTOR result for the same --model.")


if __name__ == "__main__":
    main()
