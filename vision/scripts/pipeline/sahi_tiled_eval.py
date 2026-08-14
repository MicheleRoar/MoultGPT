"""Same oracle/detector eval as final_end_to_end_eval.py, but the detector
branch runs SAHI sliced inference instead of a single full-image forward
pass. No retraining -- reuses whatever YOLO checkpoint you already have.
Tests the "small/close objects get lost at one shot" hypothesis directly.

pip install -U sahi

Usage:
    cd vision
    python scripts/pipeline/sahi_tiled_eval.py --model ../models/yolo_detect_GROUPED_SPLIT.pt
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
SPLIT_CSV = BASE_DIR / "data" / "master_split.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"

_cli = argparse.ArgumentParser()
_cli.add_argument("--model", default=str(BASE_DIR / "models" / "yolo_detect_GROUPED_SPLIT.pt"))
_cli.add_argument("--slice", type=int, default=640, help="Slice height/width.")
_cli.add_argument("--overlap", type=float, default=0.2)
_cli.add_argument("--conf", type=float, default=0.25)
_cli.add_argument("--device", default="cpu")
_args = _cli.parse_args()

YOLO_MODEL_PATH = Path(_args.model)
OUT_CSV = BASE_DIR / "scripts" / "results" / f"sahi_tiled_eval_{YOLO_MODEL_PATH.stem}.csv"

SEED = 42
IOU_CROSS_NMS = 0.50
IOU_OK = 0.5

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


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def build_training_scale_features(box_o, box_e, pil_img, taxon_group):
    f = {n: -1.0 for n in TRAIN_FEATURES}
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
        if box_o:
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
    if best_exu is not None and best_org is None:
        return "exuviae", "rule_only_exuviae"
    if best_org is not None and best_exu is None:
        return "post-moult", "rule_only_organism"
    if best_org is not None and best_exu is not None:
        return classify(clf, build_training_scale_features(best_org, best_exu, pil_img, taxon_group)), "classifier"
    return None, "no_detection"


def sahi_predict(detection_model, img_path):
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
    print(f"detector: {YOLO_MODEL_PATH.name}  slice={_args.slice}  overlap={_args.overlap}  conf={_args.conf}")
    df = pd.read_csv(FEATURES_CSV)
    split = pd.read_csv(SPLIT_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")].copy()
    both = both.merge(split, on="observation_id", how="inner").reset_index(drop=True)

    train_rows = both[both["split"] == "train"].reset_index(drop=True)
    val_rows = both[both["split"] == "val"].reset_index(drop=True)
    print(f"classifier train_n={len(train_rows)} val_n={len(val_rows)}")

    classes = ["moulting", "post-moult"]
    X_train = train_rows[TRAIN_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(-1)
    y_train = np.array([classes.index(v) for v in train_rows["stage"]])
    clf = XGBClassifier(**BEST_PARAMS)
    clf.fit(X_train, y_train)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(YOLO_MODEL_PATH),
        confidence_threshold=_args.conf, device=_args.device,
    )

    results = []
    for _, row in val_rows.iterrows():
        img_path = IMAGE_ROOT / row["stage"] / row["filename"]
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")
        best_org, best_exu = sahi_predict(detection_model, img_path)
        pred_det, branch_det = decide(best_org, best_exu, clf, pil, row["taxon_group"])
        results.append({"filename": row["filename"], "true_stage": row["stage"],
                         "pred_detector": pred_det, "branch_detector": branch_det})

    out = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    acc, macro_f1, rows = score(out, "pred_detector")
    print(f"\n=== SAHI DETECTOR — n={len(out)} ===  accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")
    for c, p, r, f1 in rows:
        print(f"  {c}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    print(out["branch_detector"].value_counts())
    print("\nCompare directly against final_end_to_end_eval.py's DETECTOR result for the same --model.")


if __name__ == "__main__":
    main()
