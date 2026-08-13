"""Definitive oracle vs. detector-in-the-loop eval, on the grouped master split.

Both the classifier and YOLO are trained/evaluated on the same
observation_id partition (data/master_split.csv), so there is no
cross-contamination in either direction. Supersedes clean_end_to_end_eval.py
and yolo_unseen_eval.py once models/yolo_detect_GROUPED_SPLIT.pt exists
(run retrain_yolo_grouped.sh first).
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from ultralytics import YOLO
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
SPLIT_CSV = BASE_DIR / "data" / "master_split.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo_detect_GROUPED_SPLIT.pt"
OUT_CSV = BASE_DIR / "scripts" / "results" / "final_end_to_end_eval.csv"

SEED = 42
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 1024
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


def bucket(cls_id, names):
    label = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(names[int(cls_id)])
    return "exuviae" if "exuv" in label.lower() else "organism"


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def build_image_path(stage, filename):
    return IMAGE_ROOT / stage / filename


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

    yolo = YOLO(str(YOLO_MODEL_PATH))
    names = yolo.names if hasattr(yolo, "names") else yolo.model.names

    results = []
    for _, row in val_rows.iterrows():
        img_path = build_image_path(row["stage"], row["filename"])
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")

        oracle_org, oracle_exu = gt_box(row, "organism"), gt_box(row, "exuviae")
        pred_oracle, branch_oracle = decide(oracle_org, oracle_exu, clf, pil, row["taxon_group"])

        r = yolo.predict(pil, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
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
        pred_det, branch_det = decide(best["organism"], best["exuviae"], clf, pil, row["taxon_group"])

        iou_org = _iou_boxes(best["organism"], oracle_org) if (best["organism"] and oracle_org) else (0.0 if oracle_org else None)
        iou_exu = _iou_boxes(best["exuviae"], oracle_exu) if (best["exuviae"] and oracle_exu) else (0.0 if oracle_exu else None)

        results.append({
            "filename": row["filename"], "observation_id": row["observation_id"], "true_stage": row["stage"],
            "pred_oracle": pred_oracle, "branch_oracle": branch_oracle,
            "pred_detector": pred_det, "branch_detector": branch_det,
            "iou_organism_vs_gt": iou_org, "iou_exuviae_vs_gt": iou_exu,
        })

    out = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    for label, col in [("ORACLE", "pred_oracle"), ("DETECTOR", "pred_detector")]:
        acc, macro_f1, rows = score(out, col)
        print(f"\n=== {label} — n={len(out)} ===  accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")
        for c, p, r_, f1 in rows:
            print(f"  {c}: precision={p:.3f} recall={r_:.3f} f1={f1:.3f}")

    errors = out[out["true_stage"] != out["pred_detector"]].copy()

    def attribute(r):
        if r["branch_detector"] != "classifier":
            return "detection (wrong branch)"
        ok_org = r["iou_organism_vs_gt"] is not None and r["iou_organism_vs_gt"] >= IOU_OK
        ok_exu = r["iou_exuviae_vs_gt"] is not None and r["iou_exuviae_vs_gt"] >= IOU_OK
        return "classification (boxes accurate, wrong stage)" if (ok_org and ok_exu) else "detection (box inaccurate, IoU<0.5)"

    if len(errors):
        errors["error_cause"] = errors.apply(attribute, axis=1)
        print(f"\n=== ERROR ATTRIBUTION ({len(errors)}/{len(out)}) ===")
        print(errors["error_cause"].value_counts())


if __name__ == "__main__":
    main()
