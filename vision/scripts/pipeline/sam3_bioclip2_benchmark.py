"""Benchmark only, not production: SAM 3.1 (text-prompt concept segmentation)
+ BioCLIP 2 embeddings + lightweight classifier, evaluated on the SAME
master_split.csv val set as final_end_to_end_eval.py for a direct,
apples-to-apples comparison against the YOLO+XGBoost pipeline.

YOLO stays the production detector (fast, cheap, CPU-capable). This script
answers one question: would a foundation-model pipeline do meaningfully
better on detection, at ~230x the latency and ~580x the size?

Prerequisites (must be done manually, not scriptable):
1. Request access to SAM 3 weights: https://huggingface.co/facebook/sam3
   Download sam3.pt once approved, place at SAM3_WEIGHTS below.
2. pip install -U ultralytics open_clip_torch  (ultralytics >= 8.3.237)

Note: SAM 3's ultralytics integration is very recent (Mar 2026). The exact
Results object schema (.boxes vs .masks-only) hasn't been hand-verified
against this exact ultralytics version -- the script prints the raw result
for the first image so you can patch field names fast if something's off.
"""

import math
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
SPLIT_CSV = BASE_DIR / "data" / "master_split.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"
SAM3_WEIGHTS = BASE_DIR / "models" / "sam3.pt"
OUT_CSV = BASE_DIR / "scripts" / "results" / "sam3_bioclip2_benchmark.csv"

SEED = 42
IOU_OK = 0.5
PCA_DIM = 16

TEXT_ORGANISM = "live arthropod organism, insect or crustacean or arachnid"
TEXT_EXUVIAE = "shed arthropod exoskeleton, molted exuvia, empty exuviae"

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def gt_box(row, prefix):
    x, y, w, h = row.get(f"x_{prefix}"), row.get(f"y_{prefix}"), row.get(f"w_{prefix}"), row.get(f"h_{prefix}")
    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def build_image_path(stage, filename):
    return IMAGE_ROOT / stage / filename


def sam3_best_box(predictor, pil_img, text_query, debug=False):
    """Runs one concept query, returns the highest-confidence box or None."""
    predictor.set_image(np.array(pil_img))
    r = predictor(text=[text_query])
    if debug:
        print("RAW SAM3 RESULT (patch field names below if this doesn't match):", r)
    res = r[0] if isinstance(r, (list, tuple)) else r
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    best = xyxy[conf.argmax()]
    return [float(v) for v in best]


def bioclip_embed(model, preprocess, pil_img, box):
    if box is None:
        return np.zeros(768, dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(768, dtype=np.float32)
    crop = pil_img.crop((x1, y1, x2, y2))
    with torch.no_grad():
        t = preprocess(crop).unsqueeze(0).to(device)
        emb = model.encode_image(t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


def shape_features(box_o, box_e):
    f = {"box_overlap": 0.0, "dist_centroids": -1.0, "has_o": float(box_o is not None), "has_e": float(box_e is not None)}
    if box_o and box_e:
        f["box_overlap"] = _iou_boxes(box_o, box_e)
        cx_o, cy_o = (box_o[0] + box_o[2]) / 2, (box_o[1] + box_o[3]) / 2
        cx_e, cy_e = (box_e[0] + box_e[2]) / 2, (box_e[1] + box_e[3]) / 2
        f["dist_centroids"] = math.hypot(cx_o - cx_e, cy_o - cy_e)
    return f


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
    if not SAM3_WEIGHTS.exists():
        raise SystemExit(f"Missing {SAM3_WEIGHTS} -- request access at huggingface.co/facebook/sam3 first.")

    df = pd.read_csv(FEATURES_CSV)
    split = pd.read_csv(SPLIT_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")].copy()
    both = both.merge(split, on="observation_id", how="inner").reset_index(drop=True)

    print("[1/3] Loading SAM 3.1 and BioCLIP 2...")
    predictor = SAM3SemanticPredictor(overrides={"conf": 0.25, "task": "segment", "mode": "predict",
                                                  "model": str(SAM3_WEIGHTS), "quantize": 16, "verbose": False})
    bio_model, _, bio_preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bio_model = bio_model.to(device).eval()

    print("[2/3] Running SAM3 + BioCLIP2 on train (for classifier fit) and val (for eval)...")
    rows_all = []
    for i, row in both.iterrows():
        img_path = build_image_path(row["stage"], row["filename"])
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")

        oracle_org, oracle_exu = gt_box(row, "organism"), gt_box(row, "exuviae")
        det_org = sam3_best_box(predictor, pil, TEXT_ORGANISM, debug=(i == 0))
        det_exu = sam3_best_box(predictor, pil, TEXT_EXUVIAE, debug=(i == 0))

        emb_o_oracle = bioclip_embed(bio_model, bio_preprocess, pil, oracle_org)
        emb_e_oracle = bioclip_embed(bio_model, bio_preprocess, pil, oracle_exu)
        emb_o_det = bioclip_embed(bio_model, bio_preprocess, pil, det_org)
        emb_e_det = bioclip_embed(bio_model, bio_preprocess, pil, det_exu)

        iou_org = _iou_boxes(det_org, oracle_org) if (det_org and oracle_org) else (0.0 if oracle_org else None)
        iou_exu = _iou_boxes(det_exu, oracle_exu) if (det_exu and oracle_exu) else (0.0 if oracle_exu else None)

        rows_all.append({
            "split": row["split"], "true_stage": row["stage"], "filename": row["filename"],
            "emb_o_oracle": emb_o_oracle, "emb_e_oracle": emb_e_oracle,
            "emb_o_det": emb_o_det, "emb_e_det": emb_e_det,
            "shape_oracle": shape_features(oracle_org, oracle_exu),
            "shape_det": shape_features(det_org, det_exu),
            "iou_organism_vs_gt": iou_org, "iou_exuviae_vs_gt": iou_exu,
            "det_branch": "classifier" if (det_org and det_exu) else ("rule_only_exuviae" if det_exu else
                          ("rule_only_organism" if det_org else "no_detection")),
        })

    def build_X(rows, emb_o_key, emb_e_key, shape_key, pca_o=None, pca_e=None):
        Eo = np.stack([r[emb_o_key] for r in rows])
        Ee = np.stack([r[emb_e_key] for r in rows])
        if pca_o is None:
            pca_o = PCA(n_components=PCA_DIM, random_state=SEED).fit(Eo)
            pca_e = PCA(n_components=PCA_DIM, random_state=SEED).fit(Ee)
        Eo_r, Ee_r = pca_o.transform(Eo), pca_e.transform(Ee)
        S = np.array([[r[shape_key]["box_overlap"], r[shape_key]["dist_centroids"],
                       r[shape_key]["has_o"], r[shape_key]["has_e"]] for r in rows])
        return np.hstack([Eo_r, Ee_r, S]), pca_o, pca_e

    train_rows = [r for r in rows_all if r["split"] == "train"]
    val_rows = [r for r in rows_all if r["split"] == "val"]
    classes = ["moulting", "post-moult"]
    y_train = np.array([classes.index(r["true_stage"]) for r in train_rows])

    print("[3/3] Fitting PCA + logistic regression on oracle features, scoring oracle vs detector...")
    X_train, pca_o, pca_e = build_X(train_rows, "emb_o_oracle", "emb_e_oracle", "shape_oracle")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)

    X_val_oracle, _, _ = build_X(val_rows, "emb_o_oracle", "emb_e_oracle", "shape_oracle", pca_o, pca_e)
    X_val_det, _, _ = build_X(val_rows, "emb_o_det", "emb_e_det", "shape_det", pca_o, pca_e)
    pred_oracle_idx = clf.predict(X_val_oracle)
    pred_det_idx = clf.predict(X_val_det)

    out = pd.DataFrame({
        "filename": [r["filename"] for r in val_rows],
        "true_stage": [r["true_stage"] for r in val_rows],
        "pred_oracle": [classes[i] for i in pred_oracle_idx],
        "pred_detector": [classes[i] if r["det_branch"] == "classifier" else
                           ("exuviae" if r["det_branch"] == "rule_only_exuviae" else
                            "post-moult" if r["det_branch"] == "rule_only_organism" else "none")
                           for i, r in zip(pred_det_idx, val_rows)],
        "det_branch": [r["det_branch"] for r in val_rows],
        "iou_organism_vs_gt": [r["iou_organism_vs_gt"] for r in val_rows],
        "iou_exuviae_vs_gt": [r["iou_exuviae_vs_gt"] for r in val_rows],
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    for label, col in [("ORACLE (GT boxes, BioCLIP2+shape features)", "pred_oracle"),
                        ("DETECTOR (SAM3.1 text-prompt boxes)", "pred_detector")]:
        acc, macro_f1, rows = score(out, col)
        print(f"\n=== {label} — n={len(out)} ===  accuracy={acc:.3f}  macro_f1={macro_f1:.3f}")
        for c, p, r_, f1 in rows:
            print(f"  {c}: precision={p:.3f} recall={r_:.3f} f1={f1:.3f}")

    errors = out[out["true_stage"] != out["pred_detector"]].copy()

    def attribute(r):
        if r["det_branch"] != "classifier":
            return "detection (wrong branch)"
        ok_org = r["iou_organism_vs_gt"] is not None and r["iou_organism_vs_gt"] >= IOU_OK
        ok_exu = r["iou_exuviae_vs_gt"] is not None and r["iou_exuviae_vs_gt"] >= IOU_OK
        return "classification (boxes accurate, wrong stage)" if (ok_org and ok_exu) else "detection (box inaccurate, IoU<0.5)"

    if len(errors):
        errors["error_cause"] = errors.apply(attribute, axis=1)
        print(f"\n=== ERROR ATTRIBUTION ({len(errors)}/{len(out)}) ===")
        print(errors["error_cause"].value_counts())

    print("\nCompare directly against final_end_to_end_eval.py's output -- same val split, same scoring.")


if __name__ == "__main__":
    main()
