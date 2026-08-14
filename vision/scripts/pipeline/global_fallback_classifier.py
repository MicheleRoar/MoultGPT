"""Global image fallback classifier for the no_detection branch: when YOLO
finds neither organism nor exuviae box, this predicts moulting/post-moult
purely from whole-image statistics (color, HSV, contrast, edge density,
entropy) plus taxon group -- no box-derived features needed.

Validated via grouped 5-fold CV on the SAME 499 training observations used
everywhere else (StratifiedGroupKFold on observation_id) BEFORE ever being
applied to the 10 no_detection val images. This is a deliberate methodology
choice: no threshold/config decision here is allowed to look at the final
held-out 125 images, per the "no astrologia statistica" requirement -- the
printed out-of-fold numbers are the only thing that should determine
whether this fallback is worth wiring into unified_classifier_eval.py at
all.

Usage:
    cd vision
    python scripts/pipeline/global_fallback_classifier.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "annotated_features.csv"
SPLIT_CSV = BASE_DIR / "data" / "master_split.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"
MODEL_OUT = BASE_DIR / "models" / "global_fallback_classifier.pkl"

SEED = 42
VALID_TAXON_GROUPS = ["Chelicerata", "Crustacea", "Hexapoda", "Myriapoda"]
GLOBAL_FEATURES = [
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
    "gray_var", "edge_density", "entropy",
    "taxon_group_Crustacea", "taxon_group_Hexapoda",
    "taxon_group_Chelicerata", "taxon_group_Myriapoda",
]
BASE_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=1.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=SEED, tree_method="hist", n_jobs=-1,
)

_cli = argparse.ArgumentParser()
_cli.add_argument("--resize", type=int, default=256, help="Whole image resized to NxN before feature extraction.")
_cli.add_argument("--n_splits", type=int, default=5)
_args = _cli.parse_args()


def compute_global_features(img_path, taxon_group):
    pil = Image.open(img_path).convert("RGB").resize((_args.resize, _args.resize))
    arr = np.asarray(pil, dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    hsv = np.asarray(pil.convert("HSV"), dtype=np.float64)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    gray_img = pil.convert("L")
    gray = np.asarray(gray_img, dtype=np.float64)
    edges = np.asarray(gray_img.filter(ImageFilter.FIND_EDGES), dtype=np.float64)

    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0

    f = {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "std_r": float(r.std()), "std_g": float(g.std()), "std_b": float(b.std()),
        "mean_h": float(h.mean()), "mean_s": float(s.mean()), "mean_v": float(v.mean()),
        "std_h": float(h.std()), "std_s": float(s.std()), "std_v": float(v.std()),
        "gray_var": float(gray.var()), "edge_density": float(edges.mean() / 255.0),
        "entropy": entropy,
    }
    for g_ in VALID_TAXON_GROUPS:
        f[f"taxon_group_{g_}"] = 1.0 if taxon_group == g_ else 0.0
    return f


def build_feature_table(rows):
    classes = ["moulting", "post-moult"]
    feats, labels, groups = [], [], []
    for _, row in rows.iterrows():
        img_path = IMAGE_ROOT / row["stage"] / row["filename"]
        if not img_path.exists():
            continue
        feats.append(compute_global_features(img_path, row["taxon_group"]))
        labels.append(classes.index(row["stage"]))
        groups.append(row["observation_id"])
    X = pd.DataFrame(feats, columns=GLOBAL_FEATURES)
    return X, np.array(labels), np.array(groups)


def score(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    f1s = []
    for c in (0, 1):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    return float(acc), float(np.mean(f1s))


def main():
    df = pd.read_csv(FEATURES_CSV)
    split = pd.read_csv(SPLIT_CSV)
    both = df[(df["has_organism_box"] == 1) & (df["has_exuviae_box"] == 1) & (df["stage"] != "pre-moult")].copy()
    both = both.merge(split, on="observation_id", how="inner").reset_index(drop=True)
    train_rows = both[both["split"] == "train"].reset_index(drop=True)
    print(f"train_n={len(train_rows)} (grouped CV only -- never touches the 125 val images)")

    X, y, groups = build_feature_table(train_rows)
    print(f"computed global features for {len(X)} images")

    n_moulting, n_postmoult = int((y == 0).sum()), int((y == 1).sum())
    spw = n_moulting / max(n_postmoult, 1)
    majority_baseline = max(n_moulting, n_postmoult) / len(y)
    print(f"class balance: moulting={n_moulting} post-moult={n_postmoult}  "
          f"majority-class baseline accuracy={majority_baseline:.3f}")

    cv = StratifiedGroupKFold(n_splits=_args.n_splits, shuffle=True, random_state=SEED)
    fold_acc, fold_f1 = [], []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        clf = XGBClassifier(**{**BASE_PARAMS, "scale_pos_weight": spw})
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        pred = clf.predict(X.iloc[te_idx])
        acc, f1 = score(y[te_idx], pred)
        fold_acc.append(acc)
        fold_f1.append(f1)
        print(f"  fold {fold}: n_test={len(te_idx)} accuracy={acc:.3f} macro_f1={f1:.3f}")

    mean_acc, mean_f1 = float(np.mean(fold_acc)), float(np.mean(fold_f1))
    print(f"\n=== GROUPED CV (out-of-fold, train only) ===")
    print(f"mean accuracy={mean_acc:.3f} (+/-{np.std(fold_acc):.3f})  mean macro_f1={mean_f1:.3f} (+/-{np.std(fold_f1):.3f})")
    print(f"majority-class baseline accuracy={majority_baseline:.3f}")

    if mean_f1 <= 0.55:
        print("\nWARNING: out-of-fold macro F1 is close to/below a weak baseline -- "
              "the global fallback likely isn't worth deploying as-is. Do NOT wire it into "
              "unified_classifier_eval.py based on this signal; consider it not validated.")
    else:
        print("\nOut-of-fold signal looks real -- training final model on all train_rows and saving.")

    clf_final = XGBClassifier(**{**BASE_PARAMS, "scale_pos_weight": spw})
    clf_final.fit(X, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as fh:
        pickle.dump({"model": clf_final, "features": GLOBAL_FEATURES, "resize": _args.resize}, fh)
    print(f"saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
