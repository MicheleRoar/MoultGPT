#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark detectors for moult (organism vs exuviae)

- Unified small-object recipe for multiple backbones
- Dual validation: (1) baseline Ultralytics, (2) top-1-per-class filter
- Saves per-model metrics/plots and a cross-model summary

USAGE (from repo root CNN/):
    python scripts/training/benchmark_models.py
"""

import os
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ========= PROJECT PATHS =========
THIS_FILE = Path(__file__).resolve()
BASE_DIR = (
    THIS_FILE.parents[2]
    if (THIS_FILE.parents[2] / "data" / "yolo" / "moulting.yaml").exists()
    else THIS_FILE.parents[0]
)
# Hardening: esegui dal repo root
os.chdir(BASE_DIR)

DATA_YAML  = BASE_DIR / "data" / "yolo" / "moulting.yaml"
RUNS_DIR   = BASE_DIR / "scripts" / "training" / "runs" / "moult_bench"
PLOTS_DIR  = BASE_DIR / "scripts" / "results" / "plots"
RESULTS_DIR = BASE_DIR / "scripts" / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ========= ULTRALYTICS SETTINGS HARDENING =========
# Se Ultralytics dovesse scaricare qualcosa, fallo finire in ./models e non nel CWD.
try:
    # 8.3.x
    from ultralytics.utils import SETTINGS as USET
    USET.update({
        'weights_dir': str(MODELS_DIR),
        'runs_dir': str(RUNS_DIR),
        # opzionale: evita check rumorosi/auto-magic
        # 'checks': False,
    })
except Exception:
    # fallback env
    os.environ.setdefault('ULTRALYTICS_WEIGHTS_DIR', str(MODELS_DIR))
    os.environ.setdefault('ULTRALYTICS_RUNS_DIR', str(RUNS_DIR))

# ========= GLOBAL CONFIG =========
EPOCHS = 300
BATCH  = -1
IMGSZ  = 1280
DEVICE = "0"
SEED   = 42

# Validation thresholds
VAL_CONF   = 0.25
VAL_IOU    = 0.60
VAL_MAXDET = 2
VAL_TTA    = False

# Candidate pretrained backbones (nomi alias noti)
CANDIDATES = [
    "yolo11n.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov10x.pt",
    "yolov8n.pt",
]
ALIAS_NAMES = set(CANDIDATES)

# Solo qui cerchiamo i pesi
MODEL_DIRS = [
    MODELS_DIR,            # first (obbligatorio)
    BASE_DIR,              # optional
    BASE_DIR / "scripts" / "training",  # last resort (dovrebbe essere vuoto)
]

# ========= TASK-AWARE RECIPE =========
TRAIN_ARGS = dict(
    optimizer="AdamW",
    cos_lr=True,
    weight_decay=0.0005,
    mosaic=0.50,
    mixup=0.05,
    copy_paste=0.05,
    close_mosaic=20,
    rect=False,
    hsv_h=0.015,
    hsv_s=0.60,
    hsv_v=0.20,
    translate=0.12,
    scale=0.60,
    degrees=0.0,
    shear=0.0,
    label_smoothing=0.005,
    box=7.5,
    cls=0.30,
    dfl=2.0,
    cache=True,
    deterministic=True,
    patience=60,
    workers=8,
    plots=True,
    warmup_epochs=5,
    lr0=0.0025,
    lrf=0.01,
)

# ========= SAFETY OVERRIDES =========
PER_MODEL_OVERRIDES: Dict[str, Dict[str, int]] = {
    "yolo11l.pt": {"imgsz": 1024, "batch": -1},
    "yolo11x.pt": {"imgsz": 1024, "batch": -1},
    "yolov10x.pt": {"imgsz": 1024, "batch": -1},
}

# ========= UTILS =========
def find_model_path(name: str) -> Optional[Path]:
    """Resolve weights strictly from MODEL_DIRS (no bare CWD)."""
    for root in MODEL_DIRS:
        cand = (root / name)
        if cand.exists():
            r = cand.resolve()
            print(f"[RESOLVE] {name} -> {r}")
            return r
    print(f"[MISS] {name} not found in: {', '.join(map(str, MODEL_DIRS))}")
    return None

def localize_to_non_alias(p: Path) -> Path:
    """
    Se il basename è un alias (es. 'yolo11n.pt'), crea/usa una
    copia 'local_<alias>.pt' in MODELS_DIR per evitare il code-path di auto-download.
    """
    if p.name in ALIAS_NAMES:
        tgt = MODELS_DIR / f"local_{p.name}"
        if not tgt.exists():
            shutil.copy2(p, tgt)
            print(f"[LOCALIZE] Copied alias weight → {tgt.name}")
        return tgt.resolve()
    return p

def run_cmd(cmd: List[str], fail_msg: str):
    print("\n=== RUN:", " ".join(map(str, cmd)), "\n")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(fail_msg)

def run_dir_from_best(weights: Path) -> Path:
    # /.../moult_bench/<run>/weights/best.pt -> <run>
    return weights.parent.parent

# ========= TRAIN / VAL =========
def yolo_train(backbone_path: Path, run_name: str) -> Path:
    """Train one model with the unified recipe + hardware safety overrides."""
    run_dir = RUNS_DIR / run_name
    run_dir.parent.mkdir(parents=True, exist_ok=True)

    ovr = PER_MODEL_OVERRIDES.get(backbone_path.name, {})
    imgsz = int(ovr.get("imgsz", IMGSZ))
    batch = int(ovr.get("batch", BATCH))

    # guard: require local file to exist (prevent auto-download)
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone missing: {backbone_path} (expected under ./models)")

    cmd = [
        "yolo", "task=detect", "mode=train",
        f"model={str(backbone_path)}",
        f"data={str(DATA_YAML)}",
        f"epochs={EPOCHS}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={DEVICE}",
        f"project={str(RUNS_DIR)}",
        f"name={run_name}",
        f"seed={SEED}",
        "verbose=True",
    ]
    for k, v in TRAIN_ARGS.items():
        cmd.append(f"{k}={v}")

    run_cmd(cmd, f"Training failed for {backbone_path.name}")
    return run_dir

def yolo_val_cli(weights: Path, desc: str, imgsz: int):
    """Baseline validation via CLI; force project/name to avoid runs/detect/*."""
    run_dir = run_dir_from_best(weights)
    args = [
        "yolo", "task=detect", "mode=val",
        f"model={str(weights)}",
        f"data={str(DATA_YAML)}",
        f"imgsz={imgsz}",
        f"device={DEVICE}",
        "plots=True",
        f"project={str(RUNS_DIR)}",
        f"name={run_dir.name}",
        "exist_ok=True",
        f"conf={VAL_CONF}",
        f"iou={VAL_IOU}",
        f"max_det={VAL_MAXDET}",
        "save_json=False",
        "save_hybrid=False",
        "rect=False",
    ]
    if VAL_TTA:
        args.append("augment=True")
    print(f"[INFO] Baseline validation ({desc}) -> {RUNS_DIR / run_dir.name}")
    run_cmd(args, f"Validation failed for {weights}")

def keep_top1_per_class(result, classes):
    """Filter: keep at most 1 box per class (highest confidence)."""
    b = result.boxes
    if b is None or len(b) == 0:
        return result
    import numpy as np
    cls = b.cls.detach().cpu().numpy()
    conf = b.conf.detach().cpu().numpy()
    keep_idx = []
    for c in classes:
        idx = np.where(cls == c)[0]
        if len(idx):
            best = idx[conf[idx].argmax()]
            keep_idx.append(int(best))
    result.boxes = b[keep_idx] if keep_idx else b[:0]
    return result

def register_filter_callback(model, classes):
    """Callback compatible with multiple Ultralytics versions."""
    def on_val_batch_end(*args, **kwargs):
        if len(args) == 6:
            trainer, batch, batch_idx, preds, losses, outputs = args
            for i, r in enumerate(outputs):
                outputs[i] = keep_top1_per_class(r, classes)
        elif len(args) == 1:
            trainer = args[0]
            if hasattr(trainer, "results"):
                for i, r in enumerate(trainer.results):
                    trainer.results[i] = keep_top1_per_class(r, classes)
    model.add_callback("on_val_batch_end", on_val_batch_end)

def yolo_val_filtered(weights: Path) -> Dict[str, float]:
    """Validation via Python API with the top-1-per-class filter. Returns key metrics."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    register_filter_callback(model, classes=[0, 1])
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        conf=VAL_CONF,
        iou=VAL_IOU,
        max_det=VAL_MAXDET,
        plots=True,
        augment=VAL_TTA,
        verbose=False,
    )
    rd = getattr(metrics, "results_dict", None)
    if rd is None:
        rd = {
            "metrics/precision(B)": getattr(metrics.box, "p", None),
            "metrics/recall(B)": getattr(metrics.box, "r", None),
            "metrics/mAP50(B)": getattr(metrics.box, "map50", None),
            "metrics/mAP50-95(B)": getattr(metrics.box, "map", None),
        }
    return {
        "precision": float(rd.get("metrics/precision(B)", 0.0) or 0.0),
        "recall": float(rd.get("metrics/recall(B)", 0.0) or 0.0),
        "map50": float(rd.get("metrics/mAP50(B)", 0.0) or 0.0),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0) or 0.0),
    }

# ========= IO / PLOTTING =========
def read_last_results_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"[SKIP] results.csv not found: {csv_path}")
        return {}
    last = None
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last or {}

def safe_get_float(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except ValueError:
                pass
    return default

def plot_individual_curves(model_name: str, results_csv: Path, out_prefix: Path):
    if not results_csv.exists():
        print(f"[SKIP] No results.csv for {model_name}")
        return
    df = pd.read_csv(results_csv)

    # Losses
    plt.figure()
    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        if col in df.columns:
            plt.plot(df.index + 1, df[col], label=col)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Losses per epoch — {model_name}")
    plt.tight_layout(); plt.savefig(str(out_prefix) + "_losses.png"); plt.close()

    # Metrics
    plt.figure()
    for col in ["metrics/precision(B)", "metrics/recall(B)",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if col in df.columns:
            plt.plot(df.index + 1, df[col], label=col)
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend()
    plt.title(f"Metrics per epoch — {model_name}")
    plt.tight_layout(); plt.savefig(str(out_prefix) + "_metrics.png"); plt.close()

def plot_multimodel_curves(model2csv: Dict[str, Path], metric_col: str, out_path: Path, title: str):
    plt.figure(); plotted = False
    for name, csv_path in model2csv.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if metric_col in df.columns:
            plt.plot(df.index + 1, df[metric_col], label=name); plotted = True
    if not plotted:
        print(f"[WARN] No curves available for {metric_col}"); plt.close(); return
    plt.xlabel("Epoch"); plt.ylabel(metric_col); plt.legend()
    plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()

def copy_ultralytics_plots(run_dir: Path, model_tag: str):
    cand_dirs = list(run_dir.glob("**/val"))
    if not cand_dirs: return
    val_dir = cand_dirs[-1]
    for png in val_dir.glob("*.png"):
        dst = PLOTS_DIR / f"{model_tag}__{png.name}"
        try: shutil.copy2(png, dst)
        except Exception: pass

def barplot_baseline_vs_filtered(model_tag: str, base_metrics: Dict[str, float], filt_metrics: Dict[str, float]):
    labels = ["mAP50", "mAP50-95"]
    base_vals = [base_metrics.get("map50", 0.0), base_metrics.get("map50_95", 0.0)]
    filt_vals = [filt_metrics.get("map50", 0.0), filt_metrics.get("map50_95", 0.0)]
    x = range(len(labels)); width = 0.35
    plt.figure()
    plt.bar([i - width/2 for i in x], base_vals, width, label="baseline")
    plt.bar([i + width/2 for i in x], filt_vals, width, label="top1_filter")
    plt.xticks(list(x), labels); plt.ylabel("Score")
    plt.title(f"Baseline vs Top1 filter — {model_tag}")
    plt.legend(); plt.tight_layout()
    out = PLOTS_DIR / f"bar_{model_tag}.png"
    plt.savefig(out); plt.close()

# ========= MAIN =========
def main():
    print(f"[INFO] Base dir: {BASE_DIR}")
    print(f"[INFO] Data yaml: {DATA_YAML}")
    print(f"[INFO] Benchmark runs dir: {RUNS_DIR}")
    print(f"[INFO] Plots dir: {PLOTS_DIR}")
    print(f"[INFO] Models dir: {MODELS_DIR}")

    summary_rows = []
    model2resultscsv: Dict[str, Path] = {}

    for name in CANDIDATES:
        print(f"\n=== CANDIDATE: {name} ===")
        p = find_model_path(name)
        if p is None:
            print(f"[SKIP] {name}: weights not found in ./models"); continue

        # Evita alias → usa copia 'local_<alias>.pt'
        backbone = localize_to_non_alias(p)
        if backbone != p:
            print(f"[USE] {backbone.name} (alias-safe)")

        run_name = f"{Path(name).stem}_moulting_bench_taskaware"
        try:
            run_dir = yolo_train(backbone, run_name)
        except Exception as e:
            print(f"[ERROR][TRAIN] {name}: {e}"); continue

        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            print(f"[WARN] {name}: best.pt not found in {best.parent}"); continue

        ovr = PER_MODEL_OVERRIDES.get(name, {})
        train_imgsz = int(ovr.get("imgsz", IMGSZ))

        # 1) Baseline validation (CLI)
        try:
            yolo_val_cli(best, desc=name, imgsz=train_imgsz)
        except Exception as e:
            print(f"[WARN][VAL-CLI] {name}: {e}")

        # 2) Filtered validation (API)
        filt_metrics = {}
        try:
            filt_metrics = yolo_val_filtered(best)
        except Exception as e:
            print(f"[WARN][VAL-TOP1] {name}: {e}")

        # Read training summary row (last epoch)
        results_csv = run_dir / "results.csv"
        stats = read_last_results_csv(results_csv)

        base_metrics = {
            "precision": safe_get_float(stats, "metrics/precision(B)", "metrics/precision"),
            "recall":    safe_get_float(stats, "metrics/recall(B)",    "metrics/recall"),
            "map50":     safe_get_float(stats, "metrics/mAP50(B)",     "metrics/mAP50"),
            "map50_95":  safe_get_float(stats, "metrics/mAP50-95(B)",  "metrics/mAP50-95"),
        }

        barplot_baseline_vs_filtered(Path(name).stem, base_metrics, filt_metrics)
        copy_ultralytics_plots(run_dir, model_tag=Path(name).stem)
        model2resultscsv[Path(name).stem] = results_csv
        out_prefix = PLOTS_DIR / f"{Path(name).stem}"
        plot_individual_curves(Path(name).stem, results_csv, out_prefix)

        row = {
            "model": name,
            "baseline_precision": base_metrics["precision"],
            "baseline_recall":    base_metrics["recall"],
            "baseline_mAP50":     base_metrics["map50"],
            "baseline_mAP50_95":  base_metrics["map50_95"],
            "filtered_precision": filt_metrics.get("precision"),
            "filtered_recall":    filt_metrics.get("recall"),
            "filtered_mAP50":     filt_metrics.get("map50"),
            "filtered_mAP50_95":  filt_metrics.get("map50_95"),
            "run_dir": str(run_dir),
        }
        summary_rows.append(row)

        def fmt(x):
            if x is None or (isinstance(x, float) and (x != x)): return "nan"
            return f"{x:.4f}"

        print(
            f"{name:>12} | base mAP50={fmt(row['baseline_mAP50'])} "
            f"| base mAP50-95={fmt(row['baseline_mAP50_95'])} "
            f"| filt mAP50={fmt(row['filtered_mAP50'])} "
            f"| filt mAP50-95={fmt(row['filtered_mAP50_95'])}"
        )

    # Save summary CSV
    out_csv = RUNS_DIR / "benchmark_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model",
            "baseline_precision","baseline_recall","baseline_mAP50","baseline_mAP50_95",
            "filtered_precision","filtered_recall","filtered_mAP50","filtered_mAP50_95",
            "run_dir"
        ])
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"\n[✓] Saved summary → {out_csv}")

    # Multi-model comparative plots
    plot_multimodel_curves(
        model2resultscsv,
        metric_col="metrics/mAP50(B)",
        out_path=PLOTS_DIR / "multimodel_mAP50_per_epoch.png",
        title="mAP@50 per epoch — multi-model",
    )
    plot_multimodel_curves(
        model2resultscsv,
        metric_col="metrics/mAP50-95(B)",
        out_path=PLOTS_DIR / "multimodel_mAP50_95_per_epoch.png",
        title="mAP@50-95 per epoch — multi-model",
    )
    plot_multimodel_curves(
        model2resultscsv,
        metric_col="metrics/precision(B)",
        out_path=PLOTS_DIR / "multimodel_precision_per_epoch.png",
        title="Precision per epoch — multi-model",
    )
    plot_multimodel_curves(
        model2resultscsv,
        metric_col="metrics/recall(B)",
        out_path=PLOTS_DIR / "multimodel_recall_per_epoch.png",
        title="Recall per epoch — multi-model",
    )

    print(f"[✓] Plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
