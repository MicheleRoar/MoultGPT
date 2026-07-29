#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ============== CONFIG GENERALE ==============
MODEL_NAME = "yolo11n.pt"   # <— corretto
BASE_DIR   = Path(__file__).resolve().parents[2]
YOLO_DATA  = BASE_DIR / "data" / "yolo" / "moulting.yaml"

PROJECT    = BASE_DIR / "scripts" / "training" / "runs" / "single_run"
EXP_NAME   = "yolo11n_moult_opt_smallobj"

# Iperparametri (small-object friendly)
IMGSZ   = 1280                    # usa 1024 se vuoi alleggerire
EPOCHS  = 300
BATCH   = -1                      # auto-batch; in alternativa 6–16
DEVICE  = os.getenv("YOLO_DEVICE", "0")
SEED    = 42

# Ottimizzati per v11-n (LR basso, augment gentile, loss bilanciate)
TRAIN_ARGS = [
    "optimizer=AdamW",
    "cos_lr=True",
    "weight_decay=0.0005",

    # Augmentazioni "gentili"
    "mosaic=0.50",
    "mixup=0.05",
    "copy_paste=0.05",
    "close_mosaic=20",
    "rect=False",
    "hsv_h=0.015",
    "hsv_s=0.60",
    "hsv_v=0.20",
    "translate=0.12",
    "scale=0.60",
    "degrees=0.0",
    "shear=0.0",
    "label_smoothing=0.005",

    # Loss balancing (più peso a box/dfl, meno a cls) + focal
    "box=7.5",
    "cls=0.30",
    "dfl=2.0",

    # QoL training
    "cache=True",
    "deterministic=True",
    "patience=60",
    "workers=8",
    "plots=True",
    "warmup_epochs=5",

    # LR adatto a 'n' (più basso del classico 0.01)
    "lr0=0.0025",
    "lrf=0.01",
]

# ============== PREDICT/FILTER CONFIG ==============
# Soglie più restrittive per ridurre duplicati
PRED_CONF = 0.35
PRED_IOU  = 0.45
AGNOSTIC  = False
MAX_DET   = 2           # 2 classi totali => tiene massimo 2 box globali
SAVE_TXT  = True        # salva CSV (nostro) con box filtrati
SAVE_IMG  = True        # salva immagini annotate con box filtrati

# Se non fornisci --source, tenta di leggere il "val" dallo YAML
DEFAULT_SOURCE = None

# ===================================================

def train():
    PROJECT.mkdir(parents=True, exist_ok=True)
    print("[INFO] Starting YOLOv11-n training…")

    cmd = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model={MODEL_NAME}",
        f"data={YOLO_DATA}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={BATCH}",
        f"device={DEVICE}",
        f"project={PROJECT}",
        f"name={EXP_NAME}",
        f"seed={SEED}",
        "verbose=True",
    ] + TRAIN_ARGS

    print("\n=== RUN:", " ".join(map(str, cmd)), "\n")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("[✓] Training completed.")
    else:
        print("[✗] Training failed.")
        sys.exit(1)


def _try_read_val_from_yaml(yaml_path: Path) -> str:
    # Prova a leggere lo YAML per trovare il path del set di validazione
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text())
        for key in ("val", "val_dataset", "val_data"):
            if key in data:
                return str(data[key])
    except Exception:
        pass
    return ""


def predict_one_per_class(
    weights_path: Path,
    source: str,
    conf: float = PRED_CONF,
    iou: float = PRED_IOU,
    agnostic: bool = AGNOSTIC,
    max_det: int = MAX_DET,
    imgsz: int = IMGSZ,
    save_dir: Path = None,
    class_map: Dict[int, str] = None,
):
    """
    Esegue la predizione con Ultralytics e applica:
      - NMS restrittivo (conf/iou/agnostic/max_det)
      - filtro "1 box per classe": tiene il top-1 per conf per ogni classe presente

    Salva:
      - immagini annotate con box filtrati (se SAVE_IMG=True)
      - CSV (uno per immagine) con box filtrati (xyxy, conf, cls)
    """
    from ultralytics import YOLO
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    model = YOLO(str(weights_path))
    print(f"[INFO] Loading weights: {weights_path}")

    # Directory di output
    if save_dir is None:
        save_dir = weights_path.parent / "filtered_predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_dir  = save_dir / "csv"
    img_dir  = save_dir / "images"
    if SAVE_TXT:
        csv_dir.mkdir(exist_ok=True, parents=True)
    if SAVE_IMG:
        img_dir.mkdir(exist_ok=True, parents=True)

    # Stream=True per processare immagine per immagine
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic,
        max_det=max_det,
        stream=True,
        save=False,
        verbose=False,
    )

    for r in results:
        path = Path(r.path)  # percorso immagine
        base = path.stem

        boxes = r.boxes  # Boxes obj
        if boxes is None or len(boxes) == 0:
            # salva vuoto
            if SAVE_TXT:
                (csv_dir / f"{base}.csv").write_text("")  # nessun box
            continue

        xyxy = boxes.xyxy.cpu()            # [N,4]
        confs = boxes.conf.cpu()           # [N]
        clss  = boxes.cls.cpu().long()     # [N]

        # --- filtro 1-per-classe ---
        keep_idx = []
        for c in torch.unique(clss):
            idx = torch.where(clss == c)[0]
            if len(idx) == 0:
                continue
            # prendi il migliore per conf
            best_local = idx[confs[idx].argmax().item()]
            keep_idx.append(best_local)

        if len(keep_idx) == 0:
            if SAVE_TXT:
                (csv_dir / f"{base}.csv").write_text("")
            continue

        keep_idx = torch.stack(keep_idx)
        k_xyxy, k_confs, k_clss = xyxy[keep_idx], confs[keep_idx], clss[keep_idx]

        # ---- salva CSV ----
        if SAVE_TXT:
            with (csv_dir / f"{base}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x1", "y1", "x2", "y2", "conf", "cls", "cls_name"])
                for i in range(len(k_xyxy)):
                    x1, y1, x2, y2 = map(float, k_xyxy[i].tolist())
                    conf_val = float(k_confs[i].item())
                    c_id     = int(k_clss[i].item())
                    c_name   = class_map.get(c_id, str(c_id)) if class_map else str(c_id)
                    w.writerow([x1, y1, x2, y2, conf_val, c_id, c_name])

        # ---- salva immagine con box ----
        if SAVE_IMG:
            try:
                im = Image.open(path).convert("RGB")
                draw = ImageDraw.Draw(im)
                for i in range(len(k_xyxy)):
                    x1, y1, x2, y2 = map(float, k_xyxy[i].tolist())
                    c_id = int(k_clss[i].item())
                    c_name = class_map.get(c_id, str(c_id)) if class_map else str(c_id)
                    label = f"{c_name} {k_confs[i].item():.2f}"
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=3)
                    draw.text((x1+4, y1+4), label, fill=(255, 255, 255))
                im.save(img_dir / f"{base}.png")
            except Exception as e:
                print(f"[WARN] Impossibile salvare immagine annotata per {path.name}: {e}")

    print(f"[✓] Predizioni filtrate salvate in: {save_dir}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict"], default="train")
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--source", type=str, default="")
    ap.add_argument("--classmap", type=str, default="")  # JSON: {"0":"organism","1":"exuviae"}
    args = ap.parse_args()

    if args.mode == "train":
        train()
        print("\nSuggerimento: per validare baseline mAP (senza filtro) usa, ad es.:\n"
              f"  yolo val model={PROJECT / EXP_NAME / 'weights' / 'best.pt'} "
              f"data={YOLO_DATA} imgsz={IMGSZ} conf={PRED_CONF} iou={PRED_IOU} "
              f"agnostic_nms={str(AGNOSTIC)} max_det={MAX_DET}\n")
    else:
        # Sorgente predizioni
        src = args.source or DEFAULT_SOURCE or _try_read_val_from_yaml(YOLO_DATA)
        if not src:
            print("[ERR] Nessuna sorgente specificata e nessun 'val' trovato nello YAML.")
            sys.exit(2)

        # Weights
        w = Path(args.weights) if args.weights else (PROJECT / EXP_NAME / "weights" / "best.pt")
        if not w.exists():
            print(f"[ERR] Weights non trovati: {w}")
            sys.exit(3)

        # Class map opzionale
        class_map = None
        if args.classmap:
            try:
                class_map = json.loads(Path(args.classmap).read_text())
            except Exception:
                class_map = None

        predict_one_per_class(
            weights_path=w,
            source=src,
            conf=PRED_CONF,
            iou=PRED_IOU,
            agnostic=AGNOSTIC,
            max_det=MAX_DET,
            imgsz=IMGSZ,
            save_dir=None,
            class_map=class_map
        )


if __name__ == "__main__":
    main()
