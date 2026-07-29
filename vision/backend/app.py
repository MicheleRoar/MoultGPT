# app.py — MoultVision backend (YOLO + XGBoost + FastSAM)
import os, io, time, platform, math, base64
from typing import List, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat
import numpy as np
import torch
from ultralytics import YOLO
import joblib

# ───────────────────────── Config ─────────────────────────
PORT               = int(os.getenv("PORT", 5001))
MODEL_PATH         = os.getenv("MODEL_PATH", "./models/yolo_detect.pt")
STAGE_MODEL_PATH   = os.getenv("STAGE_MODEL_PATH", "./models/xgboost_stage.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "./models/label_encoder.pkl")
FASTSAM_WEIGHTS    = os.getenv("FASTSAM_WEIGHTS", "./models/FastSAM-x.pt")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.35))
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD", 0.45))
IMG_SIZE       = int(os.getenv("IMG_SIZE", "1024"))
IOU_CROSS_NMS  = float(os.getenv("IOU_CROSS_NMS", "0.50"))
REQ_CUDA       = os.getenv("USE_CUDA", "auto").lower()
REQ_MPS        = os.getenv("USE_MPS",  "auto").lower()
DEBUG_FEATURES = os.getenv("DEBUG_FEATURES", "0").lower() in ("1", "true", "yes")

# Distance bias (only when both boxes are present)
ALPHA_DIST        = float(os.getenv("ALPHA_DIST", 0.45))
DIST_SCALE        = float(os.getenv("DIST_SCALE", 0.80))

# Hard rule towards post-moult
DIST_POST_HARD    = float(os.getenv("DIST_POST_HARD", 0.33))
AREA_RATIO_POST   = float(os.getenv("AREA_RATIO_POST", 0.55))
OVERLAP_POST_MAX  = float(os.getenv("OVERLAP_POST_MAX", 0.03))

# ─────────────────────── XGBoost Features ───────────────────────
TRAIN_FEATURES = [
    "box_overlap", "dist_centroids", "x_organism", "y_organism",
    "x_exuviae", "y_exuviae", "h_exuviae",
    "org_mean_g", "org_mean_gray",
    "taxon_group_Crustacea", "taxon_group_Hexapoda",
    "taxon_group_Chelicerata", "taxon_group_Myriapoda",
    "only_exuviae"
]
# Fixed mapping index → label
IDX2LABEL = ["post-moult", "moulting", "exuviae"]

# ───────────────────────── Device ─────────────────────────
def pick_device() -> str:
    """Pick best available device: CUDA → MPS → CPU."""
    if REQ_CUDA != "false" and torch.cuda.is_available():
        return "cuda"
    if REQ_MPS != "false" and platform.system() == "Darwin":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
        except Exception:
            pass
    return "cpu"

DEVICE = pick_device()
USE_AMP = DEVICE == "cuda"

# ───────────────────────── App ─────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─────────────────────── YOLO load ───────────────────────
MODEL_OK, LOAD_ERR = False, None
try:
    yolo = YOLO(MODEL_PATH)
    try:
        yolo.to(DEVICE)
    except Exception:
        # On CPU-only environments this may fail, but the model still works
        pass
    MODEL_OK = True
except Exception as e:
    LOAD_ERR = str(e)

# ───────────────────── XGBoost load ─────────────────────
STAGE_OK, STAGE_ERR = False, None
stage_clf, class_labels = None, None
try:
    stage_clf = joblib.load(STAGE_MODEL_PATH)

    # Try to recover class labels from the model / encoder
    if hasattr(stage_clf, "classes_"):
        class_labels = [str(c) for c in list(stage_clf.classes_)]
    elif hasattr(stage_clf, "label_encoder_"):
        class_labels = [str(c) for c in list(stage_clf.label_encoder_.classes_)]
    elif os.path.exists(LABEL_ENCODER_PATH):
        le = joblib.load(LABEL_ENCODER_PATH)
        if hasattr(le, "classes_"):
            class_labels = [str(c) for c in list(le.classes_)]
    STAGE_OK = True
except Exception as e:
    STAGE_ERR = f"Cannot load XGB: {e}"
    stage_clf = None
    class_labels = None

# ───────────────────── FastSAM import+init ─────────────────────
def _import_fastsam():
    """Try multiple import paths for FastSAM."""
    try:
        from ultralytics import FastSAM
        return FastSAM, "ultralytics", None
    except Exception as e1:
        err1 = str(e1)
    try:
        from ultralytics.models.fastsam import FastSAM  # type: ignore
        return FastSAM, "ultralytics.models", None
    except Exception as e2:
        err2 = str(e2)
    try:
        from fastsam import FastSAM  # type: ignore
        return FastSAM, "fastsam", None
    except Exception as e3:
        err3 = str(e3)
    try:
        from FastSAM import FastSAM  # type: ignore
        return FastSAM, "FastSAM", None
    except Exception as e4:
        err4 = str(e4)
    return None, None, (
        f"import_error: ultralytics[{err1}] | ultralytics.models[{err2}] | "
        f"fastsam[{err3}] | FastSAM[{err4}]"
    )

FastSAM, _fs_src, _fs_err = _import_fastsam()
FASTSAM_AVAILABLE = FastSAM is not None

SEG_OK, SEG_ERR, fastsam_model = False, None, None
if FASTSAM_AVAILABLE:
    if os.path.exists(FASTSAM_WEIGHTS):
        try:
            fastsam_model = FastSAM(FASTSAM_WEIGHTS)
            SEG_OK = True
        except Exception as e:
            SEG_ERR = f"load_error: {e}"
    else:
        SEG_ERR = f"weights_not_found: {FASTSAM_WEIGHTS}"
else:
    SEG_ERR = _fs_err

# ───────────────────────── Warmup ─────────────────────────
def warmup():
    """Do a single dummy forward for faster first real inference."""
    if not MODEL_OK:
        return
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    kw = dict(imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
              device=DEVICE, verbose=False)
    if USE_AMP:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yolo.predict(dummy, **kw)
    else:
        yolo.predict(dummy, **kw)

if MODEL_OK:
    warmup()

# ───────────────────────── Utils ─────────────────────────
def model_names(m):
    """Return YOLO class names from model (handles different layouts)."""
    if hasattr(m, "names") and m.names:
        return m.names
    if hasattr(m, "model") and hasattr(m.model, "names"):
        return m.model.names
    return None

def bucket(name_or_id, names):
    """
    Map YOLO class ID/label to either 'organism' or 'exuviae'.
    Anything containing 'exuv' (case-insensitive) is mapped to exuviae.
    """
    if isinstance(names, dict):
        label = names.get(int(name_or_id), str(name_or_id))
    elif isinstance(name_or_id, (int, np.integer)) and names is not None:
        try:
            label = names[int(name_or_id)]
        except Exception:
            label = str(name_or_id)
    else:
        label = str(name_or_id)
    return "exuviae" if "exuv" in str(label).lower() else "organism"

def _area(x1, y1, x2, y2):
    """Axis-aligned bounding-box area."""
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _iou_boxes(b1, b2):
    """IoU between two [x1,y1,x2,y2] boxes."""
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

def _clip(b, w, h):
    """Clip a box to [0,w-1] x [0,h-1]."""
    if not b:
        return None
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(x1), w - 1))
    y1 = max(0.0, min(float(y1), h - 1))
    x2 = max(0.0, min(float(x2), w - 1))
    y2 = max(0.0, min(float(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def _center(b):
    """Return box center (cx, cy)."""
    if not b:
        return None
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _norm(v, d):
    """Normalize coordinate v by dimension d to [0,1]."""
    return float(v / (d + 1e-9))

def _b64_png_from_mask(mask_bool: np.ndarray, color=(0, 153, 255, 115)) -> str:
    """Encode a boolean mask into a colored RGBA PNG (base64)."""
    h, w = mask_bool.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[mask_bool] = np.array(color, dtype=np.uint8)
    from PIL import Image as _Image
    im = _Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _mask_bbox(mask_bool: np.ndarray) -> Optional[List[float]]:
    """Get bounding box [x1,y1,x2,y2] of a boolean mask."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return [x1, y1, x2 + 1.0, y2 + 1.0]

def _filter_cross_overlap(dets_by_cls, thr=0.5):
    """
    Soft cross-class NMS:
    - if organism and exuviae boxes overlap > thr, keep only the one with higher confidence.
    """
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

def _box_to_mask(box, h, w) -> np.ndarray:
    """Convert a single box into a boolean mask."""
    if not box:
        return np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=bool)
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m

def build_features(pil: Image.Image, W: int, H: int, best: dict, taxon_id):
    """
    Build feature vector for XGBoost from best organism/exuviae boxes
    and image-level statistics.
    """
    f = {n: -1.0 for n in TRAIN_FEATURES}
    b_o = _clip(best.get("organism"), W, H)
    b_e = _clip(best.get("exuviae"), W, H)

    # IoU between organism and exuviae
    f["box_overlap"] = _iou_boxes(b_o, b_e) if (b_o and b_e) else 0.0

    cx_o = cy_o = cx_e = cy_e = None
    if b_o:
        cx_o, cy_o = _center(b_o)
        f["x_organism"] = _norm(cx_o, W)
        f["y_organism"] = _norm(cy_o, H)
    if b_e:
        cx_e, cy_e = _center(b_e)
        f["x_exuviae"] = _norm(cx_e, W)
        f["y_exuviae"] = _norm(cy_e, H)

    # Distance between centroids (in normalized coordinates)
    f["dist_centroids"] = float(
        math.hypot(f["x_organism"] - f["x_exuviae"], f["y_organism"] - f["y_exuviae"])
    ) if (cx_o is not None and cx_e is not None) else 1.0

    # Relative height of exuviae box
    f["h_exuviae"] = _norm((b_e[3] - b_e[1]), H) if b_e else 0.0

    # Organism color / gray stats
    try:
        if b_o:
            x1, y1, x2, y2 = map(int, b_o)
            crop = pil.crop((x1, y1, x2, y2))
            stat_rgb = ImageStat.Stat(crop)
            mean_g = (stat_rgb.mean[1] if len(stat_rgb.mean) >= 2 else 0.0) / 255.0
            gray = crop.convert("L")
            stat_g = ImageStat.Stat(gray)
            mean_gray = (stat_g.mean[0] if stat_g.mean else 0.0) / 255.0
            f["org_mean_g"] = float(mean_g)
            f["org_mean_gray"] = float(mean_gray)
        else:
            f["org_mean_g"] = 0.0
            f["org_mean_gray"] = 0.0
    except Exception:
        pass

    # One-hot taxon group
    f["taxon_group_Crustacea"] = 0.0
    f["taxon_group_Hexapoda"] = 0.0
    f["taxon_group_Chelicerata"] = 0.0
    f["taxon_group_Myriapoda"] = 0.0
    try:
        tid = int(taxon_id) if (taxon_id is not None and str(taxon_id) != "") else None
        if tid == 0:
            f["taxon_group_Crustacea"] = 1.0
        elif tid == 1:
            f["taxon_group_Hexapoda"] = 1.0
        elif tid == 2:
            f["taxon_group_Chelicerata"] = 1.0
        elif tid == 3:
            f["taxon_group_Myriapoda"] = 1.0
    except Exception:
        pass

    # Flag: only exuviae present, no organism
    f["only_exuviae"] = 1.0 if (b_e is not None and b_o is None) else 0.0

    # Clean NaNs / infs
    for k, v in f.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            f[k] = -1.0
    return f

def _safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    return obj

# ───────────────────────── Routes ─────────────────────────
@app.get("/healthz")
def healthz():
    """Basic health check endpoint."""
    return jsonify({
        "ok": MODEL_OK,
        "device": DEVICE,
        "model_loaded": MODEL_OK,
        "stage_model_loaded": STAGE_OK,
        "segmentation_loaded": SEG_OK,
        "seg_backend": _fs_src,
        "fastsam_weights": FASTSAM_WEIGHTS,
        "error": LOAD_ERR if not MODEL_OK else None,
        "stage_error": STAGE_ERR if not STAGE_OK else None,
        "seg_error": SEG_ERR if not SEG_OK else None
    }), (200 if MODEL_OK else 500)

@app.post("/predict_image")
def predict_image():
    """Main endpoint: YOLO detection + XGBoost stage + optional FastSAM segmentation."""
    if not MODEL_OK:
        return jsonify({"ok": False, "error": "YOLO not loaded"}), 500
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "missing 'image' field"}), 400

    taxon_id = request.form.get("taxon_id")
    use_seg = request.form.get("use_seg", "0").lower() in ("1", "true", "yes")

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "empty filename"}), 400

    try:
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        W, H = pil.size

        # ───────────── YOLO inference ─────────────
        kw = dict(imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                  device=DEVICE, verbose=False)
        t0 = time.time()
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                res = yolo.predict(pil, **kw)
        else:
            res = yolo.predict(pil, **kw)
        infer_ms = int((time.time() - t0) * 1000)

        names = model_names(yolo)
        dets_by_cls = {"organism": [], "exuviae": []}

        r = res[0]
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                det = {"box": [float(x1), float(y1), float(x2), float(y2)],
                       "conf": float(p)}
                dets_by_cls[bucket(c, names)].append(det)

        # Soft cross-class NMS
        _filter_cross_overlap(dets_by_cls, IOU_CROSS_NMS)

        # Best detection per class (for XGBoost and rules)
        best = {"organism": None, "exuviae": None}
        best_conf = {"organism": -1.0, "exuviae": -1.0}
        for k in ("organism", "exuviae"):
            for d in dets_by_cls[k]:
                if d["conf"] > best_conf[k]:
                    best_conf[k] = d["conf"]
                    best[k] = d["box"]

        # ───────────── Stage decision (rules + XGBoost + distance bias) ─────────────
        stage_label = None
        stage_confidence = None
        stage_error = None
        stage_rule = None
        debug_feats = None

        has_org = best.get("organism") is not None
        has_exu = best.get("exuviae") is not None

        try:
            if has_exu and not has_org:
                # Only exuviae → exuviae class
                stage_label = IDX2LABEL[2]
                stage_confidence = 1.0
                stage_rule = "only_exuviae_forced_2"

            elif has_org and not has_exu:
                # Only organism → post-moult
                stage_label = IDX2LABEL[0]
                stage_confidence = 1.0
                stage_rule = "only_organism_forced_0"

            elif has_org and has_exu and STAGE_OK and stage_clf is not None:
                # Both present → XGBoost + distance/size rules
                feat = build_features(pil, W, H, best, taxon_id)
                order = list(getattr(stage_clf, "feature_names_in_", TRAIN_FEATURES))
                X = np.array([[float(feat.get(n, -1.0)) for n in order]], dtype=float)

                if DEBUG_FEATURES:
                    debug_feats = {
                        "ordered": list(zip(order, X[0].tolist())),
                        "missing": [n for n in order if n not in feat]
                    }

                # Box geometry for hard rule
                b_o = _clip(best.get("organism"), W, H)
                b_e = _clip(best.get("exuviae"), W, H)
                iou = _iou_boxes(b_o, b_e)
                area_o = _area(*b_o)
                area_e = _area(*b_e)
                area_ratio = (area_e / (area_o + 1e-9)) if area_o > 0 else 1.0

                # Normalized centroid distance (0..1 along image diagonal)
                dist_raw = float(feat.get("dist_centroids", 0.0))   # 0..~1.414
                dist_norm = min(1.0, dist_raw / math.sqrt(2.0))    # 0..1

                # Hard rule: far apart OR exuviae much smaller with almost no overlap → post-moult
                if (dist_norm >= DIST_POST_HARD) or (iou <= OVERLAP_POST_MAX and area_ratio <= AREA_RATIO_POST):
                    stage_label = IDX2LABEL[0]  # post-moult
                    stage_confidence = float(max(0.75, 0.55 + 0.4 * dist_norm))
                    stage_rule = "hard_post_by_distance_or_size"
                else:
                    # Distance-based bias on post vs moulting (indexes 0 and 1)
                    d = max(0.0, min(1.0, dist_norm / max(1e-6, DIST_SCALE)))

                    if hasattr(stage_clf, "predict_proba"):
                        proba = stage_clf.predict_proba(X)[0]
                        p0, p1 = float(proba[0]), float(proba[1])  # 0: post, 1: moulting

                        p0_adj = p0 + ALPHA_DIST * d
                        p1_adj = p1 + ALPHA_DIST * (1.0 - d)

                        s = p0_adj + p1_adj + 1e-9
                        p0_adj /= s
                        p1_adj /= s

                        idx01 = 0 if p0_adj >= p1_adj else 1
                        stage_label = IDX2LABEL[idx01]
                        stage_confidence = float(max(p0_adj, p1_adj))
                    else:
                        idx = int(stage_clf.predict(X)[0])
                        # If model yields exuviae in the "both boxes" case, clamp to {post,moulting}
                        if idx == 2:
                            idx = 0 if d >= 0.5 else 1
                        stage_label = IDX2LABEL[idx]
                        stage_confidence = None

                    stage_rule = "both_xgboost_clamped_0_1_with_distance_bias"

            else:
                # No boxes → no stage
                stage_label = None
                stage_confidence = None
                stage_rule = "no_boxes_no_stage"

        except Exception as e:
            stage_error = str(e)

        # ───────────── Base detections for frontend ─────────────
        detections_out = []
        for cls_name in ("organism", "exuviae"):
            for d in dets_by_cls[cls_name]:
                detections_out.append({
                    "cls": cls_name,
                    "conf": float(d["conf"]),
                    "box": [float(x) for x in d["box"]],
                    "mask_png": None,
                    "quality": None
                })

        masks_out = None

        # ───────────── Optional: instance segmentation per detection ─────────────
        if use_seg and SEG_OK and fastsam_model and detections_out:
            try:
                fs_res = fastsam_model(
                    pil,
                    device=DEVICE,
                    retina_masks=True,
                    imgsz=IMG_SIZE,
                    verbose=False
                )
                if fs_res and hasattr(fs_res[0], "masks") and fs_res[0].masks is not None:
                    masks_np = fs_res[0].masks.data.cpu().numpy().astype(bool)  # (N,H,W)
                    Hh = masks_np.shape[1] if masks_np.size else H
                    Ww = masks_np.shape[2] if masks_np.size else W

                    # For each detection, pick best FastSAM mask by IoU and clamp to its box
                    det_masks = [None] * len(detections_out)
                    for idx, det in enumerate(detections_out):
                        box = det["box"]
                        best_iou, best_idx = -1.0, -1
                        for i in range(masks_np.shape[0]):
                            bb = _mask_bbox(masks_np[i])
                            if not bb:
                                continue
                            iou_m = _iou_boxes(box, bb)
                            if iou_m > best_iou:
                                best_iou, best_idx = iou_m, i
                        if best_idx >= 0 and best_iou >= 0.15:
                            box_mask = _box_to_mask(box, Hh, Ww)
                            det_masks[idx] = (masks_np[best_idx] & box_mask)

                    # If organism and exuviae masks overlap, organism "wins"
                    org_union = np.zeros((Hh, Ww), dtype=bool)
                    for idx, det in enumerate(detections_out):
                        if det_masks[idx] is not None and det["cls"] == "organism":
                            org_union |= det_masks[idx]
                    for idx, det in enumerate(detections_out):
                        if det_masks[idx] is not None and det["cls"] == "exuviae":
                            det_masks[idx] &= ~org_union

                    # Assign per-detection mask PNGs
                    for idx, det in enumerate(detections_out):
                        m = det_masks[idx]
                        if m is not None and m.any():
                            color = (255, 77, 77, 115) if det["cls"] == "organism" else (0, 153, 255, 115)
                            det["mask_png"] = _b64_png_from_mask(m, color=color)
                            det["quality"] = (
                                "High" if det["conf"] >= 0.7
                                else ("Medium" if det["conf"] >= 0.45 else "Low")
                            )
                        else:
                            det["quality"] = det.get("quality") or "NoMatch"

            except Exception as e:
                masks_out = masks_out or []
                masks_out.append({"error": f"segmentation_error: {str(e)[:120]}"})

        # ───────────── Final JSON response ─────────────
        out = {
            "ok": True,
            "model": os.path.basename(MODEL_PATH),
            "device": DEVICE,
            "inference_ms": infer_ms,
            "image": {"width": int(W), "height": int(H)},
            "detections": _safe(detections_out),
            "use_seg": bool(use_seg),
            "segmentation_loaded": SEG_OK,
            "masks": masks_out,
            "stage": stage_label,
            "stage_confidence": float(stage_confidence) if stage_confidence is not None else None,
            "class_labels": IDX2LABEL,
            "stage_rule": stage_rule,
        }
        if stage_error or not STAGE_OK:
            out["stage_error"] = (stage_error or "") + ("" if STAGE_OK else " | stage model not loaded")
        if DEBUG_FEATURES and "feat" in locals():
            out["debug_features"] = _safe(debug_feats)

        return jsonify(out)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ───────────────────────── Main ─────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
