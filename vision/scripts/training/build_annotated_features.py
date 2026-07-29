#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build enriched feature table for downstream stage classification.

This script:
1. Loads the manually annotated dataset
2. Reconstructs image paths
3. Computes geometric, relational, and normalized box features
4. Computes crop-based color statistics for organism and exuviae
5. Computes color-difference features between organism and exuviae
6. Adds taxon-group one-hot features
7. Saves an enriched CSV for classifier training

Expected input:
- Annotated CSV with at least:
    stage, observation_id, photo_id, taxon_id, taxon_name, taxon_group
    x_exuviae, y_exuviae, w_exuviae, h_exuviae
    x_organism, y_organism, w_organism, h_organism
    optionally x_suture, y_suture

Expected image layout:
data/inat_raw/<stage>/<observation_id>_<photo_id>.jpg
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_CSV = BASE_DIR / "data" / "inat_raw" / "inat_dataset.csv"
IMAGE_ROOT = BASE_DIR / "data" / "inat_raw"
OUTPUT_CSV = BASE_DIR / "data" / "annotated_features.csv"

VALID_TAXON_GROUPS = ["Chelicerata", "Crustacea", "Hexapoda", "Myriapoda"]


# ============================================================
# Helper functions
# ============================================================
def ensure_column(df: pd.DataFrame, column_name: str, default_value=np.nan) -> pd.DataFrame:
    """Ensure that a column exists in the dataframe."""
    if column_name not in df.columns:
        df[column_name] = default_value
    return df


def has_valid_box(row: pd.Series, prefix: str) -> bool:
    """Check whether a bounding box exists and is valid."""
    needed = [f"x_{prefix}", f"y_{prefix}", f"w_{prefix}", f"h_{prefix}"]
    if any(col not in row.index for col in needed):
        return False
    if any(pd.isna(row[col]) for col in needed):
        return False
    return float(row[f"w_{prefix}"]) > 0 and float(row[f"h_{prefix}"]) > 0


def get_box_xywh(row: pd.Series, prefix: str) -> Optional[Tuple[float, float, float, float]]:
    """Return a valid bounding box in xywh format."""
    if not has_valid_box(row, prefix):
        return None
    return (
        float(row[f"x_{prefix}"]),
        float(row[f"y_{prefix}"]),
        float(row[f"w_{prefix}"]),
        float(row[f"h_{prefix}"]),
    )


def get_box_xyxy(row: pd.Series, prefix: str) -> Optional[Tuple[float, float, float, float]]:
    """Convert a stored xywh box into xyxy format."""
    box = get_box_xywh(row, prefix)
    if box is None:
        return None

    x, y, w, h = box
    return x, y, x + w, y + h


def get_box_center(row: pd.Series, prefix: str) -> Optional[Tuple[float, float]]:
    """Return the center of a valid bounding box."""
    box = get_box_xywh(row, prefix)
    if box is None:
        return None

    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def get_box_area(row: pd.Series, prefix: str) -> float:
    """Return box area or NaN if invalid."""
    box = get_box_xywh(row, prefix)
    if box is None:
        return np.nan

    _, _, w, h = box
    return float(w * h)


def compute_centroid_distance(row: pd.Series) -> float:
    """Euclidean distance between organism and exuviae box centers."""
    c_org = get_box_center(row, "organism")
    c_exu = get_box_center(row, "exuviae")

    if c_org is None or c_exu is None:
        return np.nan

    return float(np.sqrt((c_org[0] - c_exu[0]) ** 2 + (c_org[1] - c_exu[1]) ** 2))


def compute_dx_dy_centroids(row: pd.Series) -> Tuple[float, float]:
    """Signed centroid differences: exuviae center - organism center."""
    c_org = get_box_center(row, "organism")
    c_exu = get_box_center(row, "exuviae")

    if c_org is None or c_exu is None:
        return np.nan, np.nan

    dx = float(c_exu[0] - c_org[0])
    dy = float(c_exu[1] - c_org[1])
    return dx, dy


def compute_box_overlap(row: pd.Series) -> float:
    """Compute IoU between organism and exuviae boxes."""
    box_a = get_box_xyxy(row, "organism")
    box_b = get_box_xyxy(row, "exuviae")

    if box_a is None or box_b is None:
        return np.nan

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return np.nan

    return float(inter_area / union_area)


def compute_intersection_over_exuviae(row: pd.Series) -> float:
    """Intersection area divided by exuviae area."""
    box_a = get_box_xyxy(row, "organism")
    box_b = get_box_xyxy(row, "exuviae")

    if box_a is None or box_b is None:
        return np.nan

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    exu_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if exu_area <= 0:
        return np.nan

    return float(inter_area / exu_area)


def compute_intersection_over_organism(row: pd.Series) -> float:
    """Intersection area divided by organism area."""
    box_a = get_box_xyxy(row, "organism")
    box_b = get_box_xyxy(row, "exuviae")

    if box_a is None or box_b is None:
        return np.nan

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    org_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if org_area <= 0:
        return np.nan

    return float(inter_area / org_area)


def compute_size_ratio(row: pd.Series) -> float:
    """Compute exuviae area / organism area."""
    org_area = get_box_area(row, "organism")
    exu_area = get_box_area(row, "exuviae")

    if pd.isna(org_area) or pd.isna(exu_area) or org_area <= 0:
        return np.nan

    return float(exu_area / org_area)


def compute_point_to_box_center_distance(
    point_x: Any,
    point_y: Any,
    row: pd.Series,
    prefix: str,
) -> float:
    """Compute the distance from a point (e.g. suture) to a box center."""
    if pd.isna(point_x) or pd.isna(point_y):
        return np.nan

    center = get_box_center(row, prefix)
    if center is None:
        return np.nan

    return float(np.sqrt((float(point_x) - center[0]) ** 2 + (float(point_y) - center[1]) ** 2))


def clamp_box_to_image(
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a box to image boundaries and return integer crop coordinates."""
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_width, int(round(x + w)))
    y2 = min(image_height, int(round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def compute_crop_color_features(
    image_array: np.ndarray,
    row: pd.Series,
    prefix: str,
) -> Dict[str, float]:
    """
    Compute color statistics from the crop defined by a bounding box.

    Returned keys:
    - mean_r / mean_g / mean_b
    - std_rgb
    - var_gray
    - mean_gray
    """
    default_prefix = "org" if prefix == "organism" else "ex"
    result = {
        f"{default_prefix}_mean_r": np.nan,
        f"{default_prefix}_mean_g": np.nan,
        f"{default_prefix}_mean_b": np.nan,
        f"{default_prefix}_std_rgb": np.nan,
        f"{default_prefix}_var_gray": np.nan,
        f"{default_prefix}_mean_gray": np.nan,
    }

    if not has_valid_box(row, prefix):
        return result

    img_h, img_w = image_array.shape[:2]
    crop_coords = clamp_box_to_image(
        x=float(row[f"x_{prefix}"]),
        y=float(row[f"y_{prefix}"]),
        w=float(row[f"w_{prefix}"]),
        h=float(row[f"h_{prefix}"]),
        image_width=img_w,
        image_height=img_h,
    )

    if crop_coords is None:
        return result

    x1, y1, x2, y2 = crop_coords
    crop = image_array[y1:y2, x1:x2]

    if crop.size == 0:
        return result

    crop = crop.astype(np.float32)

    if crop.ndim == 2:
        gray = crop
        result[f"{default_prefix}_mean_r"] = float(np.mean(gray))
        result[f"{default_prefix}_mean_g"] = float(np.mean(gray))
        result[f"{default_prefix}_mean_b"] = float(np.mean(gray))
        result[f"{default_prefix}_std_rgb"] = float(np.std(gray))
        result[f"{default_prefix}_var_gray"] = float(np.var(gray))
        result[f"{default_prefix}_mean_gray"] = float(np.mean(gray))
        return result

    r = crop[:, :, 0]
    g = crop[:, :, 1]
    b = crop[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    result[f"{default_prefix}_mean_r"] = float(np.mean(r))
    result[f"{default_prefix}_mean_g"] = float(np.mean(g))
    result[f"{default_prefix}_mean_b"] = float(np.mean(b))
    result[f"{default_prefix}_std_rgb"] = float(np.std(crop))
    result[f"{default_prefix}_var_gray"] = float(np.var(gray))
    result[f"{default_prefix}_mean_gray"] = float(np.mean(gray))

    return result


def build_filename(row: pd.Series) -> str:
    """Build a filename from observation_id and photo_id if not already present."""
    if "filename" in row.index and pd.notna(row["filename"]):
        return str(row["filename"])

    obs_id = row.get("observation_id")
    photo_id = row.get("photo_id")

    if pd.isna(obs_id) or pd.isna(photo_id):
        return ""

    return f"{int(obs_id)}_{int(photo_id)}.jpg"


def build_image_path(row: pd.Series) -> Path:
    """Reconstruct the original image path from stage and filename."""
    filename = build_filename(row)
    stage = str(row.get("stage", "")).strip()
    return IMAGE_ROOT / stage / filename


def one_hot_taxon_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot columns for the expected taxon groups."""
    for group in VALID_TAXON_GROUPS:
        col = f"taxon_group_{group}"
        df[col] = (df["taxon_group"] == group).astype(int)
    return df


# ============================================================
# Core logic
# ============================================================
def build_annotated_features():
    """Build the final enriched dataset used for stage classification."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_optional_columns = [
        "filename",
        "x_suture", "y_suture",
        "x_exuviae", "y_exuviae", "w_exuviae", "h_exuviae",
        "x_organism", "y_organism", "w_organism", "h_organism",
    ]
    for col in required_optional_columns:
        df = ensure_column(df, col, np.nan)

    optional_landmark_columns = [
        "x_head_org", "y_head_org",
        "x_thorax_org", "y_thorax_org",
        "x_head_exu", "y_head_exu",
        "x_thorax_exu", "y_thorax_exu",
        "exu_ref",
    ]
    for col in optional_landmark_columns:
        df = ensure_column(df, col, np.nan)

    if "filename" not in df.columns or df["filename"].isna().all():
        df["filename"] = df.apply(build_filename, axis=1)

    df = one_hot_taxon_group(df)

    print(f"[INFO] Computing image-dependent features for {len(df)} rows...")

    rows_out = []

    for _, row in df.iterrows():
        image_path = build_image_path(row)

        # Default values
        image_width = np.nan
        image_height = np.nan
        image_area = np.nan
        image_diag = np.nan

        color_features = {
            "ex_mean_r": np.nan,
            "ex_mean_g": np.nan,
            "ex_mean_b": np.nan,
            "ex_std_rgb": np.nan,
            "ex_var_gray": np.nan,
            "ex_mean_gray": np.nan,
            "org_mean_r": np.nan,
            "org_mean_g": np.nan,
            "org_mean_b": np.nan,
            "org_std_rgb": np.nan,
            "org_var_gray": np.nan,
            "org_mean_gray": np.nan,
        }

        if image_path.exists():
            try:
                image = Image.open(image_path).convert("RGB")
                image_array = np.array(image)
                image_height, image_width = image_array.shape[:2]
                image_area = float(image_width * image_height)
                image_diag = float(np.sqrt(image_width**2 + image_height**2))

                ex_features = compute_crop_color_features(image_array, row, "exuviae")
                org_features = compute_crop_color_features(image_array, row, "organism")
                color_features.update(ex_features)
                color_features.update(org_features)

            except Exception as exc:
                print(f"[WARN] Failed to process image {image_path.name}: {exc}")
        else:
            print(f"[WARN] Missing image: {image_path}")

        row_out = row.to_dict()
        row_out.update(color_features)

        # -----------------------------
        # Valid-box helper flags
        # -----------------------------
        has_org = int(has_valid_box(row, "organism"))
        has_exu = int(has_valid_box(row, "exuviae"))

        row_out["has_organism_box"] = has_org
        row_out["has_exuviae_box"] = has_exu
        row_out["both_boxes"] = int(has_org == 1 and has_exu == 1)
        row_out["only_exuviae"] = int(has_exu == 1 and has_org == 0)
        row_out["only_organism"] = int(has_org == 1 and has_exu == 0)
        row_out["has_suture"] = int(pd.notna(row.get("x_suture")) and pd.notna(row.get("y_suture")))

        # -----------------------------
        # Image-level metadata
        # -----------------------------
        row_out["image_width"] = image_width
        row_out["image_height"] = image_height
        row_out["image_area"] = image_area
        row_out["image_diag"] = image_diag

        # -----------------------------
        # Raw geometric features
        # -----------------------------
        org_area = get_box_area(row, "organism")
        exu_area = get_box_area(row, "exuviae")
        dist_centroids = compute_centroid_distance(row)
        dx_centroids, dy_centroids = compute_dx_dy_centroids(row)

        row_out["org_area"] = org_area
        row_out["exu_area"] = exu_area
        row_out["box_overlap"] = compute_box_overlap(row)
        row_out["overlap_over_exuviae"] = compute_intersection_over_exuviae(row)
        row_out["overlap_over_organism"] = compute_intersection_over_organism(row)
        row_out["size_ratio"] = compute_size_ratio(row)
        row_out["dist_centroids"] = dist_centroids
        row_out["dx_centroids"] = dx_centroids
        row_out["dy_centroids"] = dy_centroids
        row_out["abs_dx_centroids"] = abs(dx_centroids) if pd.notna(dx_centroids) else np.nan
        row_out["abs_dy_centroids"] = abs(dy_centroids) if pd.notna(dy_centroids) else np.nan

        # -----------------------------
        # Normalized geometric features
        # -----------------------------
        if pd.notna(image_width) and image_width > 0:
            row_out["x_organism_norm"] = float(row["x_organism"]) / image_width if pd.notna(row["x_organism"]) else np.nan
            row_out["w_organism_norm"] = float(row["w_organism"]) / image_width if pd.notna(row["w_organism"]) else np.nan
            row_out["x_exuviae_norm"] = float(row["x_exuviae"]) / image_width if pd.notna(row["x_exuviae"]) else np.nan
            row_out["w_exuviae_norm"] = float(row["w_exuviae"]) / image_width if pd.notna(row["w_exuviae"]) else np.nan
            row_out["x_suture_norm"] = float(row["x_suture"]) / image_width if pd.notna(row["x_suture"]) else np.nan
            row_out["dx_centroids_norm"] = dx_centroids / image_width if pd.notna(dx_centroids) else np.nan
        else:
            row_out["x_organism_norm"] = np.nan
            row_out["w_organism_norm"] = np.nan
            row_out["x_exuviae_norm"] = np.nan
            row_out["w_exuviae_norm"] = np.nan
            row_out["x_suture_norm"] = np.nan
            row_out["dx_centroids_norm"] = np.nan

        if pd.notna(image_height) and image_height > 0:
            row_out["y_organism_norm"] = float(row["y_organism"]) / image_height if pd.notna(row["y_organism"]) else np.nan
            row_out["h_organism_norm"] = float(row["h_organism"]) / image_height if pd.notna(row["h_organism"]) else np.nan
            row_out["y_exuviae_norm"] = float(row["y_exuviae"]) / image_height if pd.notna(row["y_exuviae"]) else np.nan
            row_out["h_exuviae_norm"] = float(row["h_exuviae"]) / image_height if pd.notna(row["h_exuviae"]) else np.nan
            row_out["y_suture_norm"] = float(row["y_suture"]) / image_height if pd.notna(row["y_suture"]) else np.nan
            row_out["dy_centroids_norm"] = dy_centroids / image_height if pd.notna(dy_centroids) else np.nan
        else:
            row_out["y_organism_norm"] = np.nan
            row_out["h_organism_norm"] = np.nan
            row_out["y_exuviae_norm"] = np.nan
            row_out["h_exuviae_norm"] = np.nan
            row_out["y_suture_norm"] = np.nan
            row_out["dy_centroids_norm"] = np.nan

        row_out["org_area_norm"] = org_area / image_area if pd.notna(org_area) and pd.notna(image_area) and image_area > 0 else np.nan
        row_out["exu_area_norm"] = exu_area / image_area if pd.notna(exu_area) and pd.notna(image_area) and image_area > 0 else np.nan
        row_out["dist_centroids_norm"] = dist_centroids / image_diag if pd.notna(dist_centroids) and pd.notna(image_diag) and image_diag > 0 else np.nan

        # -----------------------------
        # Relative spatial features
        # -----------------------------
        row_out["exuvia_left_of_org"] = int(pd.notna(dx_centroids) and dx_centroids < 0)
        row_out["exuvia_right_of_org"] = int(pd.notna(dx_centroids) and dx_centroids > 0)
        row_out["exuvia_above_org"] = int(pd.notna(dy_centroids) and dy_centroids < 0)
        row_out["exuvia_below_org"] = int(pd.notna(dy_centroids) and dy_centroids > 0)

        # -----------------------------
        # Suture distances
        # -----------------------------
        row_out["dist_suture_exuviae"] = compute_point_to_box_center_distance(
            row.get("x_suture"), row.get("y_suture"), row, "exuviae"
        )
        row_out["dist_suture_organism"] = compute_point_to_box_center_distance(
            row.get("x_suture"), row.get("y_suture"), row, "organism"
        )

        row_out["dist_suture_exuviae_norm"] = (
            row_out["dist_suture_exuviae"] / image_diag
            if pd.notna(row_out["dist_suture_exuviae"]) and pd.notna(image_diag) and image_diag > 0
            else np.nan
        )
        row_out["dist_suture_organism_norm"] = (
            row_out["dist_suture_organism"] / image_diag
            if pd.notna(row_out["dist_suture_organism"]) and pd.notna(image_diag) and image_diag > 0
            else np.nan
        )

        # -----------------------------
        # Color-difference features
        # -----------------------------
        row_out["delta_mean_r"] = (
            row_out["ex_mean_r"] - row_out["org_mean_r"]
            if pd.notna(row_out["ex_mean_r"]) and pd.notna(row_out["org_mean_r"])
            else np.nan
        )
        row_out["delta_mean_g"] = (
            row_out["ex_mean_g"] - row_out["org_mean_g"]
            if pd.notna(row_out["ex_mean_g"]) and pd.notna(row_out["org_mean_g"])
            else np.nan
        )
        row_out["delta_mean_b"] = (
            row_out["ex_mean_b"] - row_out["org_mean_b"]
            if pd.notna(row_out["ex_mean_b"]) and pd.notna(row_out["org_mean_b"])
            else np.nan
        )
        row_out["delta_mean_gray"] = (
            row_out["ex_mean_gray"] - row_out["org_mean_gray"]
            if pd.notna(row_out["ex_mean_gray"]) and pd.notna(row_out["org_mean_gray"])
            else np.nan
        )
        row_out["delta_std_rgb"] = (
            row_out["ex_std_rgb"] - row_out["org_std_rgb"]
            if pd.notna(row_out["ex_std_rgb"]) and pd.notna(row_out["org_std_rgb"])
            else np.nan
        )

        rows_out.append(row_out)

    df_out = pd.DataFrame(rows_out)

    numeric_candidate_columns = [
        "x_suture", "y_suture",
        "x_exuviae", "y_exuviae", "w_exuviae", "h_exuviae",
        "x_organism", "y_organism", "w_organism", "h_organism",
        "x_head_org", "y_head_org", "x_thorax_org", "y_thorax_org",
        "x_head_exu", "y_head_exu", "x_thorax_exu", "y_thorax_exu",
        "exu_ref",
        "image_width", "image_height", "image_area", "image_diag",
        "org_area", "exu_area",
        "box_overlap", "overlap_over_exuviae", "overlap_over_organism",
        "size_ratio", "dist_centroids", "dx_centroids", "dy_centroids",
        "abs_dx_centroids", "abs_dy_centroids",
        "x_organism_norm", "y_organism_norm", "w_organism_norm", "h_organism_norm",
        "x_exuviae_norm", "y_exuviae_norm", "w_exuviae_norm", "h_exuviae_norm",
        "x_suture_norm", "y_suture_norm",
        "dx_centroids_norm", "dy_centroids_norm", "dist_centroids_norm",
        "org_area_norm", "exu_area_norm",
        "dist_suture_exuviae", "dist_suture_organism",
        "dist_suture_exuviae_norm", "dist_suture_organism_norm",
        "ex_mean_r", "ex_mean_g", "ex_mean_b", "ex_std_rgb", "ex_var_gray", "ex_mean_gray",
        "org_mean_r", "org_mean_g", "org_mean_b", "org_std_rgb", "org_var_gray", "org_mean_gray",
        "delta_mean_r", "delta_mean_g", "delta_mean_b", "delta_mean_gray", "delta_std_rgb",
        "has_organism_box", "has_exuviae_box", "both_boxes", "only_exuviae", "only_organism",
        "has_suture",
        "exuvia_left_of_org", "exuvia_right_of_org", "exuvia_above_org", "exuvia_below_org",
        "taxon_group_Chelicerata", "taxon_group_Crustacea", "taxon_group_Hexapoda", "taxon_group_Myriapoda",
    ]

    for col in numeric_candidate_columns:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print("\n[✓] Annotated feature table created successfully.")
    print(f"[INFO] Input CSV:  {INPUT_CSV}")
    print(f"[INFO] Output CSV: {OUTPUT_CSV}")
    print(f"[INFO] Rows saved:  {len(df_out)}")
    print(f"[INFO] Columns:     {len(df_out.columns)}")


# ============================================================
# Main / entry point
# ============================================================
if __name__ == "__main__":
    build_annotated_features()