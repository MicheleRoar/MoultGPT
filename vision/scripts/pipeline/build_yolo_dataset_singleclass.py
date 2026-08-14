"""Same as build_yolo_dataset_grouped.py, but organism and exuviae are
merged into one class ("arthropod_object"). Tests whether asking YOLO to
just localize (not also classify) recovers recall -- a simpler task for a
model fine-tuned on ~500 grouped images.
"""

import os
import shutil

import pandas as pd
from PIL import Image

CSV_PATH = "../../data/inat_raw/inat_dataset.csv"
SPLIT_PATH = "../../data/master_split.csv"
IMG_ROOT = "../../data/inat_raw"
OUT_DIR = "../../data/yolo_grouped_singleclass"


def has_box(row, prefix):
    for k in ("x", "y", "w", "h"):
        if pd.isna(row.get(f"{k}_{prefix}")):
            return False
    return float(row[f"w_{prefix}"]) > 0 and float(row[f"h_{prefix}"]) > 0


def to_yolo_line(x, y, w, h, W, H):
    xc, yc = (x + w / 2) / W, (y + h / 2) / H
    return f"0 {xc:.6f} {yc:.6f} {w/W:.6f} {h/H:.6f}"  # single class id


def main():
    df = pd.read_csv(CSV_PATH)
    split = pd.read_csv(SPLIT_PATH)
    df["has_organism"] = df.apply(lambda r: has_box(r, "organism"), axis=1)
    df["has_exuviae"] = df.apply(lambda r: has_box(r, "exuviae"), axis=1)
    df = df[df["has_organism"] | df["has_exuviae"]].merge(split, on="observation_id", how="inner")

    for s in ("train", "val"):
        os.makedirs(f"{OUT_DIR}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{s}", exist_ok=True)

    n_images, n_labels, missing = 0, 0, 0
    for _, row in df.iterrows():
        fname = f"{row['observation_id']}_{row['photo_id']}.jpg"
        src_img = os.path.join(IMG_ROOT, row["stage"], fname)
        if not os.path.exists(src_img):
            missing += 1
            continue

        dst_img = f"{OUT_DIR}/images/{row['split']}/{fname}"
        dst_lbl = f"{OUT_DIR}/labels/{row['split']}/{fname.replace('.jpg', '.txt')}"
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        with Image.open(src_img) as im:
            W, H = im.size
        lines = []
        if row["has_organism"]:
            lines.append(to_yolo_line(row["x_organism"], row["y_organism"], row["w_organism"], row["h_organism"], W, H))
        if row["has_exuviae"]:
            lines.append(to_yolo_line(row["x_exuviae"], row["y_exuviae"], row["w_exuviae"], row["h_exuviae"], W, H))
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))
        n_images += 1
        n_labels += len(lines)

    with open(f"{OUT_DIR}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(OUT_DIR)}\ntrain: images/train\nval: images/val\n\n"
                 "names:\n  0: arthropod_object\n")

    print(f"images={n_images} missing={missing} label_lines={n_labels}")
    print(f"data.yaml -> {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
