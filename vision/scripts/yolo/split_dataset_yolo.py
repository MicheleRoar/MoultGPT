import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# === PATH FISSI ===
CSV_PATH = "../../data/inat_raw/inat_dataset.csv"
IMG_ROOT = "../../data/inat_raw"
OUT_DIR  = "../../data/yolo"
VAL_RATIO = 0.2
SEED = 42

# Classi YOLO
CLASS_MAP = {"organism": 0, "exuviae": 1}


def has_box(row, prefix):
    return (
        pd.notna(row.get(f"x_{prefix}")) and
        pd.notna(row.get(f"y_{prefix}")) and
        pd.notna(row.get(f"w_{prefix}")) and
        pd.notna(row.get(f"h_{prefix}")) and
        float(row.get(f"w_{prefix}", 0)) > 0 and
        float(row.get(f"h_{prefix}", 0)) > 0
    )


def to_yolo_line(cls_id, x, y, w, h, W, H):
    xc = (x + w / 2.0) / W
    yc = (y + h / 2.0) / H
    wn = w / W
    hn = h / H
    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def main():
    df = pd.read_csv(CSV_PATH)

    # Flag box
    df["has_exuviae"]  = df.apply(lambda r: has_box(r, "exuviae"), axis=1)
    df["has_organism"] = df.apply(lambda r: has_box(r, "organism"), axis=1)

    # Tieni solo immagini con almeno 1 box
    df = df[df["has_exuviae"] | df["has_organism"]].copy()

    # Stratificazione per stage
    train_idx, val_idx = train_test_split(
        df.index, test_size=VAL_RATIO, random_state=SEED, stratify=df["stage"]
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"]   = "val"

    # Crea cartelle YOLO
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    n_images, n_labels, missing = 0, 0, 0

    for _, row in df.iterrows():
        stage    = row["stage"]
        obs_id   = row["observation_id"]
        photo_id = row["photo_id"]
        fname    = f"{obs_id}_{photo_id}.jpg"
        src_img  = os.path.join(IMG_ROOT, stage, fname)
        if not os.path.exists(src_img):
            missing += 1
            continue

        split   = row["split"]
        dst_img = os.path.join(OUT_DIR, "images", split, fname)
        dst_lbl = os.path.join(OUT_DIR, "labels", split, fname.replace(".jpg", ".txt"))

        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        with Image.open(src_img) as im:
            W, H = im.size

        lines = []
        if row["has_organism"]:
            x, y, w, h = float(row["x_organism"]), float(row["y_organism"]), float(row["w_organism"]), float(row["h_organism"])
            lines.append(to_yolo_line(CLASS_MAP["organism"], x, y, w, h, W, H))
        if row["has_exuviae"]:
            x, y, w, h = float(row["x_exuviae"]), float(row["y_exuviae"]), float(row["w_exuviae"]), float(row["h_exuviae"])
            lines.append(to_yolo_line(CLASS_MAP["exuviae"], x, y, w, h, W, H))

        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))
        n_images += 1
        n_labels += len(lines)

    # Crea data.yaml
    data_yaml = f"""# Auto-generated
path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val

names:
  0: organism
  1: exuviae
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(data_yaml)

    # Summary
    print("\nExport YOLO finito.")
    print("Immagini esportate:", n_images, "| Mancanti:", missing)
    print("Label lines totali:", n_labels)
    print("Dataset YOLO root:", os.path.abspath(OUT_DIR))
    print("data.yaml:", os.path.join(OUT_DIR, "data.yaml"))

    # Extra: riepilogo
    imgs_by_stage_split = df.groupby(["split", "stage"]).size().unstack(fill_value=0)
    boxes = (
        df.assign(exuviae=df["has_exuviae"].astype(int),
                  organism=df["has_organism"].astype(int))
          .groupby("split")[["organism","exuviae"]]
          .sum()
    )
    print("\nImmagini per split & stage:")
    print(imgs_by_stage_split)
    print("\nBox counts per split & class:")
    print(boxes)


if __name__ == "__main__":
    main()
