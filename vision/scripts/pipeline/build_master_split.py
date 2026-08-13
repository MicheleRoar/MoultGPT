"""Single source of truth for train/val membership, grouped by observation_id.

Used by both the YOLO dataset builder and the classifier retrain, so neither
can ever leak into the other's validation set. Replaces the old ungrouped
splits in split_dataset_yolo.py and train_xgboost.py.

Output: vision/data/master_split.csv (observation_id, split)
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CSV_PATH = "../../data/inat_raw/inat_dataset.csv"
OUT_PATH = "../../data/master_split.csv"
SEED = 42
N_SPLITS = 5  # fold 0 -> ~20% val


def has_box(row, prefix):
    for k in ("x", "y", "w", "h"):
        if pd.isna(row.get(f"{k}_{prefix}")):
            return False
    return float(row[f"w_{prefix}"]) > 0 and float(row[f"h_{prefix}"]) > 0


def main():
    df = pd.read_csv(CSV_PATH)
    df["has_organism"] = df.apply(lambda r: has_box(r, "organism"), axis=1)
    df["has_exuviae"] = df.apply(lambda r: has_box(r, "exuviae"), axis=1)
    pop = df[df["has_organism"] | df["has_exuviae"]].copy()

    obs = pop.groupby("observation_id")["stage"].agg(lambda s: s.mode()[0]).reset_index()

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_idx, val_idx = list(sgkf.split(obs, obs["stage"], obs["observation_id"]))[0]

    obs["split"] = "train"
    obs.loc[val_idx, "split"] = "val"
    obs[["observation_id", "split"]].to_csv(OUT_PATH, index=False)

    print(f"observations: {len(obs)}  train={len(train_idx)}  val={len(val_idx)}")
    print(obs.merge(pop.drop_duplicates("observation_id")[["observation_id"]], on="observation_id")
          .assign(stage=obs["stage"]).groupby(["split", "stage"]).size())
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
