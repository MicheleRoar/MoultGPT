#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    StratifiedGroupKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = "f1_macro"
EXCLUDE_STAGE = "pre-moult"

# IMPORTANT:
# - "structural" is diagnostic and can be close to tautological
# - "realistic" is the one to use for honest benchmarking
FEATURE_SET = "realistic"   # "realistic" or "structural"

# Safety net only at prediction time
USE_BIOLOGICAL_CONSTRAINT = False

# Use biological filtering before training
USE_BIOLOGICAL_FILTER = True

# Overlap threshold for class definition during filtering
OVERLAP_THRESHOLD = 0.01

# Use grouped split/CV by observation_id when available
USE_GROUP_SPLIT = True

BASE_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = BASE_DIR / "data" / "annotated_features.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "scripts" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_SUFFIX = FEATURE_SET

BEST_MODEL_PATH = MODELS_DIR / f"best_stage_classifier_{RUN_SUFFIX}.pkl"
ENCODER_PATH = MODELS_DIR / f"stage_label_encoder_{RUN_SUFFIX}.pkl"
SUMMARY_CSV = RESULTS_DIR / f"classifier_benchmark_summary_{RUN_SUFFIX}.csv"
CLASSIFICATION_REPORT_JSON = RESULTS_DIR / f"best_model_classification_report_{RUN_SUFFIX}.json"
CONFUSION_MATRIX_PNG = PLOTS_DIR / f"best_model_confusion_matrix_{RUN_SUFFIX}.png"
XGB_IMPORTANCE_PNG = PLOTS_DIR / f"xgboost_feature_importance_{RUN_SUFFIX}.png"


# =========================
# FEATURE SETS
# =========================
FEATURES_STRUCTURAL = [
    # box presence logic
    "has_organism_box",
    "has_exuviae_box",
    "both_boxes",
    "only_exuviae",
    "only_organism",

    # overlap / geometry strongly tied to class definition
    "box_overlap",
    "overlap_over_exuviae",
    "overlap_over_organism",
    "size_ratio",

    # normalized position / size
    "x_organism_norm",
    "y_organism_norm",
    "w_organism_norm",
    "h_organism_norm",
    "x_exuviae_norm",
    "y_exuviae_norm",
    "w_exuviae_norm",
    "h_exuviae_norm",

    # normalized area / distances
    "org_area_norm",
    "exu_area_norm",
    "dist_centroids_norm",
    "dx_centroids_norm",
    "dy_centroids_norm",
    "abs_dx_centroids",
    "abs_dy_centroids",

    # directional relations
    "exuvia_left_of_org",
    "exuvia_right_of_org",
    "exuvia_above_org",
    "exuvia_below_org",

    # color stats
    "org_mean_gray",
    "org_mean_g",
    "org_std_rgb",
    "ex_mean_gray",
    "ex_mean_g",
    "ex_std_rgb",

    # color differences
    "delta_mean_gray",
    "delta_mean_g",
    "delta_std_rgb",

    # taxon
    "taxon_group_Crustacea",
    "taxon_group_Hexapoda",
    "taxon_group_Chelicerata",
    "taxon_group_Myriapoda",
]

FEATURES_REALISTIC = [
    # normalized position / size
    "x_organism_norm",
    "y_organism_norm",
    "w_organism_norm",
    "h_organism_norm",
    "x_exuviae_norm",
    "y_exuviae_norm",
    "w_exuviae_norm",
    "h_exuviae_norm",

    # normalized areas / relative distances
    "org_area_norm",
    "exu_area_norm",
    "dist_centroids_norm",
    "dx_centroids_norm",
    "dy_centroids_norm",
    "abs_dx_centroids",
    "abs_dy_centroids",

    # directional relations
    "exuvia_left_of_org",
    "exuvia_right_of_org",
    "exuvia_above_org",
    "exuvia_below_org",

    # colors
    "org_mean_gray",
    "org_mean_g",
    "org_std_rgb",
    "ex_mean_gray",
    "ex_mean_g",
    "ex_std_rgb",

    # color contrast
    "delta_mean_gray",
    "delta_mean_g",
    "delta_std_rgb",

    # taxon
    "taxon_group_Crustacea",
    "taxon_group_Hexapoda",
    "taxon_group_Chelicerata",
    "taxon_group_Myriapoda",
]

if FEATURE_SET == "structural":
    FEATURES = FEATURES_STRUCTURAL
elif FEATURE_SET == "realistic":
    FEATURES = FEATURES_REALISTIC
else:
    raise ValueError(f"Unknown FEATURE_SET: {FEATURE_SET}")


# =========================
# DATA LOADING / FILTERING
# =========================
def filter_biological_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply biologically grounded class definitions.

    exuviae:
        - exuvia present
        - organism absent

    moulting:
        - exuvia present
        - organism present
        - attached / overlapping

    post-moult:
        - organism present
        - exuvia absent OR detached / non-overlapping
    """
    exuviae_df = df[
        (df["stage"] == "exuviae") &
        (df["has_exuviae_box"] == 1) &
        (df["has_organism_box"] == 0)
    ]

    moulting_df = df[
        (df["stage"] == "moulting") &
        (df["has_exuviae_box"] == 1) &
        (df["has_organism_box"] == 1) &
        (df["box_overlap"] > OVERLAP_THRESHOLD)
    ]

    post_moult_df = df[
        (df["stage"] == "post-moult") &
        (df["has_organism_box"] == 1) &
        (
            (df["has_exuviae_box"] == 0) |
            (df["box_overlap"] <= OVERLAP_THRESHOLD)
        )
    ]

    filtered = pd.concat(
        [exuviae_df, moulting_df, post_moult_df],
        axis=0
    ).reset_index(drop=True)

    return filtered


def load_dataset(csv_path: Path):
    df_raw = pd.read_csv(csv_path)

    if EXCLUDE_STAGE:
        df_raw = df_raw[df_raw["stage"] != EXCLUDE_STAGE].copy()

    required_columns = FEATURES + ["stage"]
    extra_needed = ["has_organism_box", "has_exuviae_box", "box_overlap"]
    if USE_BIOLOGICAL_FILTER:
        required_columns += extra_needed

    missing = [c for c in required_columns if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df_raw.copy()

    if USE_BIOLOGICAL_FILTER:
        df = filter_biological_consistency(df)

        print("\n[INFO] Dataset after biological filtering:")
        print(df["stage"].value_counts())

        print("\n[INFO] Sanity check by stage:")
        sanity_cols = ["has_organism_box", "has_exuviae_box", "box_overlap"]
        print(df.groupby("stage")[sanity_cols].mean().round(4))

    return df_raw, df


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["stage"])

    if "observation_id" in df.columns:
        groups = df["observation_id"].fillna(-1).astype(str).values
    else:
        groups = None

    return X, y, label_encoder, groups


# =========================
# PREPROCESSORS
# =========================
def make_scaled_numeric_preprocessor(feature_names):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                feature_names
            )
        ],
        remainder="drop"
    )


def make_impute_only_preprocessor(feature_names):
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_names)
        ],
        remainder="drop"
    )


# =========================
# MODELS
# =========================
def get_model_search_spaces(seed: int):
    models = {}

    scaled_preprocessor = make_scaled_numeric_preprocessor(FEATURES)
    impute_only_preprocessor = make_impute_only_preprocessor(FEATURES)

    models["LogisticRegression"] = {
        "pipeline": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("clf", LogisticRegression(
                random_state=seed,
                max_iter=5000,
                multi_class="auto"
            ))
        ]),
        "params": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"]
        }
    }

    models["SVM_RBF"] = {
        "pipeline": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                random_state=seed
            ))
        ]),
        "params": {
            "clf__C": [0.5, 1.0, 5.0],
            "clf__gamma": ["scale", 0.1, 0.01],
            "clf__class_weight": [None, "balanced"]
        }
    }

    models["RandomForest"] = {
        "pipeline": Pipeline([
            ("preprocess", impute_only_preprocessor),
            ("clf", RandomForestClassifier(
                random_state=seed,
                n_jobs=-1
            ))
        ]),
        "params": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": [None, "balanced"]
        }
    }

    models["XGBoost"] = {
        "pipeline": Pipeline([
            ("preprocess", impute_only_preprocessor),
            ("clf", XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=seed,
                tree_method="hist",
                n_jobs=-1
            ))
        ]),
        "params": {
            "clf__n_estimators": [150, 250, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 4, 6],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0]
        }
    }

    return models


# =========================
# SPLITTING
# =========================
def grouped_train_val_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Group-aware train/validation split using observation_id when available.
    Falls back to standard stratified split if groups are not usable.
    """
    if groups is None or len(np.unique(groups)) == len(groups) == 0:
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        ) + (None, None)

    splitter = StratifiedGroupKFold(
        n_splits=int(round(1 / test_size)),
        shuffle=True,
        random_state=random_state
    )

    splits = list(splitter.split(X, y, groups))
    train_idx, val_idx = splits[0]

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    groups_train = groups[train_idx]
    groups_val = groups[val_idx]

    return X_train, X_val, y_train, y_val, groups_train, groups_val


# =========================
# BIOLOGICAL CONSTRAINT
# =========================
def apply_biological_constraint_from_proba(model, X_raw, label_encoder):
    """
    Safety rule:
    if has_organism_box == 1, class 'exuviae' is not allowed.

    Only works if that column exists in X_raw.
    """
    if not hasattr(model, "predict_proba"):
        return model.predict(X_raw)

    if "has_organism_box" not in X_raw.columns:
        return model.predict(X_raw)

    proba = model.predict_proba(X_raw)
    classes = np.array(label_encoder.classes_)

    if "exuviae" not in classes:
        return np.argmax(proba, axis=1)

    ex_idx = np.where(classes == "exuviae")[0][0]
    has_org = X_raw["has_organism_box"].fillna(0).astype(int).values

    adjusted_preds = []

    for i in range(len(X_raw)):
        p = proba[i].copy()

        if has_org[i] == 1:
            p[ex_idx] = 0.0
            if p.sum() > 0:
                p = p / p.sum()

        adjusted_preds.append(int(np.argmax(p)))

    return np.array(adjusted_preds)


# =========================
# TRAINING / BENCHMARK
# =========================
def benchmark_models(X_train, y_train, X_val, y_val, label_encoder, groups_train=None):
    model_spaces = get_model_search_spaces(SEED)

    if USE_GROUP_SPLIT and groups_train is not None:
        cv = StratifiedGroupKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=SEED
        )
        cv_splitter = list(cv.split(X_train, y_train, groups_train))
        use_groups_in_fit = True
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        cv_splitter = cv
        use_groups_in_fit = False

    summary_rows = []
    fitted_models = {}

    for model_name, spec in model_spaces.items():
        print(f"\n=== Benchmarking {model_name} ===")

        grid = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["params"],
            scoring=SCORING,
            cv=cv_splitter,
            n_jobs=-1,
            refit=True,
            verbose=1
        )

        if use_groups_in_fit:
            grid.fit(X_train, y_train)
        else:
            grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        if USE_BIOLOGICAL_CONSTRAINT:
            y_pred = apply_biological_constraint_from_proba(
                best_model, X_val, label_encoder
            )
        else:
            y_pred = best_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")

        row = {
            "model": model_name,
            "feature_set": FEATURE_SET,
            "cv_best_macro_f1": grid.best_score_,
            "val_accuracy": acc,
            "val_balanced_accuracy": bal_acc,
            "val_macro_f1": macro_f1,
            "biological_filter": USE_BIOLOGICAL_FILTER,
            "biological_constraint": USE_BIOLOGICAL_CONSTRAINT,
            "group_split": USE_GROUP_SPLIT and (groups_train is not None),
            "best_params": grid.best_params_
        }

        summary_rows.append(row)
        fitted_models[model_name] = best_model

        print(f"Best params: {grid.best_params_}")
        print(f"Validation accuracy: {acc:.4f}")
        print(f"Validation balanced accuracy: {bal_acc:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="val_macro_f1", ascending=False
    ).reset_index(drop=True)

    best_name = summary_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    if USE_BIOLOGICAL_CONSTRAINT:
        y_pred_best = apply_biological_constraint_from_proba(
            best_model, X_val, label_encoder
        )
    else:
        y_pred_best = best_model.predict(X_val)

    report = classification_report(
        y_val,
        y_pred_best,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    return summary_df, best_name, best_model, y_pred_best, report


# =========================
# PLOTS / SAVING
# =========================
def save_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {FEATURE_SET}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_xgb_feature_importance(best_model, out_path: Path):
    clf = best_model.named_steps["clf"]

    if not isinstance(clf, XGBClassifier):
        return

    preprocess = best_model.named_steps["preprocess"]

    try:
        transformed_feature_names = preprocess.get_feature_names_out()
    except Exception:
        transformed_feature_names = FEATURES

    importance = clf.feature_importances_
    if importance is None or len(importance) == 0:
        return

    imp_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    imp_df["feature"] = (
        imp_df["feature"]
        .astype(str)
        .str.replace("num__", "", regex=False)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=imp_df.head(20))
    plt.title(f"XGBoost Feature Importance - {FEATURE_SET}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Loading dataset from: {CSV_PATH}")
    print(f"[INFO] Feature set: {FEATURE_SET}")
    print(f"[INFO] Overlap threshold: {OVERLAP_THRESHOLD}")
    print(f"[INFO] Biological filter: {USE_BIOLOGICAL_FILTER}")
    print(f"[INFO] Biological constraint: {USE_BIOLOGICAL_CONSTRAINT}")
    print(f"[INFO] Group split: {USE_GROUP_SPLIT}")

    df_raw, df = load_dataset(CSV_PATH)

    print(f"\n[INFO] Rows before filtering: {len(df_raw)}")
    print(f"[INFO] Rows after filtering:  {len(df)}")
    print(f"[INFO] Rows removed: {len(df_raw) - len(df)}")

    print("\n[INFO] Class counts before filtering:")
    print(df_raw["stage"].value_counts())

    print("\n[INFO] Class counts after filtering:")
    print(df["stage"].value_counts())

    X, y, label_encoder, groups = prepare_xy(df)

    print(f"\n[INFO] Dataset shape after filtering: {X.shape}")
    print(f"[INFO] Classes: {list(label_encoder.classes_)}")
    print(f"[INFO] Number of features: {len(FEATURES)}")

    if USE_GROUP_SPLIT and groups is not None:
        n_groups = len(np.unique(groups))
        print(f"[INFO] Unique observation groups: {n_groups}")
        X_train, X_val, y_train, y_val, groups_train, groups_val = grouped_train_val_split(
            X, y, groups, test_size=TEST_SIZE, random_state=SEED
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=SEED
        )
        groups_train = None
        groups_val = None

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Val shape:   {X_val.shape}")

    summary_df, best_name, best_model, y_pred_best, report = benchmark_models(
        X_train, y_train, X_val, y_val, label_encoder, groups_train=groups_train
    )

    print("\n=== Benchmark summary ===")
    print(summary_df[[
        "model",
        "feature_set",
        "cv_best_macro_f1",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "biological_filter",
        "biological_constraint",
        "group_split"
    ]])

    summary_to_save = summary_df.copy()
    summary_to_save["best_params"] = summary_to_save["best_params"].astype(str)
    summary_to_save.to_csv(SUMMARY_CSV, index=False)

    with open(CLASSIFICATION_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    save_confusion_matrix(
        y_val,
        y_pred_best,
        label_encoder.classes_,
        CONFUSION_MATRIX_PNG
    )
    save_xgb_feature_importance(best_model, XGB_IMPORTANCE_PNG)

    print(f"\n[✓] Best model: {best_name}")
    print(f"[✓] Summary saved to: {SUMMARY_CSV}")
    print(f"[✓] Classification report saved to: {CLASSIFICATION_REPORT_JSON}")
    print(f"[✓] Best model saved to: {BEST_MODEL_PATH}")
    print(f"[✓] Label encoder saved to: {ENCODER_PATH}")
    print(f"[✓] Confusion matrix saved to: {CONFUSION_MATRIX_PNG}")
    print(f"[✓] XGBoost feature importance saved to: {XGB_IMPORTANCE_PNG}")


if __name__ == "__main__":
    main()