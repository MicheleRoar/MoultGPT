import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# === CONFIGURATION ===
CSV_PATH = "../../data/annotated_dataset.csv"
MODEL_PATH = "../../models/xgboost_stage.pkl"
ENCODER_PATH = "../../models/label_encoder.pkl"
FEATURE_PLOT = "../results/plots/xgboost_feature_importance.png"
SEED = 42
TEST_SIZE = 0.2
EXCLUDE_STAGE = "pre-moult"  # Optional: class to exclude from training

# === LOAD DATASET ===
df = pd.read_csv(CSV_PATH)

if EXCLUDE_STAGE:
    df = df[df["stage"] != EXCLUDE_STAGE]

# === ADD FEATURES ON-THE-FLY ===
has_organism = df["x_organism"].notna().astype(int)
has_exuviae = df["x_exuviae"].notna().astype(int)
df["only_exuviae"] = ((has_exuviae == 1) & (has_organism == 0)).astype(int)

# === SELECT FEATURES TO USE ===
FEATURES = [
    "box_overlap", "dist_centroids", "x_organism", "y_organism",
    "x_exuviae", "y_exuviae", "h_exuviae",
    "org_mean_g", "org_mean_gray",
    "taxon_group_Crustacea", "taxon_group_Hexapoda",
    "taxon_group_Chelicerata", "taxon_group_Myriapoda",
    "only_exuviae"
]

X = df[FEATURES].copy()
X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["stage"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# === TRAIN XGBOOST CLASSIFIER ===
model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(label_encoder.classes_),
    use_label_encoder=False,
    random_state=SEED,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# === EVALUATION ===
y_pred = model.predict(X_val)
print("\n=== Classification Report ===")
print(classification_report(label_encoder.inverse_transform(y_val), label_encoder.inverse_transform(y_pred)))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(label_encoder.inverse_transform(y_val), label_encoder.inverse_transform(y_pred)))

# === SAVE MODEL AND ENCODER ===
os.makedirs("../models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print(f"Model saved to {MODEL_PATH}")
print(f"Label encoder saved to {ENCODER_PATH}")

# === PLOT FEATURE IMPORTANCE ===
importance = model.get_booster().get_score(importance_type='gain')
imp_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=imp_df)
plt.tight_layout()
plt.title("XGBoost Slim Feature Importance")
plt.savefig(FEATURE_PLOT, dpi=300)
print(f"Feature importance plot saved to {FEATURE_PLOT}")
