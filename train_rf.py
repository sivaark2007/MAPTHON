# ============================================================
# LOCAL RANDOM FOREST
# SENTINEL-2 36-FEATURE CROP CLASSIFICATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)


# ============================================================
# 1. FILE PATH
# ============================================================

CSV_PATH = r"C:\Users\siva\Desktop\VIT\CNN\sentinel\tirunelveli_rf_36_features.csv"

MODEL_PATH = r"C:\Users\siva\Desktop\VIT\CNN\random_forest_crop_model.pkl"


# ============================================================
# 2. LOAD CSV
# ============================================================

print("\n==============================")
print("LOADING DATASET")
print("==============================")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. CHECK DATA
# ============================================================

print("\nClass distribution:")
print(df["class"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum().sum())


# ============================================================
# 4. SELECT 36 SENTINEL-2 FEATURES
# ============================================================

feature_columns = [

    # Q1
    "Q1_B2",
    "Q1_B3",
    "Q1_B4",
    "Q1_B5",
    "Q1_B8",
    "Q1_B11",
    "Q1_B12",
    "Q1_NDVI",
    "Q1_NDMI",

    # Q2
    "Q2_B2",
    "Q2_B3",
    "Q2_B4",
    "Q2_B5",
    "Q2_B8",
    "Q2_B11",
    "Q2_B12",
    "Q2_NDVI",
    "Q2_NDMI",

    # Q3
    "Q3_B2",
    "Q3_B3",
    "Q3_B4",
    "Q3_B5",
    "Q3_B8",
    "Q3_B11",
    "Q3_B12",
    "Q3_NDVI",
    "Q3_NDMI",

    # Q4
    "Q4_B2",
    "Q4_B3",
    "Q4_B4",
    "Q4_B5",
    "Q4_B8",
    "Q4_B11",
    "Q4_B12",
    "Q4_NDVI",
    "Q4_NDMI"
]


print("\nNumber of features:", len(feature_columns))


# ============================================================
# 5. CREATE X AND y
# ============================================================

X = df[feature_columns]

y = df["class"]


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y
)


print("\n==============================")
print("TRAIN / TEST SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 7. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(

    n_estimators=300,

    max_features="sqrt",

    random_state=42,

    class_weight="balanced",

    n_jobs=-1

)


print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")


rf.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = rf.predict(X_test)

y_probability = rf.predict_proba(X_test)


# ============================================================
# 9. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

kappa = cohen_kappa_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Kappa    : {kappa:.4f}"
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,
    y_pred,

    labels=[0, 1, 2]

)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred,

        labels=[0, 1, 2],

        target_names=[
            "Paddy",
            "Banana",
            "Other"
        ],

        digits=4

    )

)


# ============================================================
# 12. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Paddy",
        "Banana",
        "Other"
    ],

    yticklabels=[
        "Paddy",
        "Banana",
        "Other"
    ]

)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "rf_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance = rf.feature_importances_


importance_df = pd.DataFrame({

    "feature": feature_columns,

    "importance": importance

})


importance_df = importance_df.sort_values(

    "importance",

    ascending=False

)


print("\n==============================")
print("TOP FEATURES")
print("==============================")

print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# 14. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = importance_df.head(15)


plt.figure(
    figsize=(10, 7)
)

plt.barh(

    top_features["feature"][::-1],

    top_features["importance"][::-1]

)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Sentinel-2 Features"
)

plt.tight_layout()

plt.savefig(
    "rf_feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. SAVE MODEL
# ============================================================

joblib.dump(

    rf,

    MODEL_PATH

)


print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(
    MODEL_PATH
)


# ============================================================
# 16. TEST PROBABILITIES
# ============================================================

class_names = {
    0: "Paddy",
    1: "Banana",
    2: "Other"
}


print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")


for i in range(
    min(10, len(X_test))
):

    actual = class_names[
        int(y_test.iloc[i])
    ]

    predicted = class_names[
        int(y_pred[i])
    ]

    probabilities = y_probability[i]

    confidence = np.max(
        probabilities
    )

    print(
        f"Actual: {actual:8s} | "
        f"Predicted: {predicted:8s} | "
        f"Confidence: {confidence:.3f}"
    )


# ============================================================
# END
# ============================================================

print("\nDone.")