
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "match_features_advanced.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "pitchiq_model.joblib"

C = 0.01

METRICS = [
    "Points",
    "GoalsFor",
    "GoalsAgainst",
    "xGFor",
    "xGAgainst",
]

WINDOWS = [3, 5, 10]


# ============================================================
# DEFINE MODEL FEATURES
# ============================================================

OVERALL_FEATURES = [
    f"Diff_Overall_{metric}_Last{window}"
    for window in WINDOWS
    for metric in METRICS
]

VENUE_FEATURES = [
    f"Diff_Venue_{metric}_Last{window}"
    for window in WINDOWS
    for metric in METRICS
]

FEATURES = OVERALL_FEATURES + VENUE_FEATURES


# ============================================================
# LOAD AND VALIDATE TRAINING DATA
# ============================================================

def load_training_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}\n"
            "Run scripts/build_advanced_features.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    required_columns = [
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FTR",
    ] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing training columns: {missing_columns}"
        )

    # Sort matches chronologically.
    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    # Check for missing training values.
    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in training data."
        )

    # Validate match outcomes.
    valid_results = ["H", "D", "A"]

    if not df["FTR"].isin(valid_results).all():
        raise ValueError(
            "Invalid match outcomes detected."
        )

    # Check for duplicate fixtures.
    if df.duplicated(
        ["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError(
            "Duplicate fixtures detected."
        )

    # Ensure the model has all three outcome classes.
    if set(df["FTR"].unique()) != set(valid_results):
        raise ValueError(
            "Training data must contain home wins, "
            "draws and away wins."
        )

    if len(FEATURES) != 30:
        raise ValueError(
            f"Expected 30 features, found {len(FEATURES)}."
        )

    return df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                C=C,
                max_iter=3000,
                random_state=42,
            )
        ),
    ])

    return model


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def main():

    print("=" * 65)
    print("PITCHIQ - FINAL MODEL TRAINING")
    print("=" * 65)

    # 1. Load data.
    df = load_training_data()

    print("\nDATASET INFORMATION")
    print("Total matches:", len(df))
    print("Total seasons:", df["Season"].nunique())

    print("\nMatches per season:")
    print(df.groupby("Season").size())

    print("\nMatch outcomes:")
    print(df["FTR"].value_counts())

    # 2. Prepare features and target.
    X = df[FEATURES]
    y = df["FTR"]

    # 3. Build and train model.
    model = build_model()

    print("\nTraining Logistic Regression model...")

    model.fit(X, y)

    # 4. Prepare model metadata.
    artifact = {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "regularisation_C": C,
        "training_matches": len(df),
        "last_training_date": str(
            df["Date"].max().date()
        ),
    }

    # 5. Save trained model.
    # Use a temporary file to avoid replacing the existing
    # model if serialization fails.
    temporary_file = MODEL_DIR / "pitchiq_model.tmp"

    joblib.dump(artifact, temporary_file)
    temporary_file.replace(MODEL_FILE)

    # 6. Print training summary.
    print("\n" + "=" * 65)
    print("TRAINING SUMMARY")
    print("=" * 65)

    print("Training matches:", len(df))
    print("Features:", len(FEATURES))
    print("Regularisation C:", C)
    print(
        "Last training date:",
        artifact["last_training_date"]
    )
    print("Outcome classes:", artifact["classes"])

    print("\nModel saved to:")
    print(MODEL_FILE)

    print("\nFINAL MODEL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
