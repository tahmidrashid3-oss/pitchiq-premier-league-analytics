
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    required = ["FTR"] + FEATURES

    if df[required].isna().any().any():
        raise ValueError(
            "Missing training values detected."
        )

    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Invalid match outcomes.")

    # Keep the model consistent with our chosen
    # Advanced Venue configuration.
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=C,
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])

    model.fit(
        df[FEATURES],
        df["FTR"],
    )

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

    joblib.dump(artifact, MODEL_FILE)

    print("=" * 65)
    print("PITCHIQ — FINAL MODEL TRAINING")
    print("=" * 65)
    print("Training matches:", len(df))
    print("Features:", len(FEATURES))
    print("Regularisation C:", C)
    print("Last training date:", artifact["last_training_date"])
    print("Model saved to:", MODEL_FILE)
    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()
