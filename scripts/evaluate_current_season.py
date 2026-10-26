
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "match_features_advanced.csv"
)

OUTPUT_DIR = ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEASON = "2627"
CLASSES = ["A", "D", "H"]
C = 0.01

METRICS = [
    "Points",
    "GoalsFor",
    "GoalsAgainst",
    "xGFor",
    "xGAgainst",
]

WINDOWS = [3, 5, 10]

OVERALL = [
    f"Diff_Overall_{metric}_Last{window}"
    for window in WINDOWS
    for metric in METRICS
]

VENUE = [
    f"Diff_Venue_{metric}_Last{window}"
    for window in WINDOWS
    for metric in METRICS
]

FEATURES = OVERALL + VENUE


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    df["Season"] = df["Season"].str.zfill(4)

    required = [
        "Date", "Season", "HomeTeam",
        "AwayTeam", "FTR"
    ] + FEATURES

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected.")

    if df.duplicated(
        ["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError("Duplicate fixtures detected.")

    if not df["FTR"].isin(CLASSES).all():
        raise ValueError("Invalid match outcomes.")

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    train = df[df["Season"] < TEST_SEASON].copy()
    test = df[df["Season"] == TEST_SEASON].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test dataset is empty.")

    if set(train["FTR"]) != set(CLASSES):
        raise ValueError("Training set lacks an outcome class.")

    return train, test


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            max_iter=3000,
            random_state=42,
        )),
    ])


# ============================================================
# EVALUATE
# ============================================================

def main():

    print("=" * 65)
    print("PITCHIQ - 2026/27 MODEL EVALUATION")
    print("=" * 65)

    train, test = load_data()

    print("\nDATA SPLIT")
    print("Historical training matches:", len(train))
    print("2026/27 test matches:", len(test))
    print("Training cutoff:", train["Date"].max().date())
    print("Latest test match:", test["Date"].max().date())

    model = build_model()
    model.fit(train[FEATURES], train["FTR"])

    predictions = model.predict(test[FEATURES])

    probabilities = model.predict_proba(test[FEATURES])

    accuracy = accuracy_score(
        test["FTR"], predictions
    )

    loss = log_loss(
        test["FTR"],
        probabilities,
        labels=model.classes_,
    )

    # Multiclass Brier score.
    actual_one_hot = (
        test["FTR"].to_numpy()[:, None]
        == model.classes_[None, :]
    ).astype(float)

    brier = np.mean(
        np.sum(
            (probabilities - actual_one_hot) ** 2,
            axis=1
        )
    )

    # Baseline: always predict the most frequent
    # outcome from historical training matches.
    most_common = train["FTR"].mode()[0]

    baseline_predictions = np.repeat(
        most_common, len(test)
    )

    baseline_accuracy = accuracy_score(
        test["FTR"],
        baseline_predictions
    )

    # Probabilistic baseline: historical class frequencies.
    class_frequencies = (
        train["FTR"]
        .value_counts(normalize=True)
        .reindex(model.classes_, fill_value=0)
        .to_numpy()
    )

    baseline_probabilities = np.tile(
        class_frequencies,
        (len(test), 1)
    )

    baseline_loss = log_loss(
        test["FTR"],
        baseline_probabilities,
        labels=model.classes_,
    )

    print("\nMODEL PERFORMANCE")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Log loss: {loss:.4f}")
    print(f"Brier score: {brier:.4f}")

    print("\nBASELINE PERFORMANCE")
    print("Most common outcome:", most_common)
    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4f} ({baseline_accuracy:.2%})"
    )
    print(f"Baseline log loss: {baseline_loss:.4f}")

    print("\nCONFUSION MATRIX")
    print("Rows = actual, columns = predicted")
    print("Class order:", CLASSES)

    matrix = confusion_matrix(
        test["FTR"],
        predictions,
        labels=CLASSES
    )

    print(matrix)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            test["FTR"],
            predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    # Save evaluation summary.
    summary = pd.DataFrame([
        {
            "Model": "PitchIQ Logistic Regression",
            "Matches": len(test),
            "Accuracy": accuracy,
            "LogLoss": loss,
            "BrierScore": brier,
        },
        {
            "Model": "Historical frequency baseline",
            "Matches": len(test),
            "Accuracy": baseline_accuracy,
            "LogLoss": baseline_loss,
            "BrierScore": np.mean(
                np.sum(
                    (
                        baseline_probabilities
                        - actual_one_hot
                    ) ** 2,
                    axis=1
                )
            ),
        },
    ])

    summary_file = (
        OUTPUT_DIR / "current_season_evaluation.csv"
    )

    summary.to_csv(summary_file, index=False)

    # Save individual match predictions.
    predictions_df = test[
        ["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]
    ].copy()

    predictions_df["PredictedResult"] = predictions

    for i, cls in enumerate(model.classes_):
        predictions_df[f"Probability_{cls}"] = (
            probabilities[:, i]
        )

    predictions_file = (
        OUTPUT_DIR / "current_season_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
    )

    print("\nFILES SAVED")
    print(summary_file)
    print(predictions_file)

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
