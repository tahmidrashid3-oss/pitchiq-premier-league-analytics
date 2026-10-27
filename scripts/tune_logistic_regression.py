
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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

OUTPUT_DIR = ROOT / "reports" / "tables"

VALIDATION_SEASONS = [
    "2122",
    "2223",
    "2324",
    "2425",
]

# 2025/26 and 2026/27 remain outside model selection.
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

CALIBRATION_METHODS = [
    "none",
    "sigmoid",
    "isotonic",
]

CLASSES = ["A", "D", "H"]

METRICS = [
    "Points",
    "GoalsFor",
    "GoalsAgainst",
    "xGFor",
    "xGAgainst",
]

WINDOWS = [3, 5, 10]

# Preserve the exact deployed model feature ordering:
# all Overall features, followed by all Venue features.
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
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    df["Season"] = df["Season"].str.zfill(4)

    required = [
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FTR",
    ] + FEATURES

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing values detected in the dataset."
        )

    if not df["FTR"].isin(CLASSES).all():
        raise ValueError(
            "Invalid match outcomes detected."
        )

    if df.duplicated(
        ["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError(
            "Duplicate fixtures detected."
        )

    if len(FEATURES) != 30:
        raise ValueError(
            f"Expected 30 features, found {len(FEATURES)}."
        )

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    return df


# ============================================================
# BUILD LOGISTIC REGRESSION MODEL
# ============================================================

def build_model(c):

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                C=c,
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


# ============================================================
# CHRONOLOGICAL DATA SPLIT
# ============================================================

def split_historical_data(train):

    seasons = sorted(
        train["Season"].unique()
    )

    if len(seasons) < 2:
        raise ValueError(
            "At least two historical seasons are "
            "required for calibration."
        )

    calibration_season = seasons[-1]

    base_train = train[
        train["Season"] < calibration_season
    ].copy()

    calibration_data = train[
        train["Season"] == calibration_season
    ].copy()

    if base_train.empty or calibration_data.empty:
        raise ValueError(
            "Base training or calibration data is empty."
        )

    for name, subset in [
        ("Base training", base_train),
        ("Calibration", calibration_data),
    ]:
        if set(subset["FTR"]) != set(CLASSES):
            raise ValueError(
                f"{name} data is missing an outcome class."
            )

    if (
        base_train["Date"].max()
        >= calibration_data["Date"].min()
    ):
        raise ValueError(
            "Chronological overlap between base training "
            "and calibration data."
        )

    return base_train, calibration_data


# ============================================================
# TRAIN CANDIDATE MODEL
# ============================================================

def train_candidate(train, c, method):

    if method not in CALIBRATION_METHODS + ["none_base"]:
        raise ValueError(
            f"Unsupported method: {method}"
        )

    # Existing deployment configuration:
    # fit on all available historical training data.
    if method == "none":
        model = build_model(c)

        model.fit(
            train[FEATURES],
            train["FTR"],
        )

        return model

    base_train, calibration_data = (
        split_historical_data(train)
    )

    # Fair control: an uncalibrated model trained
    # on the same base data as calibrated models.
    base_model = build_model(c)

    base_model.fit(
        base_train[FEATURES],
        base_train["FTR"],
    )

    if method == "none_base":
        return base_model

    # FrozenEstimator prevents refitting the base model.
    # Calibration is fitted only on the reserved season.
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method=method,
    )

    calibrated_model.fit(
        calibration_data[FEATURES],
        calibration_data["FTR"],
    )

    return calibrated_model


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_model(model, validation):

    actual = validation["FTR"].to_numpy()

    raw_probabilities = model.predict_proba(
        validation[FEATURES]
    )

    # Align all probabilities to A, D, H.
    probabilities = pd.DataFrame(
        raw_probabilities,
        columns=model.classes_,
    )[CLASSES].to_numpy()

    predicted = np.array(CLASSES)[
        np.argmax(probabilities, axis=1)
    ]

    accuracy = accuracy_score(
        actual,
        predicted,
    )

    loss = log_loss(
        actual,
        probabilities,
        labels=CLASSES,
    )

    one_hot = (
        actual[:, None]
        == np.array(CLASSES)[None, :]
    ).astype(float)

    brier = np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1,
        )
    )

    actual_draws = actual == "D"
    predicted_draws = predicted == "D"

    draw_count = int(
        actual_draws.sum()
    )

    correct_draws = int(
        (actual_draws & predicted_draws).sum()
    )

    draw_recall = (
        correct_draws / draw_count
        if draw_count > 0
        else np.nan
    )

    draw_probability = probabilities[
        :, CLASSES.index("D")
    ]

    return {
        "Matches": len(validation),
        "Accuracy": accuracy,
        "LogLoss": loss,
        "BrierScore": brier,
        "DrawPredictions": int(
            predicted_draws.sum()
        ),
        "ActualDraws": draw_count,
        "CorrectDraws": correct_draws,
        "DrawRecall": draw_recall,
        "MeanDrawProbability": float(
            draw_probability.mean()
        ),
        "ActualDrawRate": float(
            actual_draws.mean()
        ),
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def main():

    print("=" * 70)
    print("PITCHIQ - LOGISTIC REGRESSION TUNING")
    print("=" * 70)

    df = load_data()

    print("\nDataset matches:", len(df))
    print("Model features:", len(FEATURES))

    results = []

    # Extra control for calibration comparisons.
    methods = [
        "none",
        "none_base",
        "sigmoid",
        "isotonic",
    ]

    for season in VALIDATION_SEASONS:

        train = df[
            df["Season"] < season
        ].copy()

        validation = df[
            df["Season"] == season
        ].copy()

        if train.empty or validation.empty:
            raise ValueError(
                f"Missing training or validation data "
                f"for season {season}."
            )

        if set(train["FTR"]) != set(CLASSES):
            raise ValueError(
                f"Missing training outcome class: {season}"
            )

        if (
            train["Date"].max()
            >= validation["Date"].min()
        ):
            raise ValueError(
                f"Chronological overlap for season {season}."
            )

        print("\n" + "-" * 70)
        print("Validation season:", season)
        print("Training matches:", len(train))
        print("Validation matches:", len(validation))
        print("-" * 70)

        for c in C_VALUES:

            for method in methods:

                model = train_candidate(
                    train,
                    c,
                    method,
                )

                scores = evaluate_model(
                    model,
                    validation,
                )

                results.append({
                    "Season": season,
                    "C": c,
                    "Calibration": method,
                    **scores,
                })

                print(
                    f"C={c:<6} "
                    f"Method={method:<10} "
                    f"Accuracy={scores['Accuracy']:.2%} "
                    f"LogLoss={scores['LogLoss']:.4f} "
                    f"DrawRecall={scores['DrawRecall']:.2%}"
                )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    # ========================================================
    # SUMMARISE ACROSS SEASONS
    # ========================================================

    summary = (
        results_df
        .groupby(
            ["C", "Calibration"],
            as_index=False,
        )
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanLogLoss=("LogLoss", "mean"),
            MeanBrierScore=("BrierScore", "mean"),
            TotalDrawPredictions=("DrawPredictions", "sum"),
            TotalActualDraws=("ActualDraws", "sum"),
            TotalCorrectDraws=("CorrectDraws", "sum"),
            MeanDrawProbability=("MeanDrawProbability", "mean"),
            MeanActualDrawRate=("ActualDrawRate", "mean"),
            TotalMatches=("Matches", "sum"),
        )
    )

    summary["OverallDrawRecall"] = (
        summary["TotalCorrectDraws"]
        / summary["TotalActualDraws"]
    )

    summary = summary.sort_values(
        ["MeanLogLoss", "MeanBrierScore"]
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("TUNING SUMMARY - SORTED BY LOG LOSS")
    print("=" * 70)

    print(
        summary.round(5).to_string(index=False)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_file = (
        OUTPUT_DIR / "logistic_tuning_results.csv"
    )

    summary_file = (
        OUTPUT_DIR / "logistic_tuning_summary.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # PRINT BENCHMARK
    # ========================================================

    benchmark = summary[
        (summary["C"] == 0.01)
        & (summary["Calibration"] == "none")
    ]

    print("\n" + "=" * 70)
    print("CURRENT DEPLOYED CONFIGURATION")
    print("=" * 70)

    if not benchmark.empty:

        row = benchmark.iloc[0]

        print("Model: Logistic Regression")
        print("C: 0.01")
        print("Calibration: none")
        print(
            f"Validation accuracy: "
            f"{row['MeanAccuracy']:.2%}"
        )
        print(
            f"Validation log loss: "
            f"{row['MeanLogLoss']:.5f}"
        )
        print(
            f"Validation Brier score: "
            f"{row['MeanBrierScore']:.5f}"
        )

    print("\nResults saved to:")
    print(results_file)
    print(summary_file)

    print("\nTUNING COMPLETE")
    print("The deployed model has not been modified.")


if __name__ == "__main__":
    main()
