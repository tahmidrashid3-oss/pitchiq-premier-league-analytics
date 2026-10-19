
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "match_features.csv"

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Diff_Points",
    "Diff_GoalsFor",
    "Diff_GoalsAgainst",
    "Diff_xGFor",
    "Diff_xGAgainst",
]

CLASSES = ["H", "D", "A"]
SKLEARN_CLASSES = ["A", "D", "H"]

BEST_C = 0.01

CALIBRATION_SEASONS = [
    "2122",
    "2223",
    "2324",
    "2425",
]

TEST_SEASON = "2526"

EPS = 1e-15


# ============================================================
# LOAD DATA
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
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing values detected."
        )

    if not df["FTR"].isin(CLASSES).all():
        raise ValueError(
            "Unexpected match outcomes detected."
        )

    df = df.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    print(f"Loaded {len(df)} matches")

    return df


# ============================================================
# MODEL
# ============================================================

def create_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                C=BEST_C,
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


# ============================================================
# PROBABILITY HELPERS
# ============================================================

def predict_probabilities(model, X):
    """
    Return probabilities in H, D, A order.
    """

    raw = model.predict_proba(X)

    return pd.DataFrame(
        raw,
        columns=model.classes_,
    )[CLASSES].to_numpy(dtype=float)


def temperature_scale(probabilities, temperature):
    """
    Apply temperature scaling to multiclass probabilities.

    T > 1 softens probabilities.
    T < 1 sharpens probabilities.
    T = 1 leaves probabilities unchanged.
    """

    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive."
        )

    log_probs = np.log(
        np.clip(probabilities, EPS, 1.0)
    )

    scaled_logits = log_probs / temperature

    scaled_logits -= scaled_logits.max(
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(scaled_logits)

    return exp_logits / exp_logits.sum(
        axis=1,
        keepdims=True,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):
    y_true = np.asarray(y_true)

    predictions = np.array(CLASSES)[
        np.argmax(probabilities, axis=1)
    ]

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    # Reorder H,D,A to A,D,H for sklearn.
    sklearn_probabilities = probabilities[
        :,
        [CLASSES.index(c) for c in SKLEARN_CLASSES],
    ]

    loss = log_loss(
        y_true,
        sklearn_probabilities,
        labels=SKLEARN_CLASSES,
    )

    actual_one_hot = (
        y_true[:, None]
        == np.array(CLASSES)[None, :]
    ).astype(float)

    brier = np.mean(
        np.sum(
            (probabilities - actual_one_hot) ** 2,
            axis=1,
        )
    )

    draws = int(
        np.sum(predictions == "D")
    )

    return {
        "Accuracy": accuracy,
        "LogLoss": loss,
        "BrierScore": brier,
        "DrawPredictions": draws,
    }


# ============================================================
# GENERATE HISTORICAL OUT-OF-SAMPLE PREDICTIONS
# ============================================================

def generate_calibration_data(df):
    print("\n" + "=" * 70)
    print("GENERATING HISTORICAL CALIBRATION DATA")
    print("=" * 70)

    probability_batches = []
    outcome_batches = []
    records = []

    for season in CALIBRATION_SEASONS:
        train = df[df["Season"] < season]
        validation = df[df["Season"] == season]

        if train.empty or validation.empty:
            raise ValueError(
                f"Insufficient data for {season}"
            )

        model = create_model()

        model.fit(
            train[FEATURES],
            train["FTR"],
        )

        probabilities = predict_probabilities(
            model,
            validation[FEATURES],
        )

        probability_batches.append(
            probabilities
        )

        outcome_batches.append(
            validation["FTR"].to_numpy()
        )

        batch = validation[
            [
                "Date",
                "Season",
                "HomeTeam",
                "AwayTeam",
                "FTR",
            ]
        ].copy()

        for i, outcome in enumerate(CLASSES):
            batch[
                f"Prob_{outcome}"
            ] = probabilities[:, i]

        records.append(batch)

        metrics = calculate_metrics(
            validation["FTR"],
            probabilities,
        )

        print(
            f"{season}: "
            f"{len(validation)} matches | "
            f"Log loss: {metrics['LogLoss']:.4f}"
        )

    all_probabilities = np.vstack(
        probability_batches
    )

    all_outcomes = np.concatenate(
        outcome_batches
    )

    calibration_predictions = pd.concat(
        records,
        ignore_index=True,
    )

    calibration_predictions.to_csv(
        TABLES_DIR
        / "temperature_calibration_history.csv",
        index=False,
    )

    return all_probabilities, all_outcomes


# ============================================================
# FIT TEMPERATURE
# ============================================================

def fit_temperature(probabilities, y_true):
    print("\n" + "=" * 70)
    print("OPTIMISING TEMPERATURE")
    print("=" * 70)

    def objective(temperature):
        scaled = temperature_scale(
            probabilities,
            temperature,
        )

        return calculate_metrics(
            y_true,
            scaled,
        )["LogLoss"]

    result = minimize_scalar(
        objective,
        bounds=(0.25, 5.0),
        method="bounded",
        options={"xatol": 1e-6},
    )

    if not result.success:
        raise RuntimeError(
            "Temperature optimisation failed."
        )

    best_temperature = float(result.x)

    original_loss = objective(1.0)
    calibrated_loss = objective(best_temperature)

    print(f"Optimal temperature: {best_temperature:.4f}")
    print(f"Original historical log loss: {original_loss:.5f}")
    print(f"Calibrated historical log loss: {calibrated_loss:.5f}")

    return best_temperature


# ============================================================
# FINAL TEST
# ============================================================

def final_test(df, temperature):
    print("\n" + "=" * 70)
    print("FINAL HOLDOUT TEST: 2025/26")
    print("=" * 70)

    train = df[
        df["Season"] < TEST_SEASON
    ]

    test = df[
        df["Season"] == TEST_SEASON
    ]

    if len(test) != 380:
        raise ValueError(
            f"Expected 380 test matches, found {len(test)}"
        )

    model = create_model()

    model.fit(
        train[FEATURES],
        train["FTR"],
    )

    original_probabilities = predict_probabilities(
        model,
        test[FEATURES],
    )

    calibrated_probabilities = temperature_scale(
        original_probabilities,
        temperature,
    )

    original_metrics = calculate_metrics(
        test["FTR"],
        original_probabilities,
    )

    calibrated_metrics = calculate_metrics(
        test["FTR"],
        calibrated_probabilities,
    )

    comparison = pd.DataFrame([
        {
            "Model": "Tuned Logistic Regression + xG",
            "Temperature": 1.0,
            **original_metrics,
        },
        {
            "Model": "Temperature-Calibrated Logistic Regression + xG",
            "Temperature": temperature,
            **calibrated_metrics,
        },
    ])

    comparison.to_csv(
        TABLES_DIR
        / "temperature_calibration_comparison.csv",
        index=False,
    )

    print("\nFINAL TEST COMPARISON")
    print(
        comparison.round(5).to_string(index=False)
    )

    output = test[
        [
            "Date",
            "Season",
            "HomeTeam",
            "AwayTeam",
            "FTR",
        ]
    ].copy()

    original_predictions = np.array(CLASSES)[
        np.argmax(
            original_probabilities,
            axis=1,
        )
    ]

    calibrated_predictions = np.array(CLASSES)[
        np.argmax(
            calibrated_probabilities,
            axis=1,
        )
    ]

    output["OriginalPrediction"] = original_predictions
    output["CalibratedPrediction"] = calibrated_predictions

    for i, outcome in enumerate(CLASSES):
        output[
            f"OriginalProb_{outcome}"
        ] = original_probabilities[:, i]

        output[
            f"CalibratedProb_{outcome}"
        ] = calibrated_probabilities[:, i]

    output.to_csv(
        TABLES_DIR
        / "predictions_temperature_calibrated_2526.csv",
        index=False,
    )

    return comparison


# ============================================================
# VISUALISATION
# ============================================================

def plot_comparison(comparison):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [
        "Uncalibrated",
        "Temperature-scaled",
    ]

    values = comparison["LogLoss"].to_numpy()

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "2025/26 Log Loss: Before vs After Calibration"
    )

    ax.set_ylabel(
        "Log Loss (Lower is Better)"
    )

    ax.set_ylim(
        0,
        max(values) * 1.15,
    )

    for i, value in enumerate(values):
        ax.text(
            i,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
        )

    fig.tight_layout()

    path = (
        FIGURES_DIR
        / "temperature_calibration_comparison.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved figure: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "PITCHIQ — TEMPERATURE SCALING CALIBRATION"
    )

    print("-" * 70)

    df = load_data()

    historical_probs, historical_outcomes = (
        generate_calibration_data(df)
    )

    temperature = fit_temperature(
        historical_probs,
        historical_outcomes,
    )

    comparison = final_test(
        df,
        temperature,
    )

    plot_comparison(comparison)

    print("\nCALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
