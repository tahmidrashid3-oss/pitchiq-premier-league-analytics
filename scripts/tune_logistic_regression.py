
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

VALIDATION_SEASONS = [
    "2122",
    "2223",
    "2324",
    "2425",
]

TEST_SEASON = "2526"

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {DATA_FILE}"
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
            "Unexpected match outcome detected."
        )

    df = df.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    print(f"Loaded {len(df)} matches")

    return df


# ============================================================
# MODEL
# ============================================================

def create_model(c_value):
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                C=c_value,
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


# ============================================================
# METRICS
# ============================================================

def evaluate_model(model, X, y):
    """
    Evaluate probabilities using the correct class order.
    """

    raw_probabilities = model.predict_proba(X)

    probabilities = pd.DataFrame(
        raw_probabilities,
        columns=model.classes_,
    )[CLASSES].to_numpy()

    predictions = np.array(CLASSES)[
        np.argmax(probabilities, axis=1)
    ]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    # Correct log loss using alphabetical class order.
    sklearn_probabilities = pd.DataFrame(
        raw_probabilities,
        columns=model.classes_,
    )[["A", "D", "H"]].to_numpy()

    loss = log_loss(
        y,
        sklearn_probabilities,
        labels=["A", "D", "H"],
    )

    actual = (
        np.asarray(y)[:, None]
        == np.array(CLASSES)[None, :]
    ).astype(float)

    brier = np.mean(
        np.sum(
            (probabilities - actual) ** 2,
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
    }, predictions, probabilities


# ============================================================
# ROLLING-SEASON VALIDATION
# ============================================================

def tune_regularisation(df):
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION REGULARISATION TUNING")
    print("=" * 70)

    results = []

    for c_value in C_VALUES:
        print(f"\nTesting C = {c_value}")

        for season in VALIDATION_SEASONS:
            train = df[df["Season"] < season]
            validation = df[df["Season"] == season]

            if train.empty or validation.empty:
                raise ValueError(
                    f"Missing data for season {season}"
                )

            model = create_model(c_value)

            model.fit(
                train[FEATURES],
                train["FTR"],
            )

            metrics, _, _ = evaluate_model(
                model,
                validation[FEATURES],
                validation["FTR"],
            )

            results.append({
                "C": c_value,
                "Season": season,
                "Matches": len(validation),
                **metrics,
            })

            print(
                f"  {season} | "
                f"Accuracy: {metrics['Accuracy']:.4f} | "
                f"Log loss: {metrics['LogLoss']:.4f} | "
                f"Brier: {metrics['BrierScore']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        TABLES_DIR / "logistic_tuning_results.csv",
        index=False,
    )

    summary = (
        results_df
        .groupby("C", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanLogLoss=("LogLoss", "mean"),
            MeanBrierScore=("BrierScore", "mean"),
            TotalDrawPredictions=(
                "DrawPredictions",
                "sum",
            ),
        )
        .sort_values(
            ["MeanLogLoss", "C"],
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        TABLES_DIR / "logistic_tuning_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)

    print(
        summary.round(5).to_string(index=False)
    )

    best_c = float(
        summary.iloc[0]["C"]
    )

    print(f"\nBest C by validation log loss: {best_c}")

    return best_c, summary


# ============================================================
# PLOT TUNING RESULTS
# ============================================================

def plot_tuning(summary):
    ordered = summary.sort_values("C")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        ordered["C"],
        ordered["MeanLogLoss"],
        marker="o",
        linewidth=2,
    )

    ax.set_xscale("log")

    ax.set_title(
        "Logistic Regression Regularisation vs Validation Log Loss"
    )

    ax.set_xlabel(
        "Regularisation Strength Parameter (C)"
    )

    ax.set_ylabel(
        "Mean Validation Log Loss (Lower is Better)"
    )

    ax.grid(
        alpha=0.3,
        which="both",
    )

    fig.tight_layout()

    path = (
        FIGURES_DIR
        / "logistic_regularisation_tuning.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved plot: {path}")


# ============================================================
# FINAL HOLDOUT TEST
# ============================================================

def final_test(df, best_c):
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

    model = create_model(best_c)

    model.fit(
        train[FEATURES],
        train["FTR"],
    )

    metrics, predictions, probabilities = evaluate_model(
        model,
        test[FEATURES],
        test["FTR"],
    )

    print(f"\nSelected C: {best_c}")

    print(
        f"Accuracy: {metrics['Accuracy']:.4f}"
    )

    print(
        f"Log loss: {metrics['LogLoss']:.4f}"
    )

    print(
        f"Brier score: {metrics['BrierScore']:.4f}"
    )

    print(
        f"Draw predictions: {metrics['DrawPredictions']}"
    )

    comparison = pd.DataFrame([
        {
            "Model": "Tuned Logistic Regression + xG",
            "C": best_c,
            **metrics,
        }
    ])

    comparison.to_csv(
        TABLES_DIR / "tuned_logistic_test_results.csv",
        index=False,
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

    output["Prediction"] = predictions

    for i, outcome in enumerate(CLASSES):
        output[
            f"Prob_{outcome}"
        ] = probabilities[:, i]

    output.to_csv(
        TABLES_DIR
        / "predictions_tuned_logistic_xg_2526.csv",
        index=False,
    )

    print("\nFinal test predictions saved.")


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "PITCHIQ — LOGISTIC REGRESSION HYPERPARAMETER TUNING"
    )

    print("-" * 70)

    df = load_data()

    best_c, summary = tune_regularisation(df)

    plot_tuning(summary)

    final_test(df, best_c)

    print("\nTUNING COMPLETE")


if __name__ == "__main__":
    main()
