
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["H", "D", "A"]

MODEL_FILES = {
    "Historical Frequency":
        "predictions_historical_frequency_no_features_2526.csv",

    "Logistic Regression (Baseline)":
        "predictions_logistic_regression_baseline_2526.csv",

    "Logistic Regression (xG)":
        "predictions_logistic_regression_xg_enhanced_2526.csv",

    "Random Forest (xG)":
        "predictions_random_forest_xg_enhanced_2526.csv",

    "HistGradientBoosting (xG)":
        "predictions_histgradientboosting_xg_enhanced_2526.csv",
}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():
    models = {}

    for name, filename in MODEL_FILES.items():
        path = TABLES_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {path}"
            )

        df = pd.read_csv(path)

        required = [
            "Date", "HomeTeam", "AwayTeam",
            "FTR", "Prediction",
            "Prob_H", "Prob_D", "Prob_A",
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename} is missing: {missing}"
            )

        if df[required].isna().any().any():
            raise ValueError(
                f"Missing values in {filename}"
            )

        if not df["FTR"].isin(CLASSES).all():
            raise ValueError(
                f"Unexpected outcomes in {filename}"
            )

        probabilities = df[
            ["Prob_H", "Prob_D", "Prob_A"]
        ].to_numpy(dtype=float)

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Non-finite probabilities in {filename}"
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                f"Invalid probabilities in {filename}"
            )

        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"Probabilities do not sum to 1 in {filename}"
            )

        df["MatchKey"] = (
            df["Date"].astype(str)
            + "|"
            + df["HomeTeam"].astype(str)
            + "|"
            + df["AwayTeam"].astype(str)
        )

        if df["MatchKey"].duplicated().any():
            raise ValueError(
                f"Duplicate matches in {filename}"
            )

        models[name] = df.set_index(
            "MatchKey"
        ).sort_index()

        print(f"Loaded {name}: {len(df)} matches")

    reference = next(iter(models.values()))

    for name, df in models.items():
        if not df.index.equals(reference.index):
            raise ValueError(
                f"Match mismatch for {name}"
            )

        if not df["FTR"].equals(reference["FTR"]):
            raise ValueError(
                f"Actual outcomes mismatch for {name}"
            )

    return models


# ============================================================
# CALIBRATION CURVES
# ============================================================

def plot_calibration(models):
    outcomes = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    for outcome, label in outcomes.items():
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            label="Perfect calibration",
        )

        for name, df in models.items():
            y_true = (
                df["FTR"] == outcome
            ).astype(int)

            y_prob = df[f"Prob_{outcome}"]

            observed, predicted = calibration_curve(
                y_true,
                y_prob,
                n_bins=10,
                strategy="uniform",
            )

            ax.plot(
                predicted,
                observed,
                marker="o",
                linewidth=1.8,
                label=name,
            )

        ax.set_title(
            f"Probability Calibration: {label}"
        )
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Outcome Frequency")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        filename = (
            f"calibration_{outcome.lower()}_2526.png"
        )

        fig.savefig(
            FIGURES_DIR / filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("Saved:", filename)


# ============================================================
# DRAW PROBABILITY DISTRIBUTION
# ============================================================

def plot_draw_probabilities(models):
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for name, df in models.items():
        ax.hist(
            df["Prob_D"],
            bins=np.linspace(0, 1, 21),
            alpha=0.35,
            label=name,
            density=True,
            histtype="stepfilled",
        )

    actual_draw_rate = (
        next(iter(models.values()))["FTR"] == "D"
    ).mean()

    ax.axvline(
        actual_draw_rate,
        linestyle="--",
        color="black",
        label=f"Actual draw rate: {actual_draw_rate:.1%}",
    )

    ax.set_title(
        "Distribution of Predicted Draw Probabilities"
    )
    ax.set_xlabel("Predicted Draw Probability")
    ax.set_ylabel("Density")

    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "draw_probability_distribution_2526.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved: draw_probability_distribution_2526.png"
    )


# ============================================================
# MODEL CONFIDENCE AND INCORRECT PREDICTIONS
# ============================================================

def confidence_analysis(models):
    summary = []
    mistakes = []

    for name, df in models.items():
        probabilities = df[
            ["Prob_H", "Prob_D", "Prob_A"]
        ].to_numpy(dtype=float)

        predictions = np.array(CLASSES)[
            np.argmax(probabilities, axis=1)
        ]

        actual = df["FTR"].to_numpy()

        confidence = probabilities.max(axis=1)
        incorrect = predictions != actual

        high_confidence_errors = (
            incorrect & (confidence >= 0.60)
        )

        actual_draw_rate = np.mean(
            actual == "D"
        )

        mean_draw_probability = (
            df["Prob_D"].mean()
        )

        # Probability assigned to the actual result.
        outcome_indices = np.array([
            CLASSES.index(outcome)
            for outcome in actual
        ])

        actual_probability = probabilities[
            np.arange(len(df)),
            outcome_indices,
        ]

        per_match_log_loss = -np.log(
            np.clip(
                actual_probability,
                1e-15,
                1.0,
            )
        )

        actual_one_hot = (
            np.array(CLASSES)[None, :]
            == actual[:, None]
        ).astype(float)

        brier = np.mean(
            np.sum(
                (probabilities - actual_one_hot) ** 2,
                axis=1,
            )
        )

        summary.append({
            "Model": name,
            "Matches": len(df),
            "Accuracy": np.mean(~incorrect),
            "LogLoss": np.mean(per_match_log_loss),
            "BrierScore": brier,
            "MeanConfidence": confidence.mean(),
            "MeanDrawProbability": mean_draw_probability,
            "ActualDrawRate": actual_draw_rate,
            "DrawPredictions": int(
                np.sum(predictions == "D")
            ),
            "HighConfidenceErrors_60pct": int(
                high_confidence_errors.sum()
            ),
            "MaximumMatchLogLoss": (
                per_match_log_loss.max()
            ),
        })

        error_df = df.reset_index()[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTR",
                "Prob_H",
                "Prob_D",
                "Prob_A",
            ]
        ].copy()

        error_df["Model"] = name
        error_df["PredictedOutcome"] = predictions
        error_df["Confidence"] = confidence
        error_df["ActualOutcomeProbability"] = (
            actual_probability
        )
        error_df["MatchLogLoss"] = per_match_log_loss
        error_df["Incorrect"] = incorrect

        mistakes.append(
            error_df[error_df["Incorrect"]]
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        TABLES_DIR / "calibration_summary_2526.csv",
        index=False,
    )

    all_mistakes = pd.concat(
        mistakes,
        ignore_index=True,
    )

    worst_errors = (
        all_mistakes
        .sort_values(
            "MatchLogLoss",
            ascending=False,
        )
        .groupby("Model", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )

    worst_errors.to_csv(
        TABLES_DIR / "worst_prediction_errors_2526.csv",
        index=False,
    )

    print("\nCALIBRATION SUMMARY")
    print(
        summary_df.round(4).to_string(index=False)
    )

    print("\nWorst prediction errors saved.")

    return summary_df


# ============================================================
# CONFIDENCE RELIABILITY DIAGRAM
# ============================================================

def plot_confidence_reliability(models):
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    bins = np.linspace(0, 1, 11)

    for name, df in models.items():
        probabilities = df[
            ["Prob_H", "Prob_D", "Prob_A"]
        ].to_numpy(dtype=float)

        predictions = np.array(CLASSES)[
            np.argmax(probabilities, axis=1)
        ]

        confidence = probabilities.max(axis=1)

        correct = (
            predictions == df["FTR"].to_numpy()
        ).astype(int)

        bin_ids = np.digitize(
            confidence,
            bins[1:-1],
            right=True,
        )

        x_values = []
        y_values = []

        for bin_id in range(10):
            mask = bin_ids == bin_id

            if not mask.any():
                continue

            x_values.append(
                confidence[mask].mean()
            )
            y_values.append(
                correct[mask].mean()
            )

        ax.plot(
            x_values,
            y_values,
            marker="o",
            label=name,
        )

    ax.set_title(
        "Prediction Confidence vs Actual Accuracy"
    )
    ax.set_xlabel("Mean Prediction Confidence")
    ax.set_ylabel("Observed Accuracy")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "confidence_reliability_2526.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved: confidence_reliability_2526.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PITCHIQ — PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 65)

    models = load_predictions()

    plot_calibration(models)

    plot_draw_probabilities(models)

    plot_confidence_reliability(models)

    confidence_analysis(models)

    print("\nANALYSIS COMPLETE")
    print("Figures:", FIGURES_DIR)
    print("Tables:", TABLES_DIR)


if __name__ == "__main__":
    main()
