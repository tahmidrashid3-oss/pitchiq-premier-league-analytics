
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_SEASONS = ["2122", "2223", "2324", "2425"]

C = 0.01
N_REPEATS = 5
RANDOM_STATE = 42

CLASSES = ["H", "D", "A"]

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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}\n"
            "Run scripts/build_advanced_features.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
    )

    df["Season"] = df["Season"].str.zfill(4)

    required = [
        "Season",
        "FTR",
    ] + FEATURES

    missing = [
        col for col in required
        if col not in df.columns
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

    print(f"Loaded {len(df)} matches")
    print(f"Number of model features: {len(FEATURES)}")

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
                C=C,
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ============================================================
# CORRECT MULTICLASS LOG LOSS
# ============================================================

def calculate_log_loss(model, X, y):
    """
    Calculate log loss directly from the probability
    assigned to each actual outcome.

    This avoids probability-column ordering errors.
    """

    probabilities = model.predict_proba(X)

    model_classes = model.classes_

    class_indices = {
        label: i
        for i, label in enumerate(model_classes)
    }

    actual_indices = np.array([
        class_indices[label]
        for label in np.asarray(y)
    ])

    actual_probabilities = probabilities[
        np.arange(len(y)),
        actual_indices,
    ]

    return -np.log(
        np.clip(actual_probabilities, 1e-15, 1.0)
    ).mean()


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    model,
    X_validation,
    y_validation,
    season,
):
    """
    Shuffle one feature at a time and measure
    the increase in validation log loss.

    Positive importance:
        Shuffling makes performance worse.

    Negative importance:
        Shuffling improves performance.

    Importance is calculated on out-of-sample
    historical validation matches.
    """

    baseline_loss = calculate_log_loss(
        model,
        X_validation,
        y_validation,
    )

    rng = np.random.default_rng(
        RANDOM_STATE + int(season)
    )

    results = []

    for feature in FEATURES:
        for repeat in range(N_REPEATS):

            shuffled = X_validation.copy()

            shuffled[feature] = rng.permutation(
                shuffled[feature].to_numpy()
            )

            shuffled_loss = calculate_log_loss(
                model,
                shuffled,
                y_validation,
            )

            importance = (
                shuffled_loss - baseline_loss
            )

            results.append({
                "Season": season,
                "Feature": feature,
                "Repeat": repeat + 1,
                "BaselineLogLoss": baseline_loss,
                "ShuffledLogLoss": shuffled_loss,
                "Importance": importance,
            })

    return results


# ============================================================
# STANDARDISED MODEL COEFFICIENTS
# ============================================================

def extract_coefficients(model, season):
    """
    StandardScaler ensures that coefficients are
    comparable in terms of standardised input units.

    Positive coefficients increase the relative
    log-odds of that outcome versus alternatives.
    """

    classifier = model.named_steps["model"]

    coefficients = []

    for class_index, outcome in enumerate(
        classifier.classes_
    ):
        for feature_index, feature in enumerate(
            FEATURES
        ):
            coefficients.append({
                "Season": season,
                "Outcome": outcome,
                "Feature": feature,
                "Coefficient": classifier.coef_[
                    class_index,
                    feature_index,
                ],
            })

    return coefficients


# ============================================================
# ROLLING VALIDATION
# ============================================================

def run_analysis(df):
    all_importances = []
    all_coefficients = []
    validation_metrics = []

    print("\n" + "=" * 70)
    print("ROLLING VALIDATION FEATURE IMPORTANCE")
    print("=" * 70)

    for season in VALIDATION_SEASONS:

        train = df[
            df["Season"] < season
        ]

        validation = df[
            df["Season"] == season
        ]

        if train.empty or validation.empty:
            raise ValueError(
                f"Insufficient data for {season}"
            )

        model = create_model()

        model.fit(
            train[FEATURES],
            train["FTR"],
        )

        X_validation = validation[FEATURES]
        y_validation = validation["FTR"]

        baseline_loss = calculate_log_loss(
            model,
            X_validation,
            y_validation,
        )

        print(
            f"\nSeason {season}: "
            f"{len(validation)} matches"
        )

        print(
            f"Baseline log loss: {baseline_loss:.5f}"
        )

        validation_metrics.append({
            "Season": season,
            "Matches": len(validation),
            "LogLoss": baseline_loss,
        })

        all_importances.extend(
            calculate_permutation_importance(
                model,
                X_validation,
                y_validation,
                season,
            )
        )

        all_coefficients.extend(
            extract_coefficients(
                model,
                season,
            )
        )

    importance_df = pd.DataFrame(
        all_importances
    )

    coefficient_df = pd.DataFrame(
        all_coefficients
    )

    metrics_df = pd.DataFrame(
        validation_metrics
    )

    return importance_df, coefficient_df, metrics_df


# ============================================================
# SUMMARISE IMPORTANCE
# ============================================================

def summarise_importance(importance_df):
    """
    Average each feature's importance across
    seasons and permutation repetitions.

    Also report the spread across season-level
    means rather than treating repetitions as
    independent validation seasons.
    """

    seasonal = (
        importance_df
        .groupby(
            ["Season", "Feature"],
            as_index=False,
        )
        .agg(
            MeanImportance=(
                "Importance",
                "mean",
            )
        )
    )

    summary = (
        seasonal
        .groupby("Feature", as_index=False)
        .agg(
            MeanImportance=(
                "MeanImportance",
                "mean",
            ),
            StdAcrossSeasons=(
                "MeanImportance",
                "std",
            ),
            PositiveSeasons=(
                "MeanImportance",
                lambda x: int((x > 0).sum()),
            ),
        )
        .sort_values(
            "MeanImportance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return seasonal, summary


# ============================================================
# SUMMARISE COEFFICIENTS
# ============================================================

def summarise_coefficients(coefficient_df):
    return (
        coefficient_df
        .groupby(
            ["Outcome", "Feature"],
            as_index=False,
        )
        .agg(
            MeanCoefficient=(
                "Coefficient",
                "mean",
            ),
            StdCoefficient=(
                "Coefficient",
                "std",
            ),
        )
        .sort_values(
            ["Outcome", "MeanCoefficient"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


# ============================================================
# PLOT TOP PERMUTATION IMPORTANCES
# ============================================================

def plot_importance(summary):
    top = (
        summary
        .head(15)
        .sort_values("MeanImportance")
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.barh(
        top["Feature"],
        top["MeanImportance"],
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_xlabel(
        "Increase in Validation Log Loss After Shuffling"
    )

    ax.set_title(
        "PitchIQ: Top 15 Permutation Feature Importances"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        FIGURES_DIR
        / "advanced_permutation_importance.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved figure: {path}")


# ============================================================
# PLOT COEFFICIENTS BY OUTCOME
# ============================================================

def plot_coefficients(coefficient_summary):
    """
    Produce one coefficient plot for each outcome.
    Show the 12 largest absolute coefficients.
    """

    labels = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    for outcome in CLASSES:

        subset = coefficient_summary[
            coefficient_summary["Outcome"] == outcome
        ].copy()

        subset["AbsoluteCoefficient"] = (
            subset["MeanCoefficient"].abs()
        )

        top = (
            subset
            .nlargest(
                12,
                "AbsoluteCoefficient",
            )
            .sort_values("MeanCoefficient")
        )

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        ax.barh(
            top["Feature"],
            top["MeanCoefficient"],
        )

        ax.axvline(
            0,
            color="black",
            linewidth=1,
        )

        ax.set_xlabel(
            "Mean Standardised Logistic Regression Coefficient"
        )

        ax.set_title(
            f"PitchIQ: Feature Coefficients — {labels[outcome]}"
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

        fig.tight_layout()

        path = (
            FIGURES_DIR
            / f"advanced_coefficients_{outcome}.png"
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved figure: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PITCHIQ — FEATURE IMPORTANCE AND INTERPRETATION")
    print("=" * 70)

    df = load_data()

    importance_df, coefficient_df, metrics_df = (
        run_analysis(df)
    )

    seasonal_importance, importance_summary = (
        summarise_importance(importance_df)
    )

    coefficient_summary = (
        summarise_coefficients(coefficient_df)
    )

    # Save detailed and summarised results.
    importance_df.to_csv(
        TABLES_DIR
        / "advanced_permutation_importance_detailed.csv",
        index=False,
    )

    seasonal_importance.to_csv(
        TABLES_DIR
        / "advanced_permutation_importance_by_season.csv",
        index=False,
    )

    importance_summary.to_csv(
        TABLES_DIR
        / "advanced_permutation_importance_summary.csv",
        index=False,
    )

    coefficient_df.to_csv(
        TABLES_DIR
        / "advanced_coefficients_detailed.csv",
        index=False,
    )

    coefficient_summary.to_csv(
        TABLES_DIR
        / "advanced_coefficients_summary.csv",
        index=False,
    )

    metrics_df.to_csv(
        TABLES_DIR
        / "advanced_importance_validation_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("TOP 15 FEATURES BY PERMUTATION IMPORTANCE")
    print("=" * 70)

    print(
        importance_summary.head(15)
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("VALIDATION LOG LOSS")
    print("=" * 70)

    print(
        metrics_df.round(5)
        .to_string(index=False)
    )

    plot_importance(
        importance_summary
    )

    plot_coefficients(
        coefficient_summary
    )

    print("\nANALYSIS COMPLETE")
    print("Tables:", TABLES_DIR)
    print("Figures:", FIGURES_DIR)


if __name__ == "__main__":
    main()
