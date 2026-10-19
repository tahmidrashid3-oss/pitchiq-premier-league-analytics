
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed" / "match_features.csv"
)

OUTPUT_DIR = ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Presentation order for football outcomes
CLASSES = ["H", "D", "A"]

# scikit-learn log_loss expects alphabetical class order
SKLEARN_CLASSES = ["A", "D", "H"]

VALIDATION_SEASONS = [
    "2122",
    "2223",
    "2324",
    "2425",
]

TEST_SEASON = "2526"

BASELINE_FEATURES = [
    "Diff_Points",
    "Diff_GoalsFor",
    "Diff_GoalsAgainst",
]

XG_FEATURES = BASELINE_FEATURES + [
    "Diff_xGFor",
    "Diff_xGAgainst",
]

FEATURE_SETS = {
    "Baseline": BASELINE_FEATURES,
    "xG-enhanced": XG_FEATURES,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Feature dataset missing. "
            "Run scripts/build_features.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    df["Season"] = df["Season"].str.zfill(4)

    required = (
        [
            "Date",
            "Season",
            "HomeTeam",
            "AwayTeam",
            "FTR",
        ]
        + XG_FEATURES
    )

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing feature values detected."
        )

    if not df["FTR"].isin(CLASSES).all():
        raise ValueError(
            "Unexpected match outcome found."
        )

    if df.duplicated(
        subset=["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError(
            "Duplicate fixtures detected."
        )

    df = df.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    print("Loaded matches:", len(df))
    print(
        "Available seasons:",
        df["Season"].nunique()
    )

    return df


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_model(name):

    if name == "Logistic Regression":
        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    random_state=42,
                ),
            ),
        ])

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )

    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=42,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "HistGradientBoosting",
]


# ============================================================
# PROBABILITY HELPERS
# ============================================================

def aligned_probabilities(model, X):
    """
    Return probabilities in football outcome order:
    H, D, A.
    """

    raw_probabilities = model.predict_proba(X)

    probabilities = pd.DataFrame(
        raw_probabilities,
        columns=model.classes_,
    )

    return probabilities[
        CLASSES
    ].to_numpy()


def calculate_metrics(y_true, probabilities):
    """
    Calculate accuracy, correctly ordered log loss,
    multiclass Brier score, and draw predictions.

    Input probabilities must be ordered H, D, A.
    """

    y_true = np.asarray(y_true)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.shape != (
        len(y_true),
        len(CLASSES),
    ):
        raise ValueError(
            "Unexpected probability array shape."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite probabilities detected."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Predicted probabilities do not sum to 1."
        )

    # Predicted outcome
    predictions = np.array(CLASSES)[
        np.argmax(probabilities, axis=1)
    ]

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    # --------------------------------------------------------
    # CORRECTED LOG LOSS
    # --------------------------------------------------------
    #
    # Current probability order:
    # H, D, A
    #
    # Required sklearn order:
    # A, D, H
    #
    # Reorder columns before calculating log loss.
    # --------------------------------------------------------

    sklearn_probabilities = probabilities[
        :,
        [CLASSES.index(label) for label in SKLEARN_CLASSES],
    ]

    loss = log_loss(
        y_true,
        sklearn_probabilities,
        labels=SKLEARN_CLASSES,
    )

    # --------------------------------------------------------
    # MULTICLASS BRIER SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DRAW PREDICTIONS
    # --------------------------------------------------------

    draw_predictions = int(
        np.sum(predictions == "D")
    )

    return {
        "Accuracy": accuracy,
        "LogLoss": loss,
        "BrierScore": brier,
        "DrawPredictions": draw_predictions,
    }, predictions


# ============================================================
# HISTORICAL FREQUENCY BENCHMARK
# ============================================================

def historical_probabilities(
    train,
    evaluation,
):
    """
    Predict the historical outcome frequencies
    observed in the training data.

    Uses training data only.
    """

    frequencies = (
        train["FTR"]
        .value_counts(normalize=True)
        .reindex(
            CLASSES,
            fill_value=0,
        )
        .to_numpy()
    )

    return np.tile(
        frequencies,
        (len(evaluation), 1),
    )


# ============================================================
# TRAINING AND PREDICTION
# ============================================================

def train_and_predict(
    name,
    features,
    train,
    evaluation,
):

    if name == "Historical Frequency":
        return historical_probabilities(
            train,
            evaluation,
        )

    model = create_model(name)

    model.fit(
        train[features],
        train["FTR"],
    )

    return aligned_probabilities(
        model,
        evaluation[features],
    )


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

def get_configurations():

    configurations = [
        (
            "Historical Frequency",
            "No features",
            [],
        ),
    ]

    for model_name in MODEL_NAMES:
        for feature_set, columns in FEATURE_SETS.items():
            configurations.append(
                (
                    model_name,
                    feature_set,
                    columns,
                )
            )

    return configurations


# ============================================================
# ROLLING-SEASON VALIDATION
# ============================================================

def rolling_validation(df):

    print("\n" + "=" * 70)
    print("ROLLING-SEASON VALIDATION")
    print("=" * 70)

    results = []

    configurations = get_configurations()

    for season in VALIDATION_SEASONS:

        train = df[
            df["Season"] < season
        ]

        validation = df[
            df["Season"] == season
        ]

        if train.empty or validation.empty:
            raise ValueError(
                f"Insufficient data for season {season}."
            )

        print(
            f"\nSeason {season}: "
            f"{len(train)} training matches, "
            f"{len(validation)} validation matches"
        )

        for name, feature_set, columns in configurations:

            probabilities = train_and_predict(
                name,
                columns,
                train,
                validation,
            )

            metrics, _ = calculate_metrics(
                validation["FTR"],
                probabilities,
            )

            results.append({
                "Season": season,
                "Model": name,
                "FeatureSet": feature_set,
                "Matches": len(validation),
                **metrics,
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "rolling_validation_results.csv",
        index=False,
    )

    # Average performance across validation seasons
    summary = (
        results_df
        .groupby(
            ["Model", "FeatureSet"],
            as_index=False,
        )
        .agg(
            MeanAccuracy=(
                "Accuracy",
                "mean",
            ),
            MeanLogLoss=(
                "LogLoss",
                "mean",
            ),
            MeanBrierScore=(
                "BrierScore",
                "mean",
            ),
            TotalDrawPredictions=(
                "DrawPredictions",
                "sum",
            ),
        )
        .sort_values(
            "MeanLogLoss"
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR / "rolling_validation_summary.csv",
        index=False,
    )

    print("\nAVERAGE VALIDATION PERFORMANCE")
    print(
        summary.round(4).to_string(
            index=False
        )
    )

    return summary


# ============================================================
# FINAL 2025/26 HOLDOUT TEST
# ============================================================

def final_test(
    df,
    validation_summary,
):

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
            f"Expected 380 test matches, found {len(test)}."
        )

    print(
        "Training matches:",
        len(train),
    )

    print(
        "Test matches:",
        len(test),
    )

    # --------------------------------------------------------
    # SELECT BEST MODEL USING VALIDATION ONLY
    # --------------------------------------------------------

    candidates = validation_summary[
        validation_summary["Model"]
        != "Historical Frequency"
    ]

    best = candidates.iloc[0]

    print("\nBest validation configuration:")
    print(
        "Model:",
        best["Model"],
    )

    print(
        "Features:",
        best["FeatureSet"],
    )

    print(
        "Mean validation log loss:",
        round(
            best["MeanLogLoss"],
            4,
        ),
    )

    configurations = get_configurations()

    test_results = []
    prediction_files = {}

    for name, feature_set, columns in configurations:

        probabilities = train_and_predict(
            name,
            columns,
            train,
            test,
        )

        metrics, predictions = calculate_metrics(
            test["FTR"],
            probabilities,
        )

        test_results.append({
            "Model": name,
            "FeatureSet": feature_set,
            **metrics,
        })

        print(
            f"\n{name} | {feature_set}"
        )

        print(
            "Accuracy:",
            f"{metrics['Accuracy']:.4f}",
        )

        print(
            "Log loss:",
            f"{metrics['LogLoss']:.4f}",
        )

        print(
            "Brier score:",
            f"{metrics['BrierScore']:.4f}",
        )

        print(
            "Draw predictions:",
            metrics["DrawPredictions"],
        )

        # ----------------------------------------------------
        # SAVE MATCH-LEVEL PREDICTIONS
        # ----------------------------------------------------

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

        safe_name = (
            f"{name}_{feature_set}"
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        path = (
            OUTPUT_DIR
            / f"predictions_{safe_name}_2526.csv"
        )

        output.to_csv(
            path,
            index=False,
        )

        prediction_files[
            (name, feature_set)
        ] = (
            predictions,
            probabilities,
        )

    # --------------------------------------------------------
    # FINAL MODEL COMPARISON
    # --------------------------------------------------------

    comparison = (
        pd.DataFrame(test_results)
        .sort_values(
            "LogLoss"
        )
        .reset_index(drop=True)
    )

    comparison.to_csv(
        OUTPUT_DIR / "improved_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DETAILED DIAGNOSTICS FOR VALIDATION-SELECTED MODEL
    # --------------------------------------------------------

    key = (
        best["Model"],
        best["FeatureSet"],
    )

    predictions, probabilities = (
        prediction_files[key]
    )

    print("\n" + "=" * 70)
    print("VALIDATION-SELECTED MODEL DIAGNOSTICS")
    print("=" * 70)

    print(
        classification_report(
            test["FTR"],
            predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("Confusion matrix (H, D, A):")

    print(
        confusion_matrix(
            test["FTR"],
            predictions,
            labels=CLASSES,
        )
    )

    print("\nActual outcomes:")

    print(
        test["FTR"].value_counts()
    )

    print("\nPredicted outcomes:")

    print(
        pd.Series(
            predictions
        ).value_counts()
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "PITCHIQ — CORRECTED MATCH PREDICTION"
    )

    print("-" * 70)

    df = load_data()

    validation_summary = rolling_validation(
        df
    )

    final_test(
        df,
        validation_summary,
    )

    print("\nPIPELINE COMPLETE")

    print(
        "Results saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
