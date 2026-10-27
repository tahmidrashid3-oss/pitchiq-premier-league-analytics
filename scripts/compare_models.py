
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss


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

VALIDATION_SEASONS = [
    "2122",
    "2223",
    "2324",
    "2425",
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

    if df.duplicated(
        ["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError(
            "Duplicate fixtures detected."
        )

    if not df["FTR"].isin(CLASSES).all():
        raise ValueError(
            "Invalid match outcomes."
        )

    return df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.01,
                max_iter=3000,
                random_state=42,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),

        "Hist Gradient Boosting":
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, train, validation):

    model.fit(
        train[FEATURES],
        train["FTR"],
    )

    predictions = model.predict(
        validation[FEATURES]
    )

    raw_probs = model.predict_proba(
        validation[FEATURES]
    )

    # Align probability columns explicitly.
    probabilities = pd.DataFrame(
        raw_probs,
        columns=model.classes_,
    )[CLASSES].to_numpy()

    actual = validation["FTR"].to_numpy()

    accuracy = accuracy_score(
        actual,
        predictions,
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
    predicted_draws = predictions == "D"

    draw_recall = (
        np.sum(actual_draws & predicted_draws)
        / np.sum(actual_draws)
        if np.sum(actual_draws) > 0
        else np.nan
    )

    return {
        "Matches": len(validation),
        "Accuracy": accuracy,
        "LogLoss": loss,
        "BrierScore": brier,
        "DrawPredictions": int(predicted_draws.sum()),
        "DrawRecall": draw_recall,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PITCHIQ - MODEL COMPARISON")
    print("=" * 70)

    df = load_data()

    print("\nDataset matches:", len(df))
    print("Features:", len(FEATURES))

    results = []

    for season in VALIDATION_SEASONS:

        train = df[
            df["Season"] < season
        ].copy()

        validation = df[
            df["Season"] == season
        ].copy()

        if train.empty or validation.empty:
            raise ValueError(
                f"Missing data for validation season {season}"
            )

        if set(train["FTR"]) != set(CLASSES):
            raise ValueError(
                f"Training set missing an outcome class: {season}"
            )

        print(
            f"\nValidating {season}: "
            f"{len(train)} training matches, "
            f"{len(validation)} validation matches"
        )

        for name, model in build_models().items():

            scores = evaluate_model(
                model,
                train,
                validation,
            )

            results.append({
                "Season": season,
                "Model": name,
                **scores,
            })

            print(
                f"  {name}: "
                f"Accuracy={scores['Accuracy']:.2%}, "
                f"LogLoss={scores['LogLoss']:.4f}, "
                f"DrawRecall={scores['DrawRecall']:.2%}"
            )

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("Model", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanLogLoss=("LogLoss", "mean"),
            MeanBrierScore=("BrierScore", "mean"),
            MeanDrawRecall=("DrawRecall", "mean"),
            TotalDrawPredictions=("DrawPredictions", "sum"),
            TotalMatches=("Matches", "sum"),
        )
        .sort_values("MeanLogLoss")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    print(
        summary.round(5).to_string(index=False)
    )

    results_file = (
        OUTPUT_DIR / "model_comparison_results.csv"
    )

    summary_file = (
        OUTPUT_DIR / "model_comparison_summary.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\nSaved:")
    print(results_file)
    print(summary_file)

    print("\nMODEL COMPARISON COMPLETE")
    print(
        "The deployed model has not been changed."
    )


if __name__ == "__main__":
    main()
