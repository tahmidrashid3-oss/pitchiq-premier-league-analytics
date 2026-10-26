
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_SEASONS = ["2122", "2223", "2324", "2425"]
TEST_SEASON = "2526"

CLASSES = ["H", "D", "A"]
METRICS = ["Points", "GoalsFor", "GoalsAgainst",
           "xGFor", "xGAgainst"]
WINDOWS = [3, 5, 10]
C = 0.01

KEY = ["Season", "HomeTeam", "AwayTeam"]

ORIGINAL = [
    "Diff_Points",
    "Diff_GoalsFor",
    "Diff_GoalsAgainst",
    "Diff_xGFor",
    "Diff_xGAgainst",
]

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

HISTORY = [
    "Home_PriorMatches",
    "Away_PriorMatches",
    "Home_PriorVenueMatches",
    "Away_PriorVenueMatches",
    "Home_Has5Matches",
    "Away_Has5Matches",
    "Home_Has5VenueMatches",
    "Away_Has5VenueMatches",
]

FEATURE_SETS = {
    "Original xG": ORIGINAL,
    "Advanced overall": OVERALL,
    "Advanced venue": OVERALL + VENUE,
    "Advanced full": OVERALL + VENUE + HISTORY,
}


def load_data():
    original = pd.read_csv(
        DATA_DIR / "match_features.csv",
        dtype={"Season": str},
    )

    advanced = pd.read_csv(
        DATA_DIR / "match_features_advanced.csv",
        dtype={"Season": str},
    )

    for df in [original, advanced]:
        df["Season"] = df["Season"].str.zfill(4)

        if df.duplicated(KEY).any():
            raise ValueError("Duplicate fixtures detected.")

    if len(advanced) != 4180:
        raise ValueError("Advanced dataset must have 4180 matches.")

    # Rename original features to avoid column collisions.
    original = original.rename(
        columns={col: f"Old_{col}" for col in ORIGINAL}
    )

    original_columns = KEY + ["FTR"] + [
        f"Old_{col}" for col in ORIGINAL
    ]

    common = advanced.merge(
        original[original_columns],
        on=KEY,
        how="inner",
        suffixes=("", "_old"),
        validate="one_to_one",
    )

    if not (common["FTR"] == common["FTR_old"]).all():
        raise ValueError("Outcome mismatch between datasets.")

    if common[KEY].isna().any().any():
        raise ValueError("Invalid fixture keys.")

    # Original features are renamed only in the common set.
    common_features = {
        "Original xG": [
            f"Old_{col}" for col in ORIGINAL
        ],
        "Advanced overall": OVERALL,
        "Advanced venue": OVERALL + VENUE,
        "Advanced full": OVERALL + VENUE + HISTORY,
    }

    print("Original dataset:", len(original))
    print("Advanced dataset:", len(advanced))
    print("Common fixtures:", len(common))

    return common, advanced, common_features


def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            max_iter=3000,
            random_state=42,
        )),
    ])


def evaluate(train, test, features):
    model = create_model()

    model.fit(train[features], train["FTR"])

    raw = model.predict_proba(test[features])

    # Explicitly align probability columns to H, D, A.
    probs = pd.DataFrame(
        raw, columns=model.classes_
    )[CLASSES].to_numpy()

    predicted = np.array(CLASSES)[
        np.argmax(probs, axis=1)
    ]

    actual = test["FTR"].to_numpy()

    # Direct multiclass log loss avoids class-order errors.
    actual_indices = np.array([
        CLASSES.index(label) for label in actual
    ])

    actual_probs = probs[
        np.arange(len(actual)), actual_indices
    ]

    loss = -np.log(
        np.clip(actual_probs, 1e-15, 1.0)
    ).mean()

    one_hot = (
        actual[:, None] == np.array(CLASSES)[None, :]
    ).astype(float)

    brier = np.mean(
        np.sum((probs - one_hot) ** 2, axis=1)
    )

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "LogLoss": loss,
        "BrierScore": brier,
        "DrawPredictions": int((predicted == "D").sum()),
        "Matches": len(test),
    }


def rolling_evaluation(df, feature_sets, coverage):
    results = []

    for season in VALIDATION_SEASONS:
        train = df[df["Season"] < season]
        validation = df[df["Season"] == season]

        for name, features in feature_sets.items():
            if train[features].isna().any().any():
                raise ValueError(f"Missing training features: {name}")

            if validation[features].isna().any().any():
                raise ValueError(f"Missing validation features: {name}")

            scores = evaluate(train, validation, features)

            results.append({
                "Coverage": coverage,
                "Season": season,
                "Configuration": name,
                **scores,
            })

    return pd.DataFrame(results)


def summarise(results):
    return (
        results.groupby(
            ["Coverage", "Configuration"],
            as_index=False,
        )
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanLogLoss=("LogLoss", "mean"),
            MeanBrierScore=("BrierScore", "mean"),
            TotalDrawPredictions=("DrawPredictions", "sum"),
            TotalMatches=("Matches", "sum"),
        )
        .sort_values(["Coverage", "MeanLogLoss"])
        .reset_index(drop=True)
    )


def test_evaluation(df, feature_sets, coverage):
    train = df[df["Season"] < TEST_SEASON]
    test = df[df["Season"] == TEST_SEASON]

    rows = []

    for name, features in feature_sets.items():
        scores = evaluate(train, test, features)

        rows.append({
            "Coverage": coverage,
            "Configuration": name,
            **scores,
        })

    return pd.DataFrame(rows)


def make_plot(summary):
    common = summary[
        summary["Coverage"] == "Common fixtures"
    ].sort_values("MeanLogLoss")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        common["Configuration"],
        common["MeanLogLoss"],
    )

    ax.invert_yaxis()
    ax.set_xlabel("Mean Validation Log Loss (lower is better)")
    ax.set_title("Original vs Advanced Features: Rolling Validation")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "advanced_feature_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    print("=" * 70)
    print("PITCHIQ — ADVANCED FEATURE EVALUATION")
    print("=" * 70)

    common, advanced, common_features = load_data()

    # Fair comparison: identical fixtures for every model.
    common_results = rolling_evaluation(
        common, common_features, "Common fixtures"
    )

    # Coverage experiment: all available matches.
    advanced_sets = {
        name: features
        for name, features in FEATURE_SETS.items()
        if name != "Original xG"
    }

    full_results = rolling_evaluation(
        advanced, advanced_sets, "All fixtures"
    )

    results = pd.concat(
        [common_results, full_results],
        ignore_index=True,
    )

    summary = summarise(results)

    results.to_csv(
        TABLES_DIR / "advanced_feature_validation_results.csv",
        index=False,
    )

    summary.to_csv(
        TABLES_DIR / "advanced_feature_validation_summary.csv",
        index=False,
    )

    print("\nROLLING VALIDATION SUMMARY")
    print(summary.round(5).to_string(index=False))

    # Test evaluation is diagnostic, not for feature selection.
    # Select the preferred configuration using validation only.
    common_test = test_evaluation(
        common, common_features, "Common fixtures"
    )

    full_test = test_evaluation(
        advanced, advanced_sets, "All fixtures"
    )

    test_results = pd.concat(
        [common_test, full_test],
        ignore_index=True,
    )

    test_results.to_csv(
        TABLES_DIR / "advanced_feature_test_comparison.csv",
        index=False,
    )

    print("\nFINAL TEST COMPARISON")
    print(test_results.round(5).to_string(index=False))

    make_plot(summary)

    print("\nFiles saved to:", TABLES_DIR)
    print("Figure saved to:", FIGURES_DIR)
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
