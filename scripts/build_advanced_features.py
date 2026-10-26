
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "premier_league_with_xg.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "match_features_advanced.csv"
)

WINDOWS = [3, 5, 10]

METRICS = [
    "Points",
    "GoalsFor",
    "GoalsAgainst",
    "GoalDifference",
    "xGFor",
    "xGAgainst",
    "xGDifference",
]

# Neutral starting values for teams with no history.
# These are modelling assumptions, not fitted averages.
DEFAULTS = {
    "Points": 1.0,
    "GoalsFor": 1.3,
    "GoalsAgainst": 1.3,
    "GoalDifference": 0.0,
    "xGFor": 1.3,
    "xGAgainst": 1.3,
    "xGDifference": 0.0,
}


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_matches():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    df["Season"] = df["Season"].str.zfill(4)

    required = [
        "Date", "Season", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "HomeXG", "AwayXG",
    ]

    missing = [c for c in required if c not in df]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError("Missing values in match data.")

    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Invalid match result.")

    if df.duplicated(
        ["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError("Duplicate fixtures detected.")

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    df["MatchID"] = np.arange(len(df))

    return df


# ============================================================
# CREATE TEAM-PERSPECTIVE MATCH HISTORY
# ============================================================

def create_team_history(matches):

    home = pd.DataFrame({
        "MatchID": matches["MatchID"],
        "Date": matches["Date"],
        "Team": matches["HomeTeam"],
        "Venue": "Home",
        "GoalsFor": matches["FTHG"],
        "GoalsAgainst": matches["FTAG"],
        "xGFor": matches["HomeXG"],
        "xGAgainst": matches["AwayXG"],
    })

    away = pd.DataFrame({
        "MatchID": matches["MatchID"],
        "Date": matches["Date"],
        "Team": matches["AwayTeam"],
        "Venue": "Away",
        "GoalsFor": matches["FTAG"],
        "GoalsAgainst": matches["FTHG"],
        "xGFor": matches["AwayXG"],
        "xGAgainst": matches["HomeXG"],
    })

    history = pd.concat(
        [home, away],
        ignore_index=True,
    )

    history["GoalDifference"] = (
        history["GoalsFor"] - history["GoalsAgainst"]
    )

    history["xGDifference"] = (
        history["xGFor"] - history["xGAgainst"]
    )

    history["Points"] = np.select(
        [
            history["GoalsFor"] > history["GoalsAgainst"],
            history["GoalsFor"] == history["GoalsAgainst"],
        ],
        [3, 1],
        default=0,
    )

    history = history.sort_values(
        ["Team", "Date", "MatchID"]
    ).reset_index(drop=True)

    return history


# ============================================================
# LEAKAGE-SAFE ROLLING STATISTICS
# ============================================================

def add_rolling_features(history):
    """
    Calculate statistics using only earlier match dates.

    Teams occasionally have multiple records on one date.
    Grouping by date ensures that none of those matches
    contributes to another match on the same date.
    """

    # Create one row per team and calendar date.
    daily = (
        history
        .groupby(["Team", "Date"], as_index=False)
        .agg(
            MatchesOnDate=("MatchID", "count"),
            **{
                metric: (metric, "sum")
                for metric in METRICS
            },
        )
        .sort_values(["Team", "Date"])
        .reset_index(drop=True)
    )

    # Expand daily match counts into prior-history counts.
    daily["PriorMatches"] = (
        daily.groupby("Team")["MatchesOnDate"]
        .transform(lambda s: s.cumsum() - s)
    )

    for window in WINDOWS:
        for metric in METRICS:
            column = f"Overall_{metric}_Last{window}"

            daily[column] = (
                daily.groupby("Team")[metric]
                .transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(
                            window,
                            min_periods=1,
                        )
                        .mean()
                    )
                )
            )

    overall_columns = [
        "Team", "Date", "PriorMatches"
    ] + [
        f"Overall_{metric}_Last{window}"
        for window in WINDOWS
        for metric in METRICS
    ]

    history = history.merge(
        daily[overall_columns],
        on=["Team", "Date"],
        how="left",
        validate="many_to_one",
    )

    # Separate history by venue.
    venue_daily = (
        history
        .groupby(
            ["Team", "Venue", "Date"],
            as_index=False,
        )
        .agg(
            MatchesOnDate=("MatchID", "count"),
            **{
                metric: (metric, "sum")
                for metric in METRICS
            },
        )
        .sort_values(["Team", "Venue", "Date"])
        .reset_index(drop=True)
    )

    venue_daily["PriorVenueMatches"] = (
        venue_daily
        .groupby(["Team", "Venue"])["MatchesOnDate"]
        .transform(lambda s: s.cumsum() - s)
    )

    for window in WINDOWS:
        for metric in METRICS:
            column = f"Venue_{metric}_Last{window}"

            venue_daily[column] = (
                venue_daily
                .groupby(["Team", "Venue"])[metric]
                .transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(
                            window,
                            min_periods=1,
                        )
                        .mean()
                    )
                )
            )

    venue_columns = [
        "Team", "Venue", "Date", "PriorVenueMatches"
    ] + [
        f"Venue_{metric}_Last{window}"
        for window in WINDOWS
        for metric in METRICS
    ]

    history = history.merge(
        venue_daily[venue_columns],
        on=["Team", "Venue", "Date"],
        how="left",
        validate="many_to_one",
    )

    # Replace missing prior-history features with neutral
    # values. This retains newly promoted teams' matches.
    for window in WINDOWS:
        for metric in METRICS:
            for prefix in ["Overall", "Venue"]:
                column = (
                    f"{prefix}_{metric}_Last{window}"
                )
                history[column] = (
                    history[column].fillna(DEFAULTS[metric])
                )

    return history


# ============================================================
# BUILD MATCH-LEVEL FEATURES
# ============================================================

def build_match_features(matches, history):

    feature_columns = [
        "PriorMatches",
        "PriorVenueMatches",
    ] + [
        f"{prefix}_{metric}_Last{window}"
        for prefix in ["Overall", "Venue"]
        for window in WINDOWS
        for metric in METRICS
    ]

    home = history[
        history["Venue"] == "Home"
    ][["MatchID"] + feature_columns].copy()

    away = history[
        history["Venue"] == "Away"
    ][["MatchID"] + feature_columns].copy()

    home = home.rename(
        columns={
            column: f"Home_{column}"
            for column in feature_columns
        }
    )

    away = away.rename(
        columns={
            column: f"Away_{column}"
            for column in feature_columns
        }
    )

    output = matches[
        [
            "MatchID", "Date", "Season",
            "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR",
        ]
    ].copy()

    output = output.merge(
        home,
        on="MatchID",
        how="left",
        validate="one_to_one",
    )

    output = output.merge(
        away,
        on="MatchID",
        how="left",
        validate="one_to_one",
    )

    # Home minus away feature differences.
    for window in WINDOWS:
        for metric in METRICS:
            for prefix in ["Overall", "Venue"]:
                source = (
                    f"{prefix}_{metric}_Last{window}"
                )

                output[
                    f"Diff_{source}"
                ] = (
                    output[f"Home_{source}"]
                    - output[f"Away_{source}"]
                )

    # Availability indicators.
    for side in ["Home", "Away"]:
        output[f"{side}_Has5Matches"] = (
            output[f"{side}_PriorMatches"] >= 5
        ).astype(int)

        output[f"{side}_Has5VenueMatches"] = (
            output[f"{side}_PriorVenueMatches"] >= 5
        ).astype(int)

    if len(output) != len(matches):
        raise ValueError(
            "Match count changed during feature merging."
        )

    if output.isna().any().any():
        raise ValueError(
            "Missing values remain in advanced features."
        )

    return output.drop(columns=["MatchID"])


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PITCHIQ — ADVANCED FEATURE ENGINEERING")
    print("=" * 65)

    matches = load_matches()

    print("Original matches:", len(matches))

    history = create_team_history(matches)

    history = add_rolling_features(history)

    features = build_match_features(
        matches,
        history,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Output matches:", len(features))
    print("Feature columns:", len(features.columns))

    print("\nMatches by season:")
    print(
        features.groupby("Season")
        .size()
        .to_string()
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nADVANCED FEATURE ENGINEERING COMPLETE")


if __name__ == "__main__":
    main()
