
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "premier_league_with_xg.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "match_features.csv"
)

WINDOW = 5


def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"Season": str},
        parse_dates=["Date"]
    )

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    df["MatchID"] = np.arange(len(df))

    home = pd.DataFrame({
        "MatchID": df["MatchID"],
        "Date": df["Date"],
        "Season": df["Season"],
        "Team": df["HomeTeam"],
        "Opponent": df["AwayTeam"],
        "GoalsFor": df["FTHG"],
        "GoalsAgainst": df["FTAG"],
        "xGFor": df["HomeXG"],
        "xGAgainst": df["AwayXG"],
    })

    away = pd.DataFrame({
        "MatchID": df["MatchID"],
        "Date": df["Date"],
        "Season": df["Season"],
        "Team": df["AwayTeam"],
        "Opponent": df["HomeTeam"],
        "GoalsFor": df["FTAG"],
        "GoalsAgainst": df["FTHG"],
        "xGFor": df["AwayXG"],
        "xGAgainst": df["HomeXG"],
    })

    team_matches = pd.concat(
        [home, away],
        ignore_index=True
    )

    team_matches["Points"] = np.select(
        [
            team_matches["GoalsFor"] > team_matches["GoalsAgainst"],
            team_matches["GoalsFor"] == team_matches["GoalsAgainst"]
        ],
        [3, 1],
        default=0
    )

    team_matches["GoalDifference"] = (
        team_matches["GoalsFor"]
        - team_matches["GoalsAgainst"]
    )

    team_matches["xGDifference"] = (
        team_matches["xGFor"]
        - team_matches["xGAgainst"]
    )

    team_matches = team_matches.sort_values(
        ["Team", "Date", "MatchID"]
    ).reset_index(drop=True)

    metrics = [
        "Points",
        "GoalsFor",
        "GoalsAgainst",
        "GoalDifference",
        "xGFor",
        "xGAgainst",
        "xGDifference"
    ]

    # Build features only from matches on earlier dates.
    # This also prevents leakage between same-day fixtures.
    daily = (
        team_matches
        .groupby(["Team", "Date"], as_index=False)[metrics]
        .sum()
        .sort_values(["Team", "Date"])
    )

    for metric in metrics:
        daily[f"Rolling_{metric}"] = (
            daily.groupby("Team")[metric]
            .transform(
                lambda series: (
                    series.shift(1)
                    .rolling(WINDOW, min_periods=WINDOW)
                    .mean()
                )
            )
        )

    rolling_columns = [
        f"Rolling_{metric}" for metric in metrics
    ]

    team_matches = team_matches.merge(
        daily[["Team", "Date"] + rolling_columns],
        on=["Team", "Date"],
        how="left",
        validate="many_to_one"
    )

    home_features = team_matches[
        ["MatchID", "Team"] + rolling_columns
    ].merge(
        df[["MatchID", "HomeTeam"]],
        left_on=["MatchID", "Team"],
        right_on=["MatchID", "HomeTeam"],
        how="inner",
        validate="one_to_one"
    )

    away_features = team_matches[
        ["MatchID", "Team"] + rolling_columns
    ].merge(
        df[["MatchID", "AwayTeam"]],
        left_on=["MatchID", "Team"],
        right_on=["MatchID", "AwayTeam"],
        how="inner",
        validate="one_to_one"
    )

    home_features = home_features[
        ["MatchID"] + rolling_columns
    ].rename(
        columns={
            col: f"Home_{col}"
            for col in rolling_columns
        }
    )

    away_features = away_features[
        ["MatchID"] + rolling_columns
    ].rename(
        columns={
            col: f"Away_{col}"
            for col in rolling_columns
        }
    )

    features = df[
        ["MatchID", "Date", "Season",
         "HomeTeam", "AwayTeam", "FTR"]
    ].copy()

    features = features.merge(
        home_features,
        on="MatchID",
        validate="one_to_one"
    ).merge(
        away_features,
        on="MatchID",
        validate="one_to_one"
    )

    # Differences between the two teams' recent form.
    for metric in metrics:
        features[f"Diff_{metric}"] = (
            features[f"Home_Rolling_{metric}"]
            - features[f"Away_Rolling_{metric}"]
        )

    # Keep only fixtures where both teams have
    # five earlier matches available.
    features = features.dropna().reset_index(drop=True)

    features = features.sort_values(
        ["Date", "MatchID"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(OUTPUT_FILE, index=False)

    print("PITCHIQ FEATURE ENGINEERING")
    print("-" * 45)
    print("Original matches:", len(df))
    print("Matches with complete features:", len(features))
    print("Removed warm-up matches:", len(df) - len(features))
    print("Feature columns:", len(features.columns) - 6)
    print("Missing values:", features.isna().sum().sum())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
