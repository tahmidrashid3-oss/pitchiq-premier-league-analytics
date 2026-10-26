
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "understat_premier_league_xg.csv"

START_YEAR = 2015
END_YEAR = 2026

seasons = [
    f"{year}/{year + 1}"
    for year in range(START_YEAR, END_YEAR + 1)
]

columns = [
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "has_data",
]


def main():
    print("=" * 60)
    print("PITCHIQ — UNDERSTAT XG UPDATE")
    print("=" * 60)

    print("\nDownloading Understat data...")

    understat = sd.Understat(
        leagues="ENG-Premier League",
        seasons=seasons
    )

    matches = understat.read_schedule().reset_index()
    matches = matches[columns].copy()

    matches["season"] = (
        matches["season"].astype(str).str.zfill(4)
    )

    for column in [
        "home_goals", "away_goals",
        "home_xg", "away_xg"
    ]:
        matches[column] = pd.to_numeric(
            matches[column],
            errors="coerce"
        )

    # Keep completed matches with valid scores and xG.
    complete = matches.dropna(
        subset=[
            "home_goals", "away_goals",
            "home_xg", "away_xg"
        ]
    ).copy()

    if (complete[["home_xg", "away_xg"]] < 0).any().any():
        raise ValueError("Negative xG values detected.")

    if complete.duplicated(
        ["season", "home_team", "away_team"]
    ).any():
        raise ValueError("Duplicate Understat fixtures detected.")

    historical = complete[
        complete["season"] != "2627"
    ]

    current = complete[
        complete["season"] == "2627"
    ]

    print("\nCompleted matches with xG per season:")
    print(complete.groupby("season").size())

    print("\nHistorical matches:", len(historical))
    print("2026/27 matches with xG:", len(current))
    print("Total matches with xG:", len(complete))

    if len(historical) != 4180:
        raise ValueError(
            "Historical xG coverage has changed. "
            f"Expected 4180 matches, found {len(historical)}. "
            "Existing file has not been replaced."
        )

    if len(current) == 0:
        raise ValueError(
            "No 2026/27 completed matches with xG found. "
            "Existing file has not been replaced."
        )

    print("\nMissing xG in downloaded schedule:")
    print(matches[["home_xg", "away_xg"]].isna().sum())

    # Save only after historical coverage checks pass.
    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    complete.to_csv(temp_file, index=False)
    temp_file.replace(OUTPUT_FILE)

    print("\nUNDERSTAT UPDATE COMPLETE")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
