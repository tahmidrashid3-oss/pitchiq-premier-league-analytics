
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

seasons = [
    f"{year}/{year + 1}"
    for year in range(2015, 2026)
]

print("Downloading Understat xG data...")

understat = sd.Understat(
    leagues="ENG-Premier League",
    seasons=seasons
)

matches = understat.read_schedule().reset_index()

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

matches = matches[columns].copy()

matches["home_xg"] = pd.to_numeric(
    matches["home_xg"], errors="coerce"
)
matches["away_xg"] = pd.to_numeric(
    matches["away_xg"], errors="coerce"
)

output_file = OUTPUT_DIR / "understat_premier_league_xg.csv"
matches.to_csv(output_file, index=False)

print("\nTotal matches:", len(matches))
print("\nMatches per season:")
print(matches.groupby("season").size())

print("\nMissing xG values:")
print(matches[["home_xg", "away_xg"]].isna().sum())

print("\nSaved to:", output_file)
