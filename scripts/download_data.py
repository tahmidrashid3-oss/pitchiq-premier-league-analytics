
from pathlib import Path
from io import BytesIO

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2015
END_YEAR = 2025

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]


# ============================================================
# DOWNLOAD PREMIER LEAGUE DATA
# ============================================================

def download_season(year):
    season = f"{year % 100:02d}{(year + 1) % 100:02d}"

    url = (
        f"https://www.football-data.co.uk/mmz4281/"
        f"{season}/E0.csv"
    )

    print(f"\nDownloading season {season}...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(BytesIO(response.content))

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in season {season}: "
            f"{missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Parse the ORIGINAL Football-Data dates as UK dates.
    # This prevents day/month values being reversed.
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="raise"
    )

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df["FTHG"] = pd.to_numeric(
        df["FTHG"],
        errors="raise"
    )

    df["FTAG"] = pd.to_numeric(
        df["FTAG"],
        errors="raise"
    )

    df["Season"] = season

    if len(df) != 380:
        raise ValueError(
            f"Expected 380 matches for {season}, "
            f"found {len(df)}."
        )

    if df.isna().any().any():
        raise ValueError(
            f"Missing values detected in season {season}."
        )

    output_file = (
        RAW_DIR / f"premier_league_{season}.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"Saved {len(df)} matches to {output_file.name}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    all_seasons = []

    for year in range(START_YEAR, END_YEAR + 1):
        season_df = download_season(year)
        all_seasons.append(season_df)

    combined = pd.concat(
        all_seasons,
        ignore_index=True
    )

    print("\nDOWNLOAD COMPLETE")
    print("Total matches:", len(combined))
    print("Total seasons:", combined["Season"].nunique())

    print("\nMatches per season:")
    print(combined.groupby("Season").size())

    print("\nMissing values:")
    print(combined.isna().sum())


if __name__ == "__main__":
    main()
