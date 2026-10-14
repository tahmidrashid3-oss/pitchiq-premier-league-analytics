
from pathlib import Path
from io import BytesIO

import pandas as pd
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_URL = "https://www.football-data.co.uk/mmz4281"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Premier League seasons: 2015/16 to 2025/26
SEASONS = [
    f"{year:02d}{(year + 1) % 100:02d}"
    for year in range(15, 26)
]

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR"
]


# --------------------------------------------------
# Download and clean a single season
# --------------------------------------------------

def download_season(season):
    """Download and validate one Premier League season."""

    url = f"{BASE_URL}/{season}/E0.csv"

    print(f"Downloading season {season}...")

    # Download CSV data
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Load CSV into a DataFrame
    df = pd.read_csv(BytesIO(response.content))

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Season {season} is missing columns: {missing}"
        )

    # Keep only the columns needed for analysis
    df = df[REQUIRED_COLUMNS].copy()

    # Remove rows with missing essential information
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Convert match dates to datetime
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # Convert goals to numeric values
    df["FTHG"] = pd.to_numeric(
        df["FTHG"],
        errors="coerce"
    )

    df["FTAG"] = pd.to_numeric(
        df["FTAG"],
        errors="coerce"
    )

    # Remove rows with invalid dates or scores
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])

    # Add season identifier
    df["Season"] = season

    # Sort matches chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# --------------------------------------------------
# Download all seasons
# --------------------------------------------------

def main():

    # Create raw data directory if necessary
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_seasons = []

    for season in SEASONS:

        df = download_season(season)

        # Save season data
        output = RAW_DIR / f"premier_league_{season}.csv"

        df.to_csv(output, index=False)

        print(
            f"Saved {len(df)} matches "
            f"to {output.name}"
        )

        all_seasons.append(df)

    # Combine all seasons
    combined = pd.concat(
        all_seasons,
        ignore_index=True
    )

    # Sort complete dataset by date
    combined = combined.sort_values(
        "Date"
    ).reset_index(drop=True)

    print("\nDownload complete!")
    print(f"Total matches: {len(combined)}")
    print(
        f"Total seasons: "
        f"{combined['Season'].nunique()}"
    )

    print("\nFirst five matches:")
    print(combined.head())

    print("\nLast five matches:")
    print(combined.tail())


# --------------------------------------------------
# Run script
# --------------------------------------------------

if __name__ == "__main__":
    main()
