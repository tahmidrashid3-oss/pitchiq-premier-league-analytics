
from pathlib import Path
from io import BytesIO

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2015
END_YEAR = 2026

REQUIRED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR"
]


def download_season(year):
    season = f"{year % 100:02d}{(year + 1) % 100:02d}"
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

    print(f"\nDownloading season {season}...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(BytesIO(response.content))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {season}: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    # In an ongoing season, future fixtures may have no final score.
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"]).copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="raise"
    )

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="raise")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="raise")

    if (df[["FTHG", "FTAG"]] < 0).any().any():
        raise ValueError(f"Negative goals in {season}")

    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError(f"Invalid results in {season}")

    expected_results = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"]
        else "A" if r["FTHG"] < r["FTAG"]
        else "D",
        axis=1
    )

    if not expected_results.eq(df["FTR"]).all():
        raise ValueError(f"Score/result mismatch in {season}")

    if df.duplicated(["HomeTeam", "AwayTeam"]).any():
        raise ValueError(f"Duplicate fixtures in {season}")

    if year < END_YEAR and len(df) != 380:
        raise ValueError(
            f"Expected 380 matches for {season}, found {len(df)}"
        )

    if year == END_YEAR and not 0 < len(df) <= 380:
        raise ValueError(
            f"Invalid current-season match count: {len(df)}"
        )

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["Season"] = season

    output = RAW_DIR / f"premier_league_{season}.csv"
    df.to_csv(output, index=False)

    print(f"Saved {len(df)} completed matches")
    return df


def main():
    all_seasons = [
        download_season(year)
        for year in range(START_YEAR, END_YEAR + 1)
    ]

    combined = pd.concat(all_seasons, ignore_index=True)

    print("\nDOWNLOAD COMPLETE")
    print("Total matches:", len(combined))
    print("Total seasons:", combined["Season"].nunique())
    print("\nMatches per season:")
    print(combined.groupby("Season").size())
    print("\nLatest match:", combined["Date"].max())


if __name__ == "__main__":
    main()
