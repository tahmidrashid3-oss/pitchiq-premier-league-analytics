
"""Build a validated Premier League preview without modifying production."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[1]
EXISTING_FILE = ROOT / "data" / "processed" / "premier_league_with_xg.csv"
PREVIEW_DIR = ROOT / "backups" / "preview"
PREVIEW_FILE = PREVIEW_DIR / "premier_league_with_xg_preview.csv"

SEASON = "2627"
SEASON_LABEL = "2026/2027"

KEYS = ["Season", "HomeTeam", "AwayTeam"]
COLUMNS = [
    "Date", "Season", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HomeXG", "AwayXG"
]

TEAM_NAMES = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
}


def main():
    print("=" * 60)
    print("PITCHIQ - SAFE DATASET PREVIEW")
    print("=" * 60)

    existing = pd.read_csv(EXISTING_FILE, dtype={"Season": str})
    existing["Season"] = existing["Season"].str.zfill(4)

    if list(existing.columns) != COLUMNS:
        raise ValueError("Unexpected production dataset columns.")

    if existing.duplicated(KEYS).any():
        raise ValueError("Existing dataset contains duplicate fixtures.")

    if existing.isna().any().any():
        raise ValueError("Existing dataset contains missing values.")

    print(f"Original matches: {len(existing):,}")

    url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/E0.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    results = pd.read_csv(BytesIO(response.content))
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    if not set(required).issubset(results.columns):
        raise ValueError("Football-Data columns have changed.")

    results = results[required].dropna(
        subset=["FTHG", "FTAG", "FTR"]
    ).copy()

    results["Date"] = pd.to_datetime(
        results["Date"], format="mixed", dayfirst=True
    ).dt.strftime("%Y-%m-%d")

    for col in ["FTHG", "FTAG"]:
        values = pd.to_numeric(results[col], errors="raise")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"Invalid {col} values.")
        results[col] = values.astype(int)

    if not results["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Invalid match result.")

    expected_result = results.apply(
        lambda row: (
            "H" if row["FTHG"] > row["FTAG"]
            else "A" if row["FTHG"] < row["FTAG"]
            else "D"
        ),
        axis=1,
    )

    if not expected_result.eq(results["FTR"]).all():
        raise ValueError("Score/result mismatch.")

    if results.duplicated(["HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate downloaded fixtures.")

    if not 0 < len(results) <= 380:
        raise ValueError("Invalid current-season match count.")

    results["Season"] = SEASON

    previous = existing[existing["Season"] == SEASON]
    comparison = previous.merge(
        results,
        on=["HomeTeam", "AwayTeam"],
        how="left",
        suffixes=("_old", "_new"),
        validate="one_to_one",
        indicator=True,
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError("Previously saved fixtures disappeared.")

    for col in ["Date", "FTHG", "FTAG", "FTR"]:
        if not comparison[f"{col}_old"].eq(
            comparison[f"{col}_new"]
        ).all():
            raise ValueError(f"Previously saved {col} changed.")

    existing_keys = pd.MultiIndex.from_frame(existing[KEYS])
    result_keys = pd.MultiIndex.from_frame(results[KEYS])
    new = results.loc[~result_keys.isin(existing_keys)].copy()

    print(f"New results: {len(new)}")

    if new.empty:
        print("No new matches. Creating an unchanged preview.")
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        existing.to_csv(PREVIEW_FILE, index=False)
        print(f"Preview saved to: {PREVIEW_FILE}")
        print("Production dataset and model remain unchanged.")
        return

    new["HomeTeam_standard"] = new["HomeTeam"].replace(TEAM_NAMES)
    new["AwayTeam_standard"] = new["AwayTeam"].replace(TEAM_NAMES)

    understat = sd.Understat(
        leagues="ENG-Premier League",
        seasons=[SEASON_LABEL],
    )

    xg = understat.read_schedule().reset_index()
    required_xg = [
        "season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg"
    ]

    if not set(required_xg).issubset(xg.columns):
        raise ValueError("Understat columns have changed.")

    xg = xg[required_xg].copy()
    xg["season"] = xg["season"].astype(str).str.zfill(4)
    xg = xg[xg["season"] == SEASON].copy()

    if xg.duplicated(["season", "home_team", "away_team"]).any():
        raise ValueError("Duplicate Understat fixtures.")

    for col in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        xg[col] = pd.to_numeric(xg[col], errors="coerce")

    merged = new.merge(
        xg,
        left_on=["Season", "HomeTeam_standard", "AwayTeam_standard"],
        right_on=["season", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError("Some new matches are missing from Understat.")

    required_values = [
        "home_goals", "away_goals", "home_xg", "away_xg"
    ]

    if merged[required_values].isna().any().any():
        raise ValueError("Some new matches lack scores or xG.")

    if (merged[["home_xg", "away_xg"]] < 0).any().any():
        raise ValueError("Negative xG values.")

    if not (
        merged["FTHG"].eq(merged["home_goals"])
        & merged["FTAG"].eq(merged["away_goals"])
    ).all():
        raise ValueError("Scores disagree between providers.")

    understat_dates = pd.to_datetime(
        merged["date"], utc=True, errors="raise"
    ).dt.tz_convert(None).dt.normalize()

    football_dates = pd.to_datetime(
        merged["Date"], errors="raise"
    ).dt.normalize()

    if ((football_dates - understat_dates).abs().dt.days > 1).any():
        raise ValueError("Dates disagree between providers.")

    additions = merged[
        [
            "Date", "Season", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR", "home_xg", "away_xg"
        ]
    ].rename(
        columns={"home_xg": "HomeXG", "away_xg": "AwayXG"}
    )

    preview = pd.concat(
        [existing, additions[COLUMNS]],
        ignore_index=True
    )

    preview = preview.sort_values(
        ["Date", "Season", "HomeTeam"]
    ).reset_index(drop=True)

    if preview.duplicated(KEYS).any():
        raise ValueError("Preview contains duplicate fixtures.")

    if preview.isna().any().any():
        raise ValueError("Preview contains missing values.")

    if len(preview) != len(existing) + len(additions):
        raise ValueError("Preview match count is incorrect.")

    verified = existing.merge(
        preview,
        on=KEYS,
        how="left",
        suffixes=("_old", "_preview"),
        validate="one_to_one",
        indicator=True,
    )

    if not verified["_merge"].eq("both").all():
        raise ValueError("An original fixture disappeared.")

    for col in ["Date", "FTHG", "FTAG", "FTR", "HomeXG", "AwayXG"]:
        if not verified[f"{col}_old"].eq(
            verified[f"{col}_preview"]
        ).all():
            raise ValueError(f"Original {col} values changed.")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview.to_csv(PREVIEW_FILE, index=False)

    print("\nNEW MATCHES")
    print(
        additions[
            ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        ].to_string(index=False)
    )

    print("\nVALIDATION SUCCESSFUL")
    print(f"Original matches preserved: {len(existing)}")
    print(f"New matches added: {len(additions)}")
    print(f"Preview total: {len(preview)}")
    print(f"Latest match: {preview['Date'].max()}")
    print(f"Preview saved to: {PREVIEW_FILE}")
    print("Production dataset and model remain unchanged.")


if __name__ == "__main__":
    main()
