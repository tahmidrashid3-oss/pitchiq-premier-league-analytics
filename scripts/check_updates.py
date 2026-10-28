
"""Read-only check for new Premier League results and matching Understat xG."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "premier_league_with_xg.csv"

SEASON = "2627"
SEASON_LABEL = "2026/2027"
KEYS = ["Season", "HomeTeam", "AwayTeam"]

TEAM_NAMES = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
}


def download_results():
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

    expected = results.apply(
        lambda r: (
            "H" if r["FTHG"] > r["FTAG"]
            else "A" if r["FTHG"] < r["FTAG"]
            else "D"
        ),
        axis=1,
    )

    if not expected.eq(results["FTR"]).all():
        raise ValueError("Score/result mismatch.")

    if results.duplicated(["HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate fixtures from Football-Data.")

    if not 0 < len(results) <= 380:
        raise ValueError("Invalid current-season match count.")

    results["Season"] = SEASON
    return results


def main():
    print("=" * 60)
    print("PITCHIQ - READ-ONLY UPDATE CHECK")
    print("=" * 60)

    existing = pd.read_csv(DATASET, dtype={"Season": str})
    existing["Season"] = existing["Season"].str.zfill(4)

    if existing.duplicated(KEYS).any():
        raise ValueError("Existing dataset contains duplicate fixtures.")

    print(f"Existing matches: {len(existing):,}")
    print(f"Latest saved match: {existing['Date'].max()}")

    results = download_results()
    print(f"Completed matches available: {len(results)}")

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

    print(f"New results found: {len(new)}")

    if new.empty:
        print("No new matches to process.")
        print("Existing files remain unchanged.")
        return

    new["HomeTeam_standard"] = new["HomeTeam"].replace(TEAM_NAMES)
    new["AwayTeam_standard"] = new["AwayTeam"].replace(TEAM_NAMES)

    understat = sd.Understat(
        leagues="ENG-Premier League",
        seasons=[SEASON_LABEL],
    )
    xg = understat.read_schedule().reset_index()

    required_xg = [
        "season", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg"
    ]
    if not set(required_xg).issubset(xg.columns):
        raise ValueError("Understat columns have changed.")

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
    )

    ready = 0
    waiting = 0

    print("\nNEW MATCH REPORT")
    print("-" * 60)

    for _, match in merged.iterrows():
        valid_xg = (
            pd.notna(match["home_xg"])
            and pd.notna(match["away_xg"])
            and match["home_xg"] >= 0
            and match["away_xg"] >= 0
        )
        valid_score = (
            pd.notna(match["home_goals"])
            and pd.notna(match["away_goals"])
            and match["FTHG"] == match["home_goals"]
            and match["FTAG"] == match["away_goals"]
        )

        if valid_xg and valid_score:
            status = "READY"
            ready += 1
        else:
            status = "WAITING / CHECK REQUIRED"
            waiting += 1

        print(
            f"{match['Date']} | "
            f"{match['HomeTeam']} {match['FTHG']}-{match['FTAG']} "
            f"{match['AwayTeam']} | {status}"
        )

    print("\nSUMMARY")
    print(f"New matches: {len(new)}")
    print(f"Ready: {ready}")
    print(f"Waiting or requiring investigation: {waiting}")
    print("No project datasets or models were modified.")


if __name__ == "__main__":
    main()
