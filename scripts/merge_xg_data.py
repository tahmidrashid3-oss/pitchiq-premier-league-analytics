
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2015
END_YEAR = 2026
CURRENT_SEASON = "2627"

EXPECTED_SEASONS = {
    f"{year % 100:02d}{(year + 1) % 100:02d}"
    for year in range(START_YEAR, END_YEAR + 1)
}

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
    print("PITCHIQ — RESULTS AND XG MERGE")
    print("=" * 60)

    # 1. Load Football-Data results.
    files = sorted(RAW_DIR.glob("premier_league_*.csv"))

    found_seasons = {
        file.stem.replace("premier_league_", "")
        for file in files
    }

    if found_seasons != EXPECTED_SEASONS:
        raise ValueError(
            "Season files do not match expectations.\n"
            f"Missing: {sorted(EXPECTED_SEASONS - found_seasons)}\n"
            f"Unexpected: {sorted(found_seasons - EXPECTED_SEASONS)}"
        )

    results = pd.concat(
        [
            pd.read_csv(file, dtype={"Season": str})
            for file in files
        ],
        ignore_index=True
    )

    results["Season"] = results["Season"].str.zfill(4)
    results["Date"] = pd.to_datetime(
        results["Date"], format="%Y-%m-%d", errors="raise"
    )

    for column in ["FTHG", "FTAG"]:
        results[column] = pd.to_numeric(
            results[column], errors="raise"
        )

    results["HomeTeam_standard"] = (
        results["HomeTeam"].replace(TEAM_NAMES)
    )
    results["AwayTeam_standard"] = (
        results["AwayTeam"].replace(TEAM_NAMES)
    )

    historical = results[
        results["Season"] != CURRENT_SEASON
    ]

    if len(historical) != 4180:
        raise ValueError(
            f"Expected 4180 historical results, got {len(historical)}"
        )

    if results.duplicated(
        ["Season", "HomeTeam_standard", "AwayTeam_standard"]
    ).any():
        raise ValueError("Duplicate Football-Data fixtures")

    print("\nFootball-Data matches:", len(results))
    print("Current season:", sum(results["Season"] == CURRENT_SEASON))

    # 2. Load Understat xG.
    xg_file = RAW_DIR / "understat_premier_league_xg.csv"

    if not xg_file.exists():
        raise FileNotFoundError(xg_file)

    xg = pd.read_csv(xg_file, dtype={"season": str})
    xg["season"] = xg["season"].str.zfill(4)

    xg["date"] = pd.to_datetime(
        xg["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)

    for column in [
        "home_goals", "away_goals", "home_xg", "away_xg"
    ]:
        xg[column] = pd.to_numeric(
            xg[column], errors="coerce"
        )

    if xg.duplicated(
        ["season", "home_team", "away_team"]
    ).any():
        raise ValueError("Duplicate Understat fixtures")

    print("Understat matches:", len(xg))

    # 3. Merge by season and teams.
    merged = results.merge(
        xg,
        left_on=[
            "Season", "HomeTeam_standard", "AwayTeam_standard"
        ],
        right_on=[
            "season", "home_team", "away_team"
        ],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = merged[merged["_merge"] != "both"]

    if not unmatched.empty:
        print("\nUNMATCHED FOOTBALL-DATA FIXTURES")
        print(
            unmatched[
                ["Season", "Date", "HomeTeam", "AwayTeam"]
            ].to_string(index=False)
        )
        raise ValueError(
            f"{len(unmatched)} Football-Data fixtures lack xG matches"
        )

    print("\nAll Football-Data fixtures matched to Understat.")

    # 4. Validate goals.
    score_mismatch = (
        merged["FTHG"].ne(merged["home_goals"])
        | merged["FTAG"].ne(merged["away_goals"])
    )

    if score_mismatch.any():
        print("\nSCORE MISMATCHES")
        print(
            merged.loc[
                score_mismatch,
                [
                    "Season", "HomeTeam", "AwayTeam",
                    "FTHG", "FTAG", "home_goals", "away_goals"
                ]
            ].to_string(index=False)
        )
        raise ValueError("Match scores disagree between providers")

    print("Score validation passed.")

    # 5. Validate dates.
    date_difference = (
        merged["Date"].dt.normalize()
        - merged["date"].dt.normalize()
    ).abs().dt.days

    if (date_difference > 1).any():
        print("\nDATE MISMATCHES")
        print(
            merged.loc[
                date_difference > 1,
                ["Season", "HomeTeam", "AwayTeam", "Date", "date"]
            ].to_string(index=False)
        )
        raise ValueError("Match dates differ by more than one day")

    print("Date validation passed.")

    # 6. Validate xG.
    xg_columns = ["home_xg", "away_xg"]

    if merged[xg_columns].isna().any().any():
        raise ValueError("Missing xG for matched fixtures")

    if (merged[xg_columns] < 0).any().any():
        raise ValueError("Negative xG values")

    print("xG validation passed.")

    # 7. Build the final dataset.
    final = merged[
        [
            "Date", "Season", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR", "home_xg", "away_xg"
        ]
    ].rename(
        columns={
            "home_xg": "HomeXG",
            "away_xg": "AwayXG"
        }
    )

    final = final.sort_values(
        ["Date", "Season", "HomeTeam"]
    ).reset_index(drop=True)

    if final.isna().any().any():
        raise ValueError("Missing values in final dataset")

    if final.duplicated(
        ["Season", "HomeTeam", "AwayTeam"]
    ).any():
        raise ValueError("Duplicate fixtures in final dataset")

    if len(final) != len(results):
        raise ValueError("Final dataset row count changed")

    # 8. Report Understat-only fixtures.
    matched_keys = set(
        zip(
            merged["Season"],
            merged["HomeTeam_standard"],
            merged["AwayTeam_standard"]
        )
    )

    understat_only = xg[
        ~pd.Series(
            list(zip(xg["season"], xg["home_team"], xg["away_team"])),
            index=xg.index
        ).isin(matched_keys)
    ]

    print("\nUnderstat-only fixtures:", len(understat_only))

    if not understat_only.empty:
        print(
            understat_only[
                ["date", "home_team", "away_team"]
            ].sort_values("date").to_string(index=False)
        )

    # 9. Save only after every validation passes.
    output = PROCESSED_DIR / "premier_league_with_xg.csv"
    temporary = output.with_suffix(".tmp")

    final.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d"
    )
    temporary.replace(output)

    print("\nMERGE SUCCESSFUL")
    print("Total matches:", len(final))
    print("Seasons:", final["Season"].nunique())
    print("Latest match:", final["Date"].max().date())
    print("\nMatches per season:")
    print(final.groupby("Season").size())
    print("\nSaved to:", output)


if __name__ == "__main__":
    main()
