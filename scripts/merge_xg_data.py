
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD FOOTBALL-DATA RESULTS
# ============================================================

files = sorted(RAW_DIR.glob("premier_league_*.csv"))

if len(files) != 11:
    raise ValueError(
        f"Expected 11 season files, found {len(files)}."
    )

results = pd.concat(
    [pd.read_csv(file, dtype={"Season": str}) for file in files],
    ignore_index=True
)

results["Season"] = results["Season"].str.zfill(4)

results["Date"] = pd.to_datetime(
    results["Date"],
    format="%Y-%m-%d",
    errors="raise"
)

print("Football-Data matches:", len(results))


# ============================================================
# 2. LOAD UNDERSTAT XG DATA
# ============================================================

xg_file = RAW_DIR / "understat_premier_league_xg.csv"

if not xg_file.exists():
    raise FileNotFoundError(
        "Understat data not found. "
        "Run test_understat.py first."
    )

xg = pd.read_csv(
    xg_file,
    dtype={"season": str}
)

xg["season"] = xg["season"].str.zfill(4)

# Parse as UTC, then remove timezone information
# to allow consistent calendar-date comparisons.
xg["date"] = pd.to_datetime(
    xg["date"],
    utc=True,
    errors="raise"
).dt.tz_convert(None)

print("Understat matches:", len(xg))


# ============================================================
# 3. STANDARDISE TEAM NAMES
# ============================================================

team_names = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "West Brom": "West Bromwich Albion",
    "Tottenham": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham",
    "Leicester": "Leicester",
    "Brighton": "Brighton",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
}

results["HomeTeam_standard"] = (
    results["HomeTeam"].replace(team_names)
)

results["AwayTeam_standard"] = (
    results["AwayTeam"].replace(team_names)
)


# ============================================================
# 4. MERGE DATASETS
# ============================================================

merged = results.merge(
    xg,
    left_on=[
        "Season",
        "HomeTeam_standard",
        "AwayTeam_standard"
    ],
    right_on=[
        "season",
        "home_team",
        "away_team"
    ],
    how="left",
    validate="one_to_one",
    indicator=True
)

print("\nMERGE RESULTS")
print(merged["_merge"].value_counts())

unmatched = merged[merged["_merge"] != "both"]

if not unmatched.empty:
    print("\nUNMATCHED FIXTURES")
    print(
        unmatched[
            ["Season", "HomeTeam", "AwayTeam"]
        ].to_string(index=False)
    )

    raise ValueError(
        f"{len(unmatched)} fixtures could not be matched."
    )

print("Team matching passed.")


# ============================================================
# 5. VALIDATE MATCH SCORES
# ============================================================

score_mismatches = (
    (merged["FTHG"] != merged["home_goals"])
    |
    (merged["FTAG"] != merged["away_goals"])
)

if score_mismatches.any():
    print("\nSCORE MISMATCHES")
    print(
        merged.loc[
            score_mismatches,
            [
                "Season",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "home_goals",
                "away_goals"
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        f"{score_mismatches.sum()} score mismatches found."
    )

print("Score validation passed.")


# ============================================================
# 6. VALIDATE MATCH DATES
# ============================================================

date_difference = (
    merged["Date"].dt.normalize()
    - merged["date"].dt.normalize()
).abs().dt.days

print("\nDATE DIFFERENCES")
print(date_difference.value_counts().sort_index())

date_mismatches = date_difference > 1

if date_mismatches.any():
    print("\nDATE MISMATCHES")
    print(
        merged.loc[
            date_mismatches,
            [
                "Season",
                "HomeTeam",
                "AwayTeam",
                "Date",
                "date"
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        f"{date_mismatches.sum()} fixtures have "
        "date differences greater than one day."
    )

print("Date validation passed.")


# ============================================================
# 7. VALIDATE XG VALUES
# ============================================================

xg_columns = ["home_xg", "away_xg"]

for column in xg_columns:
    merged[column] = pd.to_numeric(
        merged[column],
        errors="coerce"
    )

if merged[xg_columns].isna().any().any():
    raise ValueError("Missing xG values detected.")

if (merged[xg_columns] < 0).any().any():
    raise ValueError("Negative xG values detected.")

print("xG validation passed.")


# ============================================================
# 8. CREATE FINAL DATASET
# ============================================================

final = merged[
    [
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "home_xg",
        "away_xg"
    ]
].copy()

final = final.rename(
    columns={
        "home_xg": "HomeXG",
        "away_xg": "AwayXG"
    }
)

final = final.sort_values(
    ["Date", "Season", "HomeTeam"]
).reset_index(drop=True)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

if len(final) != 4180:
    raise ValueError(
        f"Expected 4180 matches, found {len(final)}."
    )

if final.duplicated(
    subset=["Season", "HomeTeam", "AwayTeam"]
).any():
    raise ValueError("Duplicate fixtures detected.")

if final.isna().any().any():
    raise ValueError("Missing values detected.")

print("\nFINAL VALIDATION PASSED")
print("Total matches:", len(final))
print("Seasons:", final["Season"].nunique())

print("\nMissing values:")
print(final.isna().sum())


# ============================================================
# 10. SAVE DATASET
# ============================================================

output_file = (
    PROCESSED_DIR / "premier_league_with_xg.csv"
)

final.to_csv(
    output_file,
    index=False,
    date_format="%Y-%m-%d"
)

print("\nMERGE SUCCESSFUL")
print("Saved to:", output_file)

print("\nSample data:")
print(final.head().to_string(index=False))


if __name__ == "__main__":
    pass
