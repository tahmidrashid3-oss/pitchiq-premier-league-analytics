
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

files = sorted(RAW_DIR.glob("premier_league_*.csv"))

if not files:
    raise FileNotFoundError("No Premier League CSV files found.")

df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

df["Date"] = pd.to_datetime(df["Date"], errors="raise")

print("PITCHIQ DATA VALIDATION")
print("-" * 40)

print(f"Total matches: {len(df)}")
print(f"Total seasons: {df['Season'].nunique()}")
print(f"Missing values: {df.isnull().sum().sum()}")

print("\nMatches per season:")
print(df.groupby("Season").size())

print("\nMatch outcome distribution:")
print(df["FTR"].value_counts())

print("\nManchester United matches:")
united = df[
    (df["HomeTeam"] == "Man United") |
    (df["AwayTeam"] == "Man United")
]
print(len(united))

# Validate key football rules.
assert df["FTR"].isin(["H", "D", "A"]).all()
assert (df["FTHG"] >= 0).all()
assert (df["FTAG"] >= 0).all()

expected_outcome = pd.Series(
    ["H" if h > a else "A" if h < a else "D"
     for h, a in zip(df["FTHG"], df["FTAG"])],
    index=df.index
)

assert (df["FTR"] == expected_outcome).all(), (
    "Some recorded outcomes disagree with final scores."
)

assert df.groupby("Season").size().eq(380).all(), (
    "At least one season does not contain 380 matches."
)

print("\nAll validation checks passed!")
