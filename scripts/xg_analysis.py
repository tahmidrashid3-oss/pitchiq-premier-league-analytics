
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "premier_league_with_xg.csv"
)

FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TEAM = "Man United"
LATEST_SEASON = "2526"

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
})


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run scripts/merge_xg_data.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"Season": str},
        parse_dates=["Date"]
    )

    required = [
        "Date", "Season", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "HomeXG", "AwayXG"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if len(df) != 4180:
        raise ValueError(
            f"Expected 4180 matches, found {len(df)}."
        )

    df["Season"] = df["Season"].str.zfill(4)

    for col in ["FTHG", "FTAG", "HomeXG", "AwayXG"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    print("Loaded matches:", len(df))
    print("Seasons:", df["Season"].nunique())

    return df


# ============================================================
# 2. CREATE TEAM-LEVEL MATCH DATA
# ============================================================

def create_team_match_data(df):
    # One row for the home team's perspective.
    home = pd.DataFrame({
        "Date": df["Date"],
        "Season": df["Season"],
        "Team": df["HomeTeam"],
        "Opponent": df["AwayTeam"],
        "Venue": "Home",
        "GoalsFor": df["FTHG"],
        "GoalsAgainst": df["FTAG"],
        "xGFor": df["HomeXG"],
        "xGAgainst": df["AwayXG"],
    })

    # One row for the away team's perspective.
    away = pd.DataFrame({
        "Date": df["Date"],
        "Season": df["Season"],
        "Team": df["AwayTeam"],
        "Opponent": df["HomeTeam"],
        "Venue": "Away",
        "GoalsFor": df["FTAG"],
        "GoalsAgainst": df["FTHG"],
        "xGFor": df["AwayXG"],
        "xGAgainst": df["HomeXG"],
    })

    matches = pd.concat(
        [home, away],
        ignore_index=True
    )

    matches["GoalDifference"] = (
        matches["GoalsFor"] - matches["GoalsAgainst"]
    )

    matches["xGDifference"] = (
        matches["xGFor"] - matches["xGAgainst"]
    )

    matches["Points"] = np.select(
        [
            matches["GoalsFor"] > matches["GoalsAgainst"],
            matches["GoalsFor"] == matches["GoalsAgainst"]
        ],
        [3, 1],
        default=0
    )

    matches = matches.sort_values(
        ["Season", "Date", "Team"]
    ).reset_index(drop=True)

    return matches


# ============================================================
# 3. AGGREGATE TEAM STATISTICS BY SEASON
# ============================================================

def create_season_summary(matches):
    summary = (
        matches
        .groupby(["Season", "Team"], as_index=False)
        .agg(
            Matches=("Points", "size"),
            Points=("Points", "sum"),
            GoalsFor=("GoalsFor", "sum"),
            GoalsAgainst=("GoalsAgainst", "sum"),
            xGFor=("xGFor", "sum"),
            xGAgainst=("xGAgainst", "sum"),
        )
    )

    summary["GoalDifference"] = (
        summary["GoalsFor"] - summary["GoalsAgainst"]
    )

    summary["xGDifference"] = (
        summary["xGFor"] - summary["xGAgainst"]
    )

    summary["AttackingOverperformance"] = (
        summary["GoalsFor"] - summary["xGFor"]
    )

    summary["DefensiveOverperformance"] = (
        summary["xGAgainst"] - summary["GoalsAgainst"]
    )

    summary["GDOverperformance"] = (
        summary["GoalDifference"] - summary["xGDifference"]
    )

    summary = summary.sort_values(
        ["Season", "Points", "GoalDifference", "GoalsFor"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    summary["LeaguePosition"] = (
        summary.groupby("Season").cumcount() + 1
    )

    summary["xGPosition"] = (
        summary.groupby("Season")["xGDifference"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return summary


# ============================================================
# 4. MANCHESTER UNITED: GOALS VS XG
# ============================================================

def plot_united_attacking(summary):
    united = (
        summary[summary["Team"] == TARGET_TEAM]
        .sort_values("Season")
    )

    if united.empty:
        raise ValueError("Manchester United data not found.")

    x = np.arange(len(united))
    labels = [
        f"{s[:2]}/{s[2:]}" for s in united["Season"]
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        x, united["GoalsFor"],
        marker="o", linewidth=2.5,
        label="Actual goals"
    )

    ax.plot(
        x, united["xGFor"],
        marker="s", linewidth=2.5,
        linestyle="--",
        label="Expected goals (xG)"
    )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Premier League season")
    ax.set_ylabel("Goals")
    ax.set_title(
        "Manchester United: Actual Goals vs Expected Goals",
        fontsize=16, fontweight="bold", pad=16
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    output = FIGURE_DIR / "man_united_goals_vs_xg.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output.name)


# ============================================================
# 5. MANCHESTER UNITED: GOALS CONCEDED VS XGA
# ============================================================

def plot_united_defending(summary):
    united = (
        summary[summary["Team"] == TARGET_TEAM]
        .sort_values("Season")
    )

    x = np.arange(len(united))
    labels = [
        f"{s[:2]}/{s[2:]}" for s in united["Season"]
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        x, united["GoalsAgainst"],
        marker="o", linewidth=2.5,
        label="Actual goals conceded"
    )

    ax.plot(
        x, united["xGAgainst"],
        marker="s", linewidth=2.5,
        linestyle="--",
        label="Expected goals against (xGA)"
    )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Premier League season")
    ax.set_ylabel("Goals conceded")
    ax.set_title(
        "Manchester United: Goals Conceded vs xGA",
        fontsize=16, fontweight="bold", pad=16
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    output = FIGURE_DIR / "man_united_goals_conceded_vs_xga.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output.name)


# ============================================================
# 6. LATEST SEASON: ACTUAL VS EXPECTED GOAL DIFFERENCE
# ============================================================

def plot_goal_difference(summary):
    season = (
        summary[summary["Season"] == LATEST_SEASON]
        .sort_values("xGDifference")
    )

    if len(season) != 20:
        raise ValueError(
            f"Expected 20 teams in {LATEST_SEASON}, "
            f"found {len(season)}."
        )

    y = np.arange(len(season))
    height = 0.38

    fig, ax = plt.subplots(figsize=(13, 10))

    ax.barh(
        y - height / 2,
        season["GoalDifference"],
        height=height,
        label="Actual goal difference"
    )

    ax.barh(
        y + height / 2,
        season["xGDifference"],
        height=height,
        label="Expected goal difference (xGD)"
    )

    ax.set_yticks(y, season["Team"])
    ax.axvline(0, color="gray", linewidth=1)

    ax.set_xlabel("Goal difference")
    ax.set_title(
        "2025/26 Premier League: Goal Difference vs xGD",
        fontsize=16, fontweight="bold", pad=16
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    output = FIGURE_DIR / "premier_league_gd_vs_xgd_2526.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output.name)


# ============================================================
# 7. LATEST SEASON: XG PERFORMANCE RANKING
# ============================================================

def plot_xg_performance(summary):
    season = (
        summary[summary["Season"] == LATEST_SEASON]
        .sort_values("GDOverperformance")
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = np.where(
        season["GDOverperformance"] >= 0,
        "seagreen",
        "indianred"
    )

    ax.barh(
        season["Team"],
        season["GDOverperformance"],
        color=colors
    )

    ax.axvline(0, color="gray", linewidth=1)

    ax.set_xlabel(
        "Actual goal difference minus expected goal difference"
    )

    ax.set_title(
        "2025/26 Premier League: Performance Relative to xG",
        fontsize=16, fontweight="bold", pad=16
    )

    fig.tight_layout()

    output = (
        FIGURE_DIR / "premier_league_xg_overperformance_2526.png"
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output.name)


# ============================================================
# 8. SAVE SUMMARY TABLES
# ============================================================

def save_tables(summary):
    all_seasons_file = (
        TABLE_DIR / "team_season_xg_summary.csv"
    )

    summary.round(3).to_csv(
        all_seasons_file,
        index=False
    )

    latest = (
        summary[summary["Season"] == LATEST_SEASON]
        .sort_values(
            "xGDifference",
            ascending=False
        )
        .copy()
    )

    latest.insert(
        0,
        "xGTableRank",
        np.arange(1, len(latest) + 1)
    )

    latest_file = (
        TABLE_DIR / "premier_league_xg_table_2526.csv"
    )

    latest.round(3).to_csv(
        latest_file,
        index=False
    )

    print("Saved:", all_seasons_file.name)
    print("Saved:", latest_file.name)

    return latest


# ============================================================
# 9. PRINT KEY RESULTS
# ============================================================

def print_results(summary, latest):
    print("\n" + "=" * 65)
    print("MANCHESTER UNITED: SEASON-BY-SEASON XG ANALYSIS")
    print("=" * 65)

    united = (
        summary[summary["Team"] == TARGET_TEAM]
        .sort_values("Season")
    )

    columns = [
        "Season",
        "LeaguePosition",
        "Points",
        "GoalsFor",
        "xGFor",
        "GoalsAgainst",
        "xGAgainst",
        "GoalDifference",
        "xGDifference"
    ]

    print(
        united[columns]
        .round(2)
        .to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("2025/26 PREMIER LEAGUE XG TABLE")
    print("=" * 65)

    columns = [
        "xGTableRank",
        "Team",
        "LeaguePosition",
        "Points",
        "GoalDifference",
        "xGDifference",
        "GDOverperformance"
    ]

    print(
        latest[columns]
        .round(2)
        .to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("2025/26: TOP FIVE GD OVERPERFORMERS")
    print("=" * 65)

    top = latest.nlargest(
        5, "GDOverperformance"
    )

    print(
        top[["Team", "GDOverperformance"]]
        .round(2)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("PITCHIQ — MILESTONE 4: XG ANALYSIS")
    print("-" * 65)

    df = load_data()

    matches = create_team_match_data(df)
    summary = create_season_summary(matches)

    if len(matches) != 2 * len(df):
        raise ValueError("Team-match data has incorrect row count.")

    if not (summary["Matches"] == 38).all():
        raise ValueError("Some teams do not have 38 matches.")

    plot_united_attacking(summary)
    plot_united_defending(summary)
    plot_goal_difference(summary)
    plot_xg_performance(summary)

    latest = save_tables(summary)
    print_results(summary, latest)

    print("\nMILESTONE 4 COMPLETE")
    print("Figures saved to:", FIGURE_DIR)
    print("Tables saved to:", TABLE_DIR)


if __name__ == "__main__":
    main()
