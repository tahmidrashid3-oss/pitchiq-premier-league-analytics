
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project directories
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    """Load all downloaded Premier League seasons."""

    files = sorted(RAW_DIR.glob("premier_league_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No data found. Run download_data.py first."
        )

    df = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df.sort_values("Date").reset_index(drop=True)


def analyse_home_advantage(df):
    """Calculate and visualise match outcomes."""

    labels = {
        "H": "Home win",
        "D": "Draw",
        "A": "Away win"
    }

    counts = (
        df["FTR"]
        .value_counts()
        .reindex(["H", "D", "A"], fill_value=0)
    )

    percentages = counts / len(df) * 100

    print("\nMATCH OUTCOMES")
    print(percentages.round(2))

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=[labels[result] for result in counts.index],
        y=percentages.values,
        ax=ax
    )

    ax.set_title("Premier League Match Outcomes (2015–2026)")
    ax.set_xlabel("Match outcome")
    ax.set_ylabel("Percentage of matches")

    for i, value in enumerate(percentages):
        ax.text(
            i, value + 0.5, f"{value:.1f}%",
            ha="center"
        )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "match_outcomes.png",
        dpi=300
    )
    plt.close(fig)


def analyse_goals(df):
    """Analyse goals scored by home and away teams."""

    home_average = df["FTHG"].mean()
    away_average = df["FTAG"].mean()

    print("\nAVERAGE GOALS PER MATCH")
    print(f"Home teams: {home_average:.2f}")
    print(f"Away teams: {away_average:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=["Home teams", "Away teams"],
        y=[home_average, away_average],
        ax=ax
    )

    ax.set_title("Average Goals per Match")
    ax.set_ylabel("Average goals")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "average_goals.png",
        dpi=300
    )
    plt.close(fig)


def analyse_manchester_united(df):
    """Analyse Manchester United's results by season."""

    united = df[
        (df["HomeTeam"] == "Man United")
        | (df["AwayTeam"] == "Man United")
    ].copy()

    united["Result"] = united.apply(
        lambda row: (
            "Win" if (
                (row["HomeTeam"] == "Man United" and row["FTR"] == "H")
                or
                (row["AwayTeam"] == "Man United" and row["FTR"] == "A")
            )
            else "Draw" if row["FTR"] == "D"
            else "Loss"
        ),
        axis=1
    )

    results = (
        united.groupby(["Season", "Result"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Win", "Draw", "Loss"], fill_value=0)
    )

    print("\nMANCHESTER UNITED RESULTS BY SEASON")
    print(results)

    ax = results.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6)
    )

    ax.set_title("Manchester United Results by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of matches")
    ax.legend(title="Result")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "man_united_results.png",
        dpi=300
    )
    plt.close()


def main():
    df = load_data()

    print(f"Loaded {len(df)} Premier League matches.")

    analyse_home_advantage(df)
    analyse_goals(df)
    analyse_manchester_united(df)

    print("\nAnalysis complete!")
    print(f"Charts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
