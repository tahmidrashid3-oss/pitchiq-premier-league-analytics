from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "data/processed/premier_league_with_xg.csv"
PREVIEW = ROOT / "backups/preview/premier_league_with_xg_preview.csv"

def main():
    if not PRODUCTION.exists() or not PREVIEW.exists():
        raise FileNotFoundError("Production or preview dataset missing")

    old = pd.read_csv(PRODUCTION, dtype={"Season": str})
    new = pd.read_csv(PREVIEW, dtype={"Season": str})

    if list(old.columns) != list(new.columns):
        raise ValueError("Dataset columns have changed")

    keys = ["Date", "HomeTeam", "AwayTeam"]

    for name, df in [("production", old), ("preview", new)]:
        if df[keys].isna().any().any():
            raise ValueError(f"Missing match identifiers in {name}")
        if df.duplicated(keys).any():
            raise ValueError(f"Duplicate matches in {name}")

    if len(new) < len(old):
        raise ValueError("Preview contains fewer matches than production")

    existing = new.merge(
        old[keys], on=keys, how="inner"
    )

    if len(existing) != len(old):
        raise ValueError("Preview is missing existing matches")

    old_indexed = old.set_index(keys).sort_index()
    new_existing = new.set_index(keys).loc[old_indexed.index]

    pd.testing.assert_frame_equal(
        old_indexed,
        new_existing,
        check_dtype=False,
        check_exact=True,
    )

    additions = len(new) - len(old)

    if additions == 0:
        print("No new matches. Nothing to publish.")
        return

    if new.isna().sum().sum() > old.isna().sum().sum():
        raise ValueError("Preview introduces missing values")

    if not new["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Invalid match results")

    temporary = PRODUCTION.with_suffix(".publishing.tmp")
    new.to_csv(temporary, index=False)
    temporary.replace(PRODUCTION)

    print(f"Published {additions} new matches.")
    print(f"Production dataset now contains {len(new)} matches.")

if __name__ == "__main__":
    main()
