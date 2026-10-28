
from pathlib import Path

import numpy as np
import pandas as pd

from build_advanced_features import (
    create_team_history,
    add_rolling_features,
    build_match_features,
)
from build_features_v2 import (
    compute_pre_match_features,
    KEY,
    NEW_FEATURES,
)
from train_model_v2 import FEATURES

ROOT = Path(__file__).resolve().parents[1]

PREVIEW = (
    ROOT / "backups" / "preview"
    / "premier_league_with_xg_preview.csv"
)

PRODUCTION = (
    ROOT / "data" / "processed"
    / "premier_league_with_xg.csv"
)


def main():
    print("=" * 60)
    print("PITCHIQ - IN-MEMORY FEATURE COMPATIBILITY TEST")
    print("=" * 60)

    original = pd.read_csv(
        PRODUCTION,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    matches = pd.read_csv(
        PREVIEW,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    for df in (original, matches):
        df["Season"] = df["Season"].str.zfill(4)
        df["Date"] = df["Date"].dt.normalize()

        if df[KEY].isna().any().any():
            raise ValueError("Missing fixture identifiers.")

        if df.duplicated(KEY).any():
            raise ValueError("Duplicate fixtures detected.")

    if len(matches) < len(original):
        raise ValueError("Preview has fewer matches than production.")

    matches = matches.sort_values(KEY).reset_index(drop=True)
    matches["MatchID"] = np.arange(len(matches))

    print(f"\nProduction matches: {len(original)}")
    print(f"Preview matches: {len(matches)}")

    print("\nBuilding advanced rolling features...")
    history = create_team_history(matches)
    history = add_rolling_features(history)
    advanced = build_match_features(matches, history)

    print("Building Elo and rest-day features...")
    extra = compute_pre_match_features(matches)

    combined = advanced.merge(
        extra,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    if len(combined) != len(matches):
        raise ValueError("Feature generation changed match count.")

    if combined.duplicated(KEY).any():
        raise ValueError("Duplicate feature rows.")

    if combined[NEW_FEATURES].isna().any().any():
        raise ValueError("Missing Elo or rest-day values.")

    if combined[FEATURES].isna().any().any():
        raise ValueError("Missing model features.")

    if not np.isfinite(
        combined[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite model features.")

    if len(FEATURES) != 31:
        raise ValueError("Expected 31 model features.")

    current = combined[
        combined["Season"] == "2627"
    ]

    print("\nVALIDATION SUCCESSFUL")
    print(f"Feature rows: {len(combined)}")
    print(f"Model input features: {len(FEATURES)}")
    print(f"2026/27 matches: {len(current)}")
    print(f"Latest match: {combined['Date'].max().date()}")
    print("Duplicate feature rows: 0")
    print("Missing model features: 0")

    print("\nNo CSV or model files were written.")


if __name__ == "__main__":
    main()
