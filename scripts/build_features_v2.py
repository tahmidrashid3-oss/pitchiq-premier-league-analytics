"""Build PitchIQ V2 features without modifying the production dataset.

Adds leakage-safe pre-match Elo and Premier-League-only rest-day features.
Same-day fixtures are processed as a batch before any ratings are updated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = ROOT / 'data' / 'processed' / 'premier_league_with_xg.csv'
BASE_PATH = ROOT / 'data' / 'processed' / 'match_features_advanced.csv'
OUTPUT_PATH = ROOT / 'data' / 'processed' / 'match_features_v2.csv'

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0
DEFAULT_REST_DAYS = 7.0
MAX_REST_DAYS = 21.0

KEY = ['Date', 'Season', 'HomeTeam', 'AwayTeam']
NEW_FEATURES = ['Home_Elo', 'Away_Elo', 'Diff_Elo', 'Diff_RestDays']


def load_data():
    if not MATCHES_PATH.exists() or not BASE_PATH.exists():
        raise FileNotFoundError('Run the existing advanced feature pipeline first; required CSV files are missing.')
    matches = pd.read_csv(MATCHES_PATH, dtype={'Season': str}, parse_dates=['Date'])
    base = pd.read_csv(BASE_PATH, dtype={'Season': str}, parse_dates=['Date'])
    for df in (matches, base):
        df['Season'] = df['Season'].str.zfill(4)
        df['Date'] = df['Date'].dt.normalize()
        if df[KEY].isna().any().any() or df.duplicated(KEY).any():
            raise ValueError('Missing or duplicated match identifiers detected.')
    if len(matches) != len(base) or set(map(tuple, matches[KEY].itertuples(index=False, name=None))) != set(map(tuple, base[KEY].itertuples(index=False, name=None))):
        raise ValueError('Advanced features and raw matches contain different fixtures.')
    required = ['FTHG', 'FTAG', 'FTR']
    if matches[required].isna().any().any():
        raise ValueError('Missing full-time scores or results.')
    return matches.sort_values(KEY).reset_index(drop=True), base


def compute_pre_match_features(matches, return_state=False):
    elo = {}
    last_match_date = {}
    rows = []

    for date, fixtures in matches.groupby('Date', sort=True):
        updates = []
        teams_today = set()
        for match in fixtures.itertuples(index=False):
            home, away = match.HomeTeam, match.AwayTeam
            if home == away:
                raise ValueError('A team cannot play itself.')
            if home in teams_today or away in teams_today:
                raise ValueError(f'Team has multiple fixtures on {date.date()}; same-day rest calculation is ambiguous.')
            teams_today.update((home, away))
            h_rating = elo.get(home, INITIAL_ELO)
            a_rating = elo.get(away, INITIAL_ELO)

            def rest_days(team):
                previous = last_match_date.get(team)
                if previous is None:
                    return DEFAULT_REST_DAYS
                elapsed = (date - previous).days
                if elapsed < 0:
                    raise ValueError('Match dates are not chronological.')
                return float(min(elapsed, MAX_REST_DAYS))

            h_rest, a_rest = rest_days(home), rest_days(away)
            rows.append({
                'Date': date, 'Season': match.Season,
                'HomeTeam': home, 'AwayTeam': away,
                'Home_Elo': h_rating, 'Away_Elo': a_rating,
                'Diff_Elo': h_rating - a_rating,
                'Diff_RestDays': h_rest - a_rest,
            })
            expected_home = 1.0 / (1.0 + 10.0 ** ((a_rating - h_rating - HOME_ADVANTAGE) / 400.0))
            actual_home = 1.0 if match.FTHG > match.FTAG else (0.5 if match.FTHG == match.FTAG else 0.0)
            delta = K_FACTOR * (actual_home - expected_home)
            updates.append((home, away, h_rating + delta, a_rating - delta))

        # Apply updates only after every fixture on this date was evaluated.
        for home, away, new_home, new_away in updates:
            elo[home] = new_home
            elo[away] = new_away
            last_match_date[home] = date
            last_match_date[away] = date

    result = pd.DataFrame(rows)
    if return_state:
        return result, dict(elo)
    return result


def main():
    matches, base = load_data()
    extra = compute_pre_match_features(matches)
    output = base.merge(extra, on=KEY, how='left', validate='one_to_one')
    if len(output) != len(base) or output[NEW_FEATURES].isna().any().any():
        raise ValueError('V2 merge produced missing values or changed row count.')
    if not np.isfinite(output[NEW_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('V2 features contain non-finite values.')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f'Input matches: {len(matches):,}')
    print(f'Output matches: {len(output):,}')
    print(f'Added features: {", ".join(NEW_FEATURES)}')
    print(f'Saved: {OUTPUT_PATH}')
    print('Existing advanced dataset and production model were not modified.')


if __name__ == '__main__':
    main()

