from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'processed' / 'match_features_v2.csv'
OUTPUT_DIR = ROOT / 'reports' / 'tables'
VALIDATION_SEASONS = ['2122', '2223', '2324', '2425']
CLASSES = ['A', 'D', 'H']
METRICS = ['Points', 'GoalsFor', 'GoalsAgainst', 'xGFor', 'xGAgainst']
WINDOWS = [3, 5, 10]
BASELINE = [
    f'Diff_{prefix}_{metric}_Last{window}'
    for prefix in ['Overall', 'Venue']
    for window in WINDOWS
    for metric in METRICS
]
FEATURE_SETS = {
    'Baseline (30)': BASELINE,
    'Baseline + Elo (31)': BASELINE + ['Diff_Elo'],
    'Baseline + Elo + Rest (32)': BASELINE + ['Diff_Elo', 'Diff_RestDays'],
}


def make_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
    ])


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'Missing {DATA_FILE}; run build_features_v2.py first.')
    df = pd.read_csv(DATA_FILE, dtype={'Season': str}, parse_dates=['Date'])
    df['Season'] = df['Season'].str.zfill(4)
    required = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR'] + sorted(set(sum(FEATURE_SETS.values(), [])))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df[required].isna().any().any():
        raise ValueError('Missing values in required data.')
    if not df['FTR'].isin(CLASSES).all():
        raise ValueError('Unexpected match result label.')
    if df.duplicated(['Season', 'HomeTeam', 'AwayTeam']).any():
        raise ValueError('Duplicate fixtures detected.')
    return df.sort_values(['Date', 'Season', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)


def score(model, validation, features):
    actual = validation['FTR'].to_numpy()
    predicted = model.predict(validation[features])
    raw_prob = model.predict_proba(validation[features])
    probs = pd.DataFrame(raw_prob, columns=model.classes_)[CLASSES].to_numpy()
    one_hot = (actual[:, None] == np.array(CLASSES)[None, :]).astype(float)
    draw_mask = actual == 'D'
    return {
        'Matches': len(validation),
        'Accuracy': accuracy_score(actual, predicted),
        'LogLoss': log_loss(actual, probs, labels=CLASSES),
        'BrierScore': np.mean(np.sum((probs - one_hot) ** 2, axis=1)),
        'DrawPredictions': int(np.sum(predicted == 'D')),
        'DrawRecall': (np.mean(predicted[draw_mask] == 'D') if draw_mask.any() else np.nan),
    }


def main():
    print('=' * 70)
    print('PITCHIQ - V2 FEATURE COMPARISON (CHRONOLOGICAL VALIDATION)')
    print('=' * 70)
    df = load_data()
    print(f'Matches: {len(df):,}')
    rows = []
    for season in VALIDATION_SEASONS:
        train = df[df['Season'] < season]
        validation = df[df['Season'] == season]
        if train.empty or validation.empty:
            raise ValueError(f'Missing training or validation matches for {season}')
        if set(train['FTR']) != set(CLASSES):
            raise ValueError(f'Training data missing outcome class for {season}')
        print(f'\nSeason {season}: train={len(train):,}, validate={len(validation):,}')
        for name, features in FEATURE_SETS.items():
            model = make_model()
            model.fit(train[features], train['FTR'])
            metrics = score(model, validation, features)
            rows.append({'Season': season, 'FeatureSet': name, 'Features': len(features), **metrics})
            print(f"  {name}: accuracy={metrics['Accuracy']:.2%}, "
                  f"log loss={metrics['LogLoss']:.5f}, "
                  f"Brier={metrics['BrierScore']:.5f}, "
                  f"draw recall={metrics['DrawRecall']:.2%}")

    results = pd.DataFrame(rows)
    summary = (results.groupby(['FeatureSet', 'Features'], as_index=False)
               .agg(MeanAccuracy=('Accuracy', 'mean'),
                    MeanLogLoss=('LogLoss', 'mean'),
                    MeanBrierScore=('BrierScore', 'mean'),
                    MeanDrawRecall=('DrawRecall', 'mean'),
                    TotalDrawPredictions=('DrawPredictions', 'sum'),
                    TotalMatches=('Matches', 'sum'))
               .sort_values('MeanLogLoss').reset_index(drop=True))
    print('\n' + '=' * 70)
    print('V2 FEATURE COMPARISON SUMMARY (LOWER LOG LOSS IS BETTER)')
    print('=' * 70)
    print(summary.round(5).to_string(index=False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / 'feature_v2_comparison_results.csv'
    summary_path = OUTPUT_DIR / 'feature_v2_comparison_summary.csv'
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f'\nSaved:\n{results_path}\n{summary_path}')
    print('Production model and app have not been changed.')


if __name__ == '__main__':
    main()
