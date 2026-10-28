"""Train a separate 31-feature PitchIQ V2 artifact; never overwrite production."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/match_features_v2.csv'
OUTPUT = ROOT / 'models/pitchiq_model_v2.joblib'
METRICS = ['Points', 'GoalsFor', 'GoalsAgainst', 'xGFor', 'xGAgainst']
WINDOWS = [3, 5, 10]
FEATURES = ([f'Diff_Overall_{m}_Last{w}' for w in WINDOWS for m in METRICS]
            + [f'Diff_Venue_{m}_Last{w}' for w in WINDOWS for m in METRICS]
            + ['Diff_Elo'])

def main():
    df = pd.read_csv(DATA, dtype={'Season': str}, parse_dates=['Date'])
    if df[FEATURES + ['FTR']].isna().any().any():
        raise ValueError('Missing model inputs')
    if not df['FTR'].isin(['A', 'D', 'H']).all():
        raise ValueError('Invalid results')
    model = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(C=0.01, max_iter=3000, random_state=42))])
    model.fit(df[FEATURES], df['FTR'])
    artifact = {'model': model, 'features': FEATURES, 'classes': list(model.classes_),
                'regularisation_C': 0.01, 'training_matches': len(df),
                'last_training_date': str(df['Date'].max().date()),
                'version': 'v2-elo', 'historical_validation_log_loss': 0.95664}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix('.tmp')
    joblib.dump(artifact, tmp)
    tmp.replace(OUTPUT)
    print(f'Saved V2 model: {OUTPUT}')
    print(f'Matches: {len(df):,}; features: {len(FEATURES)}')
    print('Production model unchanged.')

if __name__ == '__main__':
    main()
