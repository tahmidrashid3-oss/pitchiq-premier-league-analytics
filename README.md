# ⚽ PitchIQ — Premier League Analytics & Match Prediction

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://pitchiq-premier-league.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-blue)

**PitchIQ** is an interactive football analytics application that estimates the probabilities of a **home win, draw, or away win** in hypothetical Premier League matchups. It combines historical match results, expected goals (xG), rolling team form and Elo ratings in a reproducible machine-learning workflow.

**[Try the live dashboard](https://pitchiq-premier-league.streamlit.app/)**

## What the app does

- Select home and away teams and generate three-way match outcome probabilities.
- Compare teams using historical form and performance indicators.
- Use an Elo-enhanced model trained on 4,250 completed matches in the bundled dataset, through **19 October 2026**.
- Explore predictions based on the latest available historical data, **not live match odds or real-time results**.

## Model and features

The deployed **V2 model** uses multinomial match outcome classification with **Logistic Regression**, `StandardScaler`, and regularisation parameter `C = 0.01`.

It uses **31 pre-match features**:

- **30 rolling-form differences** (home minus away), derived from points, goals scored, goals conceded, xG for and xG against over 3-, 5- and 10-match windows, calculated both overall and for home/away venue context.
- **1 Elo difference feature** (`Diff_Elo`), representing the difference between the teams' ratings before the fixture.

Elo ratings are updated sequentially using completed historical matches. The V2 experiments also tested rest-day differences, but these were **not included** in the deployed model because they did not improve the historical validation results.

## Evaluation

Models were compared using chronological validation rather than random train/test splitting, to better reflect predictions on future matches.

### Historical validation (2021/22–2024/25)

| Metric | Baseline (30 features) | V2 + Elo (31 features) |
|---|---:|---:|
| Matches evaluated | 1,520 | 1,520 |
| Accuracy | 55.66% | **56.05%** |
| Multiclass log loss ↓ | 0.96687 | **0.95664** |
| Multiclass Brier score ↓ | 0.57397 | **0.56731** |

### Separate 2026/27 holdout (through 19 October 2026)

The model was trained on **4,180 earlier matches** and evaluated on **70 completed 2026/27 matches**.

| Metric | Baseline | V2 + Elo |
|---|---:|---:|
| Accuracy | 41.43% | **44.29%** |
| Multiclass log loss ↓ | 1.05275 | **1.03322** |
| Multiclass Brier score ↓ | 0.63396 | **0.62178** |

**Interpretation:** Elo produced modest improvements in accuracy and probability quality in both evaluations. The current-season holdout is small, so these results should not be treated as a guarantee of future performance. Draw prediction remains a limitation: draw recall was near zero in historical validation and zero in the 2026/27 holdout. The model is an analytical demonstration, not betting advice.

## Technical approach

1. Prepare historical Premier League results and xG data.
2. Engineer pre-match rolling-form features and sequential Elo ratings.
3. Compare candidate feature sets using chronological validation.
4. Train the selected 31-feature model and save the fitted pipeline.
5. Load the model in Streamlit, reconstruct the current Elo state and generate match probabilities for user-selected teams.

A consistency check compared all **4,250** stored pre-match Elo differences against recomputed values; the maximum absolute difference was approximately **5.68 × 10⁻¹⁴** (floating-point rounding).

## Tech stack

**Python · pandas · NumPy · scikit-learn · Streamlit · joblib · Git/GitHub**

## Repository structure

```text
app.py                              # Deployed Streamlit dashboard
scripts/build_advanced_features.py  # Rolling-form feature engineering
scripts/build_features_v2.py        # Elo and rest-day feature engineering
scripts/compare_features_v2.py      # Chronological feature-set comparison
scripts/train_model_v2.py           # Train the V2 model
models/pitchiq_model.joblib         # Deployed trained model
data/processed/                    # Prepared historical data
```

## Run locally

Clone the repository and install its dependencies:

```bash
git clone https://github.com/tahmidrashid3-oss/pitchiq-premier-league-analytics.git
cd pitchiq-premier-league-analytics
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repository must include the processed historical input data and the trained model at the paths referenced by `app.py`.

## Future improvements

- Improve draw probability estimation and assess calibration by outcome.
- Extend walk-forward evaluation to additional seasons.
- Add fixture scheduling and fresher data updates.
- Investigate alternative ratings and team-strength features.

---

**Project:** [GitHub repository](https://github.com/tahmidrashid3-oss/pitchiq-premier-league-analytics) · [Live Streamlit app](https://pitchiq-premier-league.streamlit.app/)
