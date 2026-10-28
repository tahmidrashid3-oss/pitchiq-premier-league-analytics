from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scripts.build_features_v2 import compute_pre_match_features


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "models" / "pitchiq_model.joblib"
DATA_FILE = (
    ROOT / "data" / "processed"
    / "premier_league_with_xg.csv"
)

WINDOWS = [3, 5, 10]

METRICS = [
    "Points",
    "GoalsFor",
    "GoalsAgainst",
    "xGFor",
    "xGAgainst",
]

DEFAULTS = {
    "Points": 1.0,
    "GoalsFor": 1.3,
    "GoalsAgainst": 1.3,
    "xGFor": 1.3,
    "xGAgainst": 1.3,
}

TEAM_NAMES = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham Hotspur",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "Nott'm Forest": "Nottingham Forest",
}

OUTCOME_LABELS = {
    "H": "Home win",
    "D": "Draw",
    "A": "Away win",
}

st.set_page_config(
    page_title="PitchIQ | Premier League Analytics",
    page_icon="\u26bd",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0B1020;
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background: #111A2E;
        border-right: 1px solid #26354D;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    .hero {
        background: linear-gradient(
            125deg, #182C50, #10192C
        );
        border: 1px solid #314A70;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #38BDF8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #AFC1D9;
        font-size: 1rem;
    }

    .match-card {
        background: #131F35;
        border: 1px solid #2A3C59;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
    }

    .team-name {
        color: #FFFFFF;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .team-role {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 5px;
    }

    .versus {
        color: #38BDF8;
        font-size: 1.8rem;
        font-weight: 800;
        padding-top: 20px;
    }

    .prediction-card {
        background: #131F35;
        border: 1px solid #2A3C59;
        border-radius: 14px;
        padding: 24px 15px;
        text-align: center;
    }

    .prediction-label {
        color: #AFC1D9;
        font-size: 0.9rem;
    }

    .prediction-value {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-description {
        color: #94A3B8;
        margin-bottom: 20px;
    }

    .form-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0;
    }

    .form-win, .form-draw, .form-loss {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 8px;
        color: white;
        font-weight: 800;
    }

    .form-win {
        background: #16A34A;
    }

    .form-draw {
        background: #D97706;
    }

    .form-loss {
        background: #DC2626;
    }

    .stat-card {
        background: #131F35;
        border: 1px solid #2A3C59;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .stat-label {
        color: #94A3B8;
        font-size: 0.85rem;
    }

    .stat-value {
        color: #F8FAFC;
        font-size: 1.6rem;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background: #2563EB;
        border: none;
        font-weight: 700;
        border-radius: 10px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1D4ED8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model missing. Run "
            "python scripts/train_model_v2.py"
        )

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_matches():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"Season": str},
        parse_dates=["Date"],
    )

    required = [
        "Date", "Season", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HomeXG", "AwayXG",
    ]

    if df[required].isna().any().any():
        raise ValueError("Missing match values detected.")

    return df.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)


# ============================================================
# TEAM HISTORY
# ============================================================

def create_team_history(matches):
    home = pd.DataFrame({
        "Date": matches["Date"],
        "Team": matches["HomeTeam"],
        "Venue": "Home",
        "Opponent": matches["AwayTeam"],
        "GoalsFor": matches["FTHG"],
        "GoalsAgainst": matches["FTAG"],
        "xGFor": matches["HomeXG"],
        "xGAgainst": matches["AwayXG"],
    })

    away = pd.DataFrame({
        "Date": matches["Date"],
        "Team": matches["AwayTeam"],
        "Venue": "Away",
        "Opponent": matches["HomeTeam"],
        "GoalsFor": matches["FTAG"],
        "GoalsAgainst": matches["FTHG"],
        "xGFor": matches["AwayXG"],
        "xGAgainst": matches["HomeXG"],
    })

    history = pd.concat(
        [home, away],
        ignore_index=True,
    )

    history["Points"] = np.select(
        [
            history["GoalsFor"] > history["GoalsAgainst"],
            history["GoalsFor"] == history["GoalsAgainst"],
        ],
        [3, 1],
        default=0,
    )

    history["Result"] = np.select(
        [
            history["GoalsFor"] > history["GoalsAgainst"],
            history["GoalsFor"] == history["GoalsAgainst"],
        ],
        ["W", "D"],
        default="L",
    )

    return history.sort_values(
        ["Team", "Date"]
    ).reset_index(drop=True)


# ============================================================
# MODEL FEATURE ENGINEERING
# ============================================================

def recent_form(history, team, venue, as_of_date):
    team_history = history[
        (history["Team"] == team)
        & (history["Date"] < as_of_date)
    ].copy()

    venue_history = team_history[
        team_history["Venue"] == venue
    ]

    features = {}

    for window in WINDOWS:
        for metric in METRICS:
            overall = team_history[metric].tail(window)
            venue_values = venue_history[metric].tail(window)

            features[
                f"Overall_{metric}_Last{window}"
            ] = (
                float(overall.mean())
                if not overall.empty
                else DEFAULTS[metric]
            )

            features[
                f"Venue_{metric}_Last{window}"
            ] = (
                float(venue_values.mean())
                if not venue_values.empty
                else DEFAULTS[metric]
            )

    return features, team_history, venue_history


def build_prediction_features(
    history,
    home_team,
    away_team,
    as_of_date,
    expected_features,
    current_elo,
):
    home_form, home_history, home_venue = recent_form(
        history, home_team, "Home", as_of_date
    )

    away_form, away_history, away_venue = recent_form(
        history, away_team, "Away", as_of_date
    )

    differences = {
        f"Diff_{feature}": (
            home_form[feature] - away_form[feature]
        )
        for feature in home_form
    }

    # Reconstruct Elo ratings from completed fixtures using the same
    # pre-match and same-day update rules as V2 training.
    if "Diff_Elo" in expected_features:
        differences["Diff_Elo"] = float(
            current_elo.get(home_team, 1500.0)
            - current_elo.get(away_team, 1500.0)
        )

    X = pd.DataFrame([differences])

    missing = [
        feature for feature in expected_features
        if feature not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = X[expected_features]

    metadata = {
        "home_form": home_form,
        "away_form": away_form,
        "home_matches": len(home_history),
        "away_matches": len(away_history),
        "home_venue_matches": len(home_venue),
        "away_venue_matches": len(away_venue),
    }

    return X, metadata


# ============================================================
# TEAM ANALYTICS
# ============================================================

def get_recent_results(history, team, limit=5):
    team_matches = history[
        history["Team"] == team
    ].sort_values("Date")

    return team_matches.tail(limit)


def get_head_to_head(matches, home_team, away_team):
    mask = (
        (
            (matches["HomeTeam"] == home_team)
            & (matches["AwayTeam"] == away_team)
        )
        |
        (
            (matches["HomeTeam"] == away_team)
            & (matches["AwayTeam"] == home_team)
        )
    )

    return (
        matches[mask]
        .sort_values("Date", ascending=False)
        .head(5)
    )


def format_team_name(team):
    return TEAM_NAMES.get(team, team)


def render_form_badges(results):
    badges = []

    for result in results["Result"]:
        css_class = {
            "W": "form-win",
            "D": "form-draw",
            "L": "form-loss",
        }[result]

        badges.append(
            f'<span class="{css_class}">{result}</span>'
        )

    return (
        '<div class="form-row">'
        + "".join(badges)
        + "</div>"
    )


def show_stat_card(label, value):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

def main():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                FOOTBALL ANALYTICS &middot; MACHINE LEARNING
            </div>
            <div class="hero-title">&#9917; PitchIQ</div>
            <div class="hero-subtitle">
                Premier League Match Prediction & Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        artifact = load_model()
        matches = load_matches()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()
        return

    model = artifact["model"]
    features = artifact["features"]

    # Compute post-match Elo state from completed historical matches.
    # Only uses results available in the bundled dataset.
    _, current_elo = compute_pre_match_features(
        matches.sort_values(["Date", "Season", "HomeTeam", "AwayTeam"]),
        return_state=True,
    )

    history = create_team_history(matches)

    latest_date = matches["Date"].max()
    as_of_date = latest_date + pd.Timedelta(days=1)
    latest_season = matches["Season"].max()

    latest_teams = sorted(
        matches.loc[
            matches["Season"] == latest_season,
            "HomeTeam",
        ].unique()
    )

    all_teams = sorted(
        set(matches["HomeTeam"])
        | set(matches["AwayTeam"])
    )

    st.sidebar.markdown("## Match settings")

    scope = st.sidebar.radio(
        "Select teams from",
        ["Latest dataset season", "All historical teams"],
    )

    teams = (
        latest_teams
        if scope == "Latest dataset season"
        else all_teams
    )

    home_index = (
        teams.index("Arsenal")
        if "Arsenal" in teams else 0
    )

    away_index = (
        teams.index("Man City")
        if "Man City" in teams else 1
    )

    home_team = st.sidebar.selectbox(
        "Home team", teams, index=home_index
    )

    away_team = st.sidebar.selectbox(
        "Away team", teams, index=away_index
    )

    st.sidebar.divider()

    st.sidebar.caption(
        f"Historical data through {latest_date.date()}"
    )

    st.sidebar.caption(
        f"Training matches: "
        f"{artifact['training_matches']:,}"
    )

    st.sidebar.caption(
        "Model: Logistic Regression + Elo | C = 0.01"
    )

    if home_team == away_team:
        st.warning(
            "Please select two different teams."
        )
        return

    # ========================================================
    # MATCH PREVIEW
    # ========================================================

    st.markdown("## Match preview")

    left, middle, right = st.columns(
        [4, 1, 4],
        vertical_alignment="center",
    )

    with left:
        st.markdown(
            f"""
            <div class="match-card">
                <div class="team-name">
                    {format_team_name(home_team)}
                </div>
                <div class="team-role">HOME</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        st.markdown(
            '<div class="versus">VS</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="match-card">
                <div class="team-name">
                    {format_team_name(away_team)}
                </div>
                <div class="team-role">AWAY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Hypothetical match based on historical data "
        f"through {latest_date.date()}."
    )

    if st.button(
        "Generate match prediction",
        type="primary",
        width='stretch',
    ):
        X, metadata = build_prediction_features(
            history,
            home_team,
            away_team,
            as_of_date,
            features,
            current_elo,
        )

        raw_probabilities = model.predict_proba(X)[0]

        probabilities = dict(
            zip(model.classes_, raw_probabilities)
        )

        home_prob = float(probabilities["H"])
        draw_prob = float(probabilities["D"])
        away_prob = float(probabilities["A"])

        st.divider()
        st.markdown("## Match outcome probabilities")

        cols = st.columns(3)

        for col, label, value in zip(
            cols,
            ["Home win", "Draw", "Away win"],
            [home_prob, draw_prob, away_prob],
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-label">
                            {label}
                        </div>
                        <div class="prediction-value">
                            {value:.1%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.progress(
            home_prob,
            text=f"Home win: {home_prob:.1%}",
        )

        st.progress(
            draw_prob,
            text=f"Draw: {draw_prob:.1%}",
        )

        st.progress(
            away_prob,
            text=f"Away win: {away_prob:.1%}",
        )

        most_likely = max(
            probabilities,
            key=probabilities.get,
        )

        st.info(
            "Most likely outcome: "
            f"**{OUTCOME_LABELS[most_likely]}**"
        )

        # ====================================================
        # TEAM COMPARISON
        # ====================================================

        st.divider()
        st.markdown("## Team performance comparison")

        st.markdown(
            '<div class="section-description">'
            'Average statistics over the last 10 '
            'matches at the relevant venue.'
            '</div>',
            unsafe_allow_html=True,
        )

        home_form = metadata["home_form"]
        away_form = metadata["away_form"]

        comparison = pd.DataFrame({
            "Metric": [
                "Expected goals (xG)",
                "Expected goals against (xGA)",
                "Goals scored",
                "Goals conceded",
                "Points per match",
            ],
            format_team_name(home_team): [
                home_form["Venue_xGFor_Last10"],
                home_form["Venue_xGAgainst_Last10"],
                home_form["Venue_GoalsFor_Last10"],
                home_form["Venue_GoalsAgainst_Last10"],
                home_form["Venue_Points_Last10"],
            ],
            format_team_name(away_team): [
                away_form["Venue_xGFor_Last10"],
                away_form["Venue_xGAgainst_Last10"],
                away_form["Venue_GoalsFor_Last10"],
                away_form["Venue_GoalsAgainst_Last10"],
                away_form["Venue_Points_Last10"],
            ],
        })

        st.dataframe(
            comparison.round(2),
            hide_index=True,
            width='stretch',
        )

        # ====================================================
        # RECENT FORM
        # ====================================================

        st.divider()
        st.markdown("## Recent form")

        left, right = st.columns(2)

        for col, team in [
            (left, home_team),
            (right, away_team),
        ]:
            with col:
                st.subheader(format_team_name(team))

                results = get_recent_results(
                    history, team, limit=5
                )

                st.markdown(
                    render_form_badges(results),
                    unsafe_allow_html=True,
                )

                display = results[
                    [
                        "Date",
                        "Opponent",
                        "Venue",
                        "GoalsFor",
                        "GoalsAgainst",
                        "Result",
                    ]
                ].copy()

                display["Date"] = (
                    display["Date"]
                    .dt.strftime("%d %b %Y")
                )

                display = display.rename(
                    columns={
                        "GoalsFor": "GF",
                        "GoalsAgainst": "GA",
                    }
                )

                st.dataframe(
                    display.iloc[::-1],
                    hide_index=True,
                    width='stretch',
                )

        # ====================================================
        # HEAD-TO-HEAD
        # ====================================================

        st.divider()
        st.markdown("## Head-to-head")

        h2h = get_head_to_head(
            matches, home_team, away_team
        )

        if h2h.empty:
            st.info(
                "No previous Premier League meetings "
                "found in the dataset."
            )
        else:
            display = h2h[
                [
                    "Date",
                    "HomeTeam",
                    "FTHG",
                    "FTAG",
                    "AwayTeam",
                ]
            ].copy()

            display["Date"] = (
                display["Date"]
                .dt.strftime("%d %b %Y")
            )

            display["Score"] = (
                display["FTHG"].astype(str)
                + " - "
                + display["FTAG"].astype(str)
            )

            display = display[
                [
                    "Date",
                    "HomeTeam",
                    "Score",
                    "AwayTeam",
                ]
            ]

            st.dataframe(
                display,
                hide_index=True,
                width='stretch',
            )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        with st.expander(
            "How does PitchIQ make predictions?"
        ):
            st.write(
                "PitchIQ uses multinomial Logistic "
                "Regression trained on historical "
                "Premier League match results and "
                "expected goals statistics."
            )

            st.write(
                "The model considers recent overall "
                "form and home/away-specific form "
                "over 3, 5 and 10 matches, plus Elo team-strength ratings."
            )

            st.write(
                "**Model:** Logistic Regression"
            )

            st.write(
                "**Regularisation:** C = 0.01"
            )

            st.write(
                f"**Input features:** {len(features)}"
            )

            st.write(
                "**Historical validation log loss:** "
                "approximately 0.957 (Elo V2)"
            )

            st.write(
                "Predictions do not account for "
                "current injuries, line-ups, transfers "
                "or other live information."
            )

    st.divider()

    st.caption(
        "PitchIQ | Premier League analytics powered "
        "by historical match results and expected goals. "
        "Predictions are estimates, not guarantees."
    )


if __name__ == "__main__":
    main()

