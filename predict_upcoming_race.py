"""
Pre-race finishing-order prediction tab.
Drop this file next to app.py in your repo.
Requires f1_race_predictor.pkl and historical_features.parquet in the repo root.
"""

import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timezone

MODEL_PATH = "f1_race_predictor.pkl"
HIST_PATH  = "historical_features.parquet"


@st.cache_resource
def load_predictor():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


@st.cache_data(show_spinner=False)
def load_historical_features():
    return pd.read_parquet(HIST_PATH)


@st.cache_data(show_spinner=False)
def get_next_event(year):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule["EventFormat"] != "testing"]
    now = pd.Timestamp.now()          # timezone-naive to match EventDate
    upcoming = schedule[schedule["EventDate"] >= now]
    if upcoming.empty:
        return None
    return upcoming.sort_values("EventDate").iloc[0]


def _latest_stat(hist_df, key_col, key_val, col):
    """Get the most recent value of `col` for a given key, with fallback."""
    if col not in hist_df.columns:
        return np.nan
    rows = hist_df[hist_df[key_col] == key_val].sort_values(["Year", "Round"])
    if rows.empty:
        return np.nan
    val = rows[col].dropna()
    return val.iloc[-1] if not val.empty else np.nan


def try_get_grid(year, rnd):
    """Returns real grid from qualifying if it has happened, else None."""
    try:
        q = fastf1.get_session(year, rnd, "Q")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        res = q.results[["Abbreviation", "TeamName", "Position"]].copy()
        res = res.rename(columns={"Position": "GridPosition"})
        return res
    except Exception:
        return None


def build_prediction_table(event, hist_df):
    year       = int(pd.Timestamp(event["EventDate"]).year)
    rnd        = int(event["RoundNumber"])
    event_name = event["EventName"]

    grid_df    = try_get_grid(year, rnd)
    quali_done = grid_df is not None

    if not quali_done:
        latest  = hist_df.sort_values(["Year", "Round"]).groupby("Abbreviation").tail(1)
        grid_df = latest[["Abbreviation", "TeamName"]].copy()
        grid_df["GridPosition"] = grid_df["Abbreviation"].apply(
            lambda a: _latest_stat(hist_df, "Abbreviation", a, "DriverAvgGridLast5")
        )

    rows = []
    for _, r in grid_df.iterrows():
        drv, team = r["Abbreviation"], r["TeamName"]

        circuit_hist = hist_df[
            (hist_df["Abbreviation"] == drv) &
            (hist_df["EventName"]    == event_name)
        ]
        circuit_avg  = circuit_hist["Position"].mean() if not circuit_hist.empty else np.nan
        driver_form  = _latest_stat(hist_df, "Abbreviation", drv, "DriverAvgFinishLast5")

        rows.append({
            "Abbreviation":         drv,
            "TeamName":             team,
            "GridPosition":         r["GridPosition"],
            "DriverAvgFinishLast5": driver_form,
            "DriverAvgGridLast5":   _latest_stat(hist_df, "Abbreviation", drv, "DriverAvgGridLast5"),
            "DriverDNFRateLast10":  _latest_stat(hist_df, "Abbreviation", drv, "DriverDNFRateLast10"),
            "DriverPointsCumSeason":_latest_stat(hist_df, "Abbreviation", drv, "DriverPointsCumSeason"),
            "TeamAvgFinishLast5":   _latest_stat(hist_df, "TeamName", team, "TeamAvgFinishLast5"),
            "TeamPointsCumSeason":  _latest_stat(hist_df, "TeamName", team, "TeamPointsCumSeason"),
            "CircuitAvgFinish":     circuit_avg if pd.notna(circuit_avg) else driver_form,
            "DriverRaceCount":      _latest_stat(hist_df, "Abbreviation", drv, "DriverRaceCount"),
        })

    return pd.DataFrame(rows), quali_done


def render_prediction_tab():
    st.subheader("🔮 UPCOMING RACE PREDICTION")

    try:
        model, features = load_predictor()
        hist_df         = load_historical_features()
    except Exception as e:
        st.error(
            f"Prediction model not found. Run train_race_predictor.py "
            f"and commit the resulting files to your repo. ({e})"
        )
        return

    year  = datetime.now(timezone.utc).year
    event = get_next_event(year)
    if event is None:
        st.info("No upcoming race found on the current calendar.")
        return

    st.markdown(f"### {event['EventName']} — Round {int(event['RoundNumber'])} ({year})")
    st.caption(f"Race date: {pd.Timestamp(event['EventDate']).date()}")

    with st.spinner("Building prediction..."):
        pred_df, quali_done = build_prediction_table(event, hist_df)

        # Ensure every feature the model expects exists in pred_df
        # If a column is missing, fill it with the column median from
        # historical data (or a sensible midfield default of 10).
        for col in features:
            if col not in pred_df.columns:
                # try to get a global median from history
                if col in hist_df.columns:
                    pred_df[col] = hist_df[col].median()
                else:
                    pred_df[col] = 10.0   # sensible midfield default
            else:
                # fill NaNs within the column
                median_val = pred_df[col].median()
                if pd.isna(median_val) and col in hist_df.columns:
                    median_val = hist_df[col].median()
                if pd.isna(median_val):
                    median_val = 10.0
                pred_df[col] = pred_df[col].fillna(median_val)

        pred_df["PredictedPosition"] = model.predict(pred_df[features])
        pred_df = pred_df.sort_values("PredictedPosition").reset_index(drop=True)
        pred_df["PredictedRank"] = range(1, len(pred_df) + 1)

    if quali_done:
        st.success("✅ Using the actual qualifying grid for this prediction.")
    else:
        st.warning(
            "⚠️ Qualifying hasn't happened yet — grid is estimated from "
            "recent form. Re-run after qualifying for a sharper prediction."
        )

    # Results table
    st.dataframe(
        pred_df[["PredictedRank", "Abbreviation", "TeamName", "GridPosition", "PredictedPosition"]]
        .rename(columns={"Abbreviation": "Driver", "GridPosition": "Grid",
                         "PredictedPosition": "Predicted Score (lower = better)"}),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    fig = px.bar(
        pred_df, x="Abbreviation", y="PredictedPosition",
        color="TeamName", template="plotly_dark",
        title="Predicted Finishing Position (lower = better)",
        labels={"Abbreviation": "Driver", "PredictedPosition": "Predicted Position"}
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    # Podium cards
    st.markdown("#### 🏆 Predicted Podium")
    podium = pred_df.head(3)
    medals = ["🥇", "🥈", "🥉"]
    cols   = st.columns(3)
    for col, (_, row), medal in zip(cols, podium.iterrows(), medals):
        with col:
            st.markdown(f"""
            <div style="background:#151515;padding:16px;border-radius:12px;
                        border-left:5px solid #ff1e00;text-align:center;">
                <h2>{medal}</h2>
                <h4 style="color:white;">{row['Abbreviation']}</h4>
                <p style="color:#bbb;">{row['TeamName']}</p>
            </div>""", unsafe_allow_html=True)
