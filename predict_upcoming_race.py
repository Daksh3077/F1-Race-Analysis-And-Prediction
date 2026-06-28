"""
Pre-race finishing-order prediction tab.
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

# 2026 F1 driver lineup — used when qualifying hasn't happened yet
DRIVERS_2026 = [
    {"Abbreviation": "VER", "TeamName": "Red Bull Racing"},
    {"Abbreviation": "LAW", "TeamName": "Red Bull Racing"},
    {"Abbreviation": "NOR", "TeamName": "McLaren"},
    {"Abbreviation": "PIA", "TeamName": "McLaren"},
    {"Abbreviation": "LEC", "TeamName": "Ferrari"},
    {"Abbreviation": "HAM", "TeamName": "Ferrari"},
    {"Abbreviation": "RUS", "TeamName": "Mercedes"},
    {"Abbreviation": "ANT", "TeamName": "Mercedes"},
    {"Abbreviation": "ALO", "TeamName": "Aston Martin"},
    {"Abbreviation": "STR", "TeamName": "Aston Martin"},
    {"Abbreviation": "GAS", "TeamName": "Alpine"},
    {"Abbreviation": "DOO", "TeamName": "Alpine"},
    {"Abbreviation": "ALB", "TeamName": "Williams"},
    {"Abbreviation": "SAI", "TeamName": "Williams"},
    {"Abbreviation": "HUL", "TeamName": "Kick Sauber"},
    {"Abbreviation": "BOR", "TeamName": "Kick Sauber"},
    {"Abbreviation": "TSU", "TeamName": "RB"},
    {"Abbreviation": "HAD", "TeamName": "RB"},
    {"Abbreviation": "BEA", "TeamName": "Haas F1 Team"},
    {"Abbreviation": "OCO", "TeamName": "Haas F1 Team"},
]


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
    # Compare dates only (not timestamps) so today's race still shows
    # as "upcoming" even if we're past midnight on race day.
    today = pd.Timestamp.now().normalize()
    upcoming = schedule[schedule["EventDate"].dt.normalize() >= today]
    if upcoming.empty:
        return None
    return upcoming.sort_values("EventDate").iloc[0]


def try_get_grid(year, rnd):
    """Real qualifying grid if available, else None."""
    try:
        q = fastf1.get_session(year, rnd, "Q")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        res = q.results[["Abbreviation", "TeamName", "Position"]].copy()
        res = res.rename(columns={"Position": "GridPosition"})
        res = res.dropna(subset=["Abbreviation"])
        return res if not res.empty else None
    except Exception:
        return None


def _latest_stat(hist_df, key_col, key_val, col):
    if col not in hist_df.columns or key_col not in hist_df.columns:
        return np.nan
    rows = hist_df[hist_df[key_col] == key_val]
    if rows.empty:
        return np.nan
    val = rows[col].dropna()
    return float(val.iloc[-1]) if not val.empty else np.nan


def build_prediction_table(event, hist_df):
    year       = int(pd.Timestamp(event["EventDate"]).year)
    rnd        = int(event["RoundNumber"])
    event_name = event["EventName"]

    # Try real qualifying grid
    grid_df    = try_get_grid(year, rnd)
    quali_done = grid_df is not None

    if not quali_done:
        # Use hardcoded 2026 lineup — reliable, no network calls needed
        grid_df = pd.DataFrame(DRIVERS_2026)
        grid_df["GridPosition"] = 10.0  # estimated midfield; overridden by form below

    rows = []
    for _, r in grid_df.iterrows():
        drv  = str(r.get("Abbreviation", "UNK"))
        team = str(r.get("TeamName", "Unknown"))
        grid = float(r.get("GridPosition", 10.0))

        if not quali_done:
            # Estimate grid from recent qualifying form
            form_grid = _latest_stat(hist_df, "Abbreviation", drv, "DriverAvgGridLast5")
            grid = float(form_grid) if pd.notna(form_grid) else 10.0

        circuit_avg = np.nan
        if "Abbreviation" in hist_df.columns and "EventName" in hist_df.columns:
            ch = hist_df[
                (hist_df["Abbreviation"] == drv) &
                (hist_df["EventName"] == event_name)
            ]
            if not ch.empty and "Position" in ch.columns:
                circuit_avg = ch["Position"].mean()

        driver_form = _latest_stat(hist_df, "Abbreviation", drv, "DriverAvgFinishLast5")

        rows.append({
            "Abbreviation":          drv,
            "TeamName":              team,
            "GridPosition":          grid,
            "DriverAvgFinishLast5":  driver_form,
            "DriverAvgGridLast5":    _latest_stat(hist_df, "Abbreviation", drv, "DriverAvgGridLast5"),
            "DriverDNFRateLast10":   _latest_stat(hist_df, "Abbreviation", drv, "DriverDNFRateLast10"),
            "DriverPointsCumSeason": _latest_stat(hist_df, "Abbreviation", drv, "DriverPointsCumSeason"),
            "TeamAvgFinishLast5":    _latest_stat(hist_df, "TeamName", team, "TeamAvgFinishLast5"),
            "TeamPointsCumSeason":   _latest_stat(hist_df, "TeamName", team, "TeamPointsCumSeason"),
            "CircuitAvgFinish":      float(circuit_avg) if pd.notna(circuit_avg) else (driver_form if pd.notna(driver_form) else 10.0),
            "DriverRaceCount":       _latest_stat(hist_df, "Abbreviation", drv, "DriverRaceCount"),
        })

    return pd.DataFrame(rows), quali_done


def make_X(pred_df, features, hist_df):
    """Build a clean NaN-free feature matrix."""
    hist_medians = {}
    for col in features:
        if col in hist_df.columns:
            m = pd.to_numeric(hist_df[col], errors="coerce").median()
            hist_medians[col] = float(m) if pd.notna(m) else 10.0
        else:
            hist_medians[col] = 10.0

    X_rows = []
    for _, row in pred_df.iterrows():
        feature_row = {}
        for col in features:
            val = row.get(col, np.nan)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = np.nan
            if pd.isna(val) or np.isinf(val):
                val = hist_medians[col]
            feature_row[col] = val
        X_rows.append(feature_row)

    return pd.DataFrame(X_rows, columns=features)


def render_prediction_tab():
    st.subheader("🔮 UPCOMING RACE PREDICTION")

    try:
        model, features = load_predictor()
        hist_df         = load_historical_features()
    except Exception as e:
        st.error(f"Could not load model/data: {e}")
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
        X = make_X(pred_df, features, hist_df)

        try:
            pred_df["PredictedPosition"] = model.predict(X)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.write("X shape:", X.shape)
            st.dataframe(X)
            return

        pred_df = pred_df.sort_values("PredictedPosition").reset_index(drop=True)
        pred_df["PredictedRank"] = range(1, len(pred_df) + 1)

    if quali_done:
        st.success("✅ Using the actual qualifying grid for this prediction.")
    else:
        st.warning(
            "⚠️ Qualifying hasn't happened yet — grid estimated from recent form. "
            "Reload after qualifying for a sharper prediction."
        )

    st.dataframe(
        pred_df[["PredictedRank", "Abbreviation", "TeamName", "GridPosition", "PredictedPosition"]]
        .rename(columns={
            "Abbreviation":      "Driver",
            "GridPosition":      "Grid",
            "PredictedPosition": "Predicted Score (lower = better)"
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        pred_df, x="Abbreviation", y="PredictedPosition",
        color="TeamName", template="plotly_dark",
        title="Predicted Finishing Position (lower = better)",
        labels={"Abbreviation": "Driver", "PredictedPosition": "Predicted Position"}
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏆 Predicted Podium")
    cols   = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for col, (_, row), medal in zip(cols, pred_df.head(3).iterrows(), medals):
        with col:
            st.markdown(f"""
            <div style="background:#151515;padding:16px;border-radius:12px;
                        border-left:5px solid #ff1e00;text-align:center;">
                <h2>{medal}</h2>
                <h4 style="color:white;">{row['Abbreviation']}</h4>
                <p style="color:#bbb;">{row['TeamName']}</p>
            </div>""", unsafe_allow_html=True)
