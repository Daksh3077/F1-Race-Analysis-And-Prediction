import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from predict_upcoming_race import render_prediction_tab

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="F1 Race Analytics and Predictions",
    page_icon="🏎️",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] { background-color: #0a0a0a; color: white; }
.stApp { background-color: #0a0a0a; }
h1, h2, h3, h4 { color: #ff1e00; font-family: sans-serif; }
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 2px solid #ff1e00;
}
div[data-testid="metric-container"] {
    background-color: #151515;
    border: 2px solid #ff1e00;
    padding: 20px;
    border-radius: 15px;
}
.stDataFrame { border: 1px solid #ff1e00; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🏎️ F1 RACE ANALYTICS AND PREDICTIONS</h1>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# TABS — Live Analysis (existing) + Prediction (new)
# =====================================================
tab_live, tab_predict = st.tabs(["📊 Live Race Analysis", "🔮 Predict Upcoming Race"])

with tab_live:

    # =====================================================
    # FASTF1 CACHE
    # =====================================================
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")

    # =====================================================
    # HELPER — plain feature engineering (no st.cache_data,
    # called only after a fresh session load)
    # =====================================================
    def engineer_features(laps):
        df = laps.copy().reset_index(drop=True)
        df = df.dropna(subset=["LapTime"])
        df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
        compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
        df["CompoundEncoded"] = df["Compound"].map(compound_map)
        df["NextPosition"] = df.groupby("Driver")["Position"].shift(-1)
        df["FutureOvertake"] = (df["NextPosition"] < df["Position"]).astype(int)
        df["AvgPace3"] = (
            df.groupby("Driver")["LapTimeSeconds"]
            .rolling(3).mean().reset_index(0, drop=True)
        )
        df["PositionTrend"] = df.groupby("Driver")["Position"].diff()
        df["LapDelta"] = df.groupby("Driver")["LapTimeSeconds"].diff()
        df["TyreAdvantage"] = (
            df["TyreLife"] - df.groupby("LapNumber")["TyreLife"].transform("mean")
        )
        df["Pace"] = 1 / df["LapTimeSeconds"]
        df["Sector1"] = df["Sector1Time"].dt.total_seconds()
        df["Sector2"] = df["Sector2Time"].dt.total_seconds()
        df["Sector3"] = df["Sector3Time"].dt.total_seconds()
        return df.fillna(0)

    # =====================================================
    # LOAD MODEL
    # =====================================================
    MODEL_PATH = "f1_overtake_model.pkl"

    @st.cache_resource
    def load_model():
        print("Files in directory:", os.listdir("."))
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            st.error(f"Model file not found: {MODEL_PATH}")
            return None

    try:
        model = load_model()
        model_loaded = model is not None
    except Exception as e:
        model_loaded = False
        st.error(f"Error loading model: {e}")

    # =====================================================
    # RACE CALENDAR (cached by year — pure/serialisable)
    # =====================================================
    @st.cache_data(show_spinner=False)
    def get_event_schedule(year):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        schedule = schedule[schedule["EventFormat"] != "testing"]
        return schedule[["EventName", "RoundNumber"]].dropna()

    # =====================================================
    # SIDEBAR
    # =====================================================
    st.sidebar.title("⚙️ Race Settings")
    year = st.sidebar.selectbox("Season", list(range(2018, 2027)), index=5)

    with st.spinner("Fetching race calendar..."):
        try:
            schedule_df = get_event_schedule(year)
            race_names = schedule_df["EventName"].tolist()
        except Exception as e:
            st.sidebar.error(f"Could not load calendar: {e}")
            race_names = ["Bahrain"]

    race = st.sidebar.selectbox("Grand Prix", race_names)

    # =====================================================
    # SESSION LOADING — keyed by (year, race) in session_state
    # =====================================================
    session_key = f"{year}_{race}"

    if st.session_state.get("session_key") != session_key:
        with st.spinner(f"Loading {year} {race} race data… (first load may take a minute)"):
            try:
                session = fastf1.get_session(year, race, "R")
                session.load(laps=True, telemetry=True, weather=True, messages=False)
                df = engineer_features(session.laps)
                st.session_state["session_key"] = session_key
                st.session_state["session"] = session
                st.session_state["df"] = df
            except Exception as e:
                st.error(f"❌ Could not load session: {e}")
                st.stop()
    else:
        session = st.session_state["session"]
        df = st.session_state["df"]

    # =====================================================
    # DYNAMIC DRIVER LIST
    # =====================================================
    all_drivers = sorted(df["Driver"].dropna().unique().tolist())
    st.sidebar.markdown("---")
    driver_choice = st.sidebar.selectbox("Telemetry Driver", all_drivers, index=0)

    # =====================================================
    # TELEMETRY HELPERS (keyed by race + driver — never hash session)
    # =====================================================
    def get_telemetry_one(driver):
        key = f"tel_{session_key}_{driver}"
        if key not in st.session_state:
            lap = session.laps.pick_driver(driver).pick_fastest()
            st.session_state[key] = lap.get_car_data().add_distance()
        return st.session_state[key]

    def get_telemetry_two(d1, d2):
        key = f"tel2_{session_key}_{d1}_{d2}"
        if key not in st.session_state:
            lap1 = session.laps.pick_driver(d1).pick_fastest()
            tel1 = lap1.get_car_data().add_distance()
            lap2 = session.laps.pick_driver(d2).pick_fastest()
            tel2 = lap2.get_car_data().add_distance()
            st.session_state[key] = (tel1, tel2)
        return st.session_state[key]

    # =====================================================
    # FEATURES & PREDICTIONS
    # =====================================================
    features = [
        "LapNumber", "Position", "TyreLife", "CompoundEncoded",
        "LapTimeSeconds", "AvgPace3", "PositionTrend",
        "LapDelta", "TyreAdvantage", "Pace"
    ]

    if model_loaded:
        df["OvertakeProbability"] = model.predict_proba(df[features])[:, 1] * 100
    else:
        df["OvertakeProbability"] = np.nan

    latest = df.sort_values("LapNumber").groupby("Driver").tail(1)

    # =====================================================
    # TOP METRICS
    # =====================================================
    col1, col2, col3, col4 = st.columns(4)

    leader = latest.sort_values("Position").iloc[0]["Driver"]
    highest_prob = round(latest["OvertakeProbability"].max(), 2) if model_loaded else "N/A"
    avg_prob = round(latest["OvertakeProbability"].mean(), 2) if model_loaded else "N/A"
    current_lap = int(latest["LapNumber"].max())

    with col1: st.metric("🏁 Race Leader", leader)
    with col2: st.metric("🔥 Highest Overtake %", highest_prob)
    with col3: st.metric("📊 Avg Probability", avg_prob)
    with col4: st.metric("🛞 Current Lap", current_lap)

    st.markdown("---")

    # =====================================================
    # LIVE TABLE
    # =====================================================
    st.subheader("📊 LIVE OVERTAKE PROBABILITIES")
    table = (
        latest[["Driver", "Position", "Compound", "TyreLife", "OvertakeProbability"]]
        .sort_values("OvertakeProbability", ascending=False)
    )
    st.dataframe(table, use_container_width=True)

    # =====================================================
    # BAR CHART
    # =====================================================
    if model_loaded:
        st.subheader("🔥 OVERTAKE INTENSITY")
        bar_fig = px.bar(
            table, x="Driver", y="OvertakeProbability",
            color="OvertakeProbability", template="plotly_dark", text_auto=".2f"
        )
        bar_fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
        st.plotly_chart(bar_fig, use_container_width=True)

    # =====================================================
    # POSITION CHANGES
    # =====================================================
    st.subheader("📈 DRIVER POSITION CHANGES")
    pos_fig = px.line(df, x="LapNumber", y="Position", color="Driver", template="plotly_dark")
    pos_fig.update_yaxes(autorange="reversed")
    pos_fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
    st.plotly_chart(pos_fig, use_container_width=True)

    # =====================================================
    # TYRE DEGRADATION
    # =====================================================
    st.subheader("🛞 TYRE DEGRADATION ANALYSIS")
    tyre_fig = px.scatter(
        df, x="TyreLife", y="LapTimeSeconds", color="Compound",
        template="plotly_dark", hover_data=["Driver"]
    )
    tyre_fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
    st.plotly_chart(tyre_fig, use_container_width=True)

    # =====================================================
    # PIT STOP STRATEGY
    # =====================================================
    st.subheader("🛞 PIT STOP STRATEGY")
    pit_data = (
        df.groupby(["Driver", "Stint"])
        .agg({"LapNumber": "min", "Compound": "first"})
        .reset_index()
    )
    pit_fig = px.scatter(
        pit_data, x="LapNumber", y="Driver", color="Compound",
        template="plotly_dark", title="Pit Stop Timeline"
    )
    pit_fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
    st.plotly_chart(pit_fig, use_container_width=True)

    # =====================================================
    # SECTOR PERFORMANCE
    # =====================================================
    st.subheader("⚡ SECTOR PERFORMANCE")
    sector_avg = df.groupby("Driver")[["Sector1", "Sector2", "Sector3"]].mean().reset_index()
    sec_fig = go.Figure()
    for s in ["Sector1", "Sector2", "Sector3"]:
        sec_fig.add_trace(go.Bar(x=sector_avg["Driver"], y=sector_avg[s], name=s))
    sec_fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font_color="white", title="Average Sector Performance"
    )
    st.plotly_chart(sec_fig, use_container_width=True)

    # =====================================================
    # FASTEST LAPS
    # =====================================================
    st.subheader("⚡ FASTEST LAPS")
    fastest = df.groupby("Driver")["LapTimeSeconds"].min().reset_index().sort_values("LapTimeSeconds")
    fast_fig = px.bar(fastest, x="Driver", y="LapTimeSeconds", color="LapTimeSeconds", template="plotly_dark")
    fast_fig.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white")
    st.plotly_chart(fast_fig, use_container_width=True)

    # =====================================================
    # SINGLE DRIVER TELEMETRY
    # =====================================================
    st.subheader(f"🏎️ FASTEST LAP TELEMETRY — {driver_choice}")
    with st.spinner(f"Loading telemetry for {driver_choice}…"):
        try:
            telemetry = get_telemetry_one(driver_choice)
            tele_fig = go.Figure()
            tele_fig.add_trace(go.Scatter(
                x=telemetry["Distance"], y=telemetry["Speed"],
                mode="lines", name="Speed", line=dict(color="#ff1e00")
            ))
            tele_fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a", font_color="white",
                title=f"{driver_choice} — Fastest Lap Speed Trace",
                xaxis_title="Distance (m)", yaxis_title="Speed (km/h)"
            )
            st.plotly_chart(tele_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Telemetry unavailable for {driver_choice}: {e}")

    st.markdown("---")

    # =====================================================
    # DRIVER vs DRIVER — FULL RACE COMPARISON
    # =====================================================
    st.subheader("⚔️ DRIVER vs DRIVER — FULL RACE COMPARISON")

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        driver1 = st.selectbox("Select Driver 1", all_drivers, index=0, key="d1")
    with cmp2:
        driver2 = st.selectbox("Select Driver 2", all_drivers, index=min(1, len(all_drivers)-1), key="d2")

    if driver1 == driver2:
        st.warning("Please select two different drivers.")
    else:
        d1_df = df[df["Driver"] == driver1]
        d2_df = df[df["Driver"] == driver2]

        # Lap times
        st.markdown("#### 📉 Lap Time Battle")
        lt_fig = go.Figure()
        lt_fig.add_trace(go.Scatter(x=d1_df["LapNumber"], y=d1_df["LapTimeSeconds"],
                                     mode="lines+markers", name=driver1, line=dict(color="#ff1e00")))
        lt_fig.add_trace(go.Scatter(x=d2_df["LapNumber"], y=d2_df["LapTimeSeconds"],
                                     mode="lines+markers", name=driver2, line=dict(color="#00d2ff")))
        lt_fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                              plot_bgcolor="#0a0a0a", font_color="white",
                              title=f"{driver1} vs {driver2} — Lap Times",
                              xaxis_title="Lap Number", yaxis_title="Lap Time (s)")
        st.plotly_chart(lt_fig, use_container_width=True)

        # Position battle
        st.markdown("#### 🏁 Position Battle")
        pf = go.Figure()
        pf.add_trace(go.Scatter(x=d1_df["LapNumber"], y=d1_df["Position"],
                                 mode="lines+markers", name=driver1, line=dict(color="#ff1e00")))
        pf.add_trace(go.Scatter(x=d2_df["LapNumber"], y=d2_df["Position"],
                                 mode="lines+markers", name=driver2, line=dict(color="#00d2ff")))
        pf.update_yaxes(autorange="reversed")
        pf.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                          plot_bgcolor="#0a0a0a", font_color="white",
                          title=f"{driver1} vs {driver2} — Race Position",
                          xaxis_title="Lap Number", yaxis_title="Position")
        st.plotly_chart(pf, use_container_width=True)

        # Tyre strategy
        st.markdown("#### 🛞 Tyre Strategy")
        COMPOUND_COLORS = {
            "SOFT": "#e8002d", "MEDIUM": "#ffd700", "HARD": "#f0f0f0",
            "INTERMEDIATE": "#39b54a", "WET": "#0067ff"
        }
        sc1, sc2 = st.columns(2)
        for col, drv, drv_df in [(sc1, driver1, d1_df), (sc2, driver2, d2_df)]:
            stints = (
                drv_df.groupby("Stint")
                .agg(LapStart=("LapNumber", "min"), LapEnd=("LapNumber", "max"), Compound=("Compound", "first"))
                .reset_index()
            )
            sf = go.Figure()
            for _, row in stints.iterrows():
                color = COMPOUND_COLORS.get(row["Compound"], "#888888")
                sf.add_trace(go.Bar(
                    x=[row["LapEnd"] - row["LapStart"] + 1], y=[drv],
                    base=[row["LapStart"] - 1], orientation="h",
                    marker_color=color, name=row["Compound"],
                    text=f"{row['Compound']} Laps {int(row['LapStart'])}–{int(row['LapEnd'])}",
                    hovertemplate="%{text}<extra></extra>"
                ))
            sf.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                              plot_bgcolor="#0a0a0a", font_color="white",
                              title=f"{drv} Tyre Strategy", xaxis_title="Lap Number",
                              barmode="stack", height=200, showlegend=False)
            with col:
                st.plotly_chart(sf, use_container_width=True)

        # Sector times
        st.markdown("#### ⚡ Average Sector Times")
        sec_cmp = go.Figure()
        for drv, drv_df, color in [(driver1, d1_df, "#ff1e00"), (driver2, d2_df, "#00d2ff")]:
            sec_cmp.add_trace(go.Bar(
                x=["Sector 1", "Sector 2", "Sector 3"],
                y=[drv_df["Sector1"].replace(0, np.nan).mean(),
                   drv_df["Sector2"].replace(0, np.nan).mean(),
                   drv_df["Sector3"].replace(0, np.nan).mean()],
                name=drv, marker_color=color
            ))
        sec_cmp.update_layout(barmode="group", template="plotly_dark",
                               paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font_color="white",
                               title=f"{driver1} vs {driver2} — Avg Sector Times (s)", yaxis_title="Time (s)")
        st.plotly_chart(sec_cmp, use_container_width=True)

        # Telemetry overlay
        st.markdown("#### 🏎️ Fastest Lap Speed Trace Overlay")
        with st.spinner("Loading comparison telemetry…"):
            try:
                tel1, tel2 = get_telemetry_two(driver1, driver2)
                ov = go.Figure()
                ov.add_trace(go.Scatter(x=tel1["Distance"], y=tel1["Speed"],
                                         mode="lines", name=driver1, line=dict(color="#ff1e00")))
                ov.add_trace(go.Scatter(x=tel2["Distance"], y=tel2["Speed"],
                                         mode="lines", name=driver2, line=dict(color="#00d2ff")))
                ov.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                                  plot_bgcolor="#0a0a0a", font_color="white",
                                  title=f"{driver1} vs {driver2} — Fastest Lap Speed Trace",
                                  xaxis_title="Distance (m)", yaxis_title="Speed (km/h)")
                st.plotly_chart(ov, use_container_width=True)
            except Exception as e:
                st.error(f"Telemetry overlay unavailable: {e}")

        # Head-to-head summary cards
        st.markdown("#### 📋 Head-to-Head Race Summary")
        hc1, hc2 = st.columns(2)
        for col, drv, drv_df, color in [
            (hc1, driver1, d1_df, "#ff1e00"),
            (hc2, driver2, d2_df, "#00d2ff")
        ]:
            last = drv_df.sort_values("LapNumber").iloc[-1]
            best = drv_df["LapTimeSeconds"].replace(0, np.nan).min()
            avg = drv_df["LapTimeSeconds"].replace(0, np.nan).mean()
            gained = int(drv_df.sort_values("LapNumber").iloc[0]["Position"] - last["Position"])
            compounds = ", ".join(drv_df["Compound"].dropna().unique())

            with col:
                st.markdown(f"""
                <div style="background:#151515;padding:16px;border-radius:12px;border-left:5px solid {color};">
                    <h3 style="color:white;">{drv}</h3>
                    <p style="color:#ccc;">Finish Position: <b>{int(last['Position'])}</b></p>
                    <p style="color:#ccc;">Best Lap: <b>{round(best,3) if not np.isnan(best) else 'N/A'}s</b></p>
                    <p style="color:#ccc;">Avg Lap: <b>{round(avg,3) if not np.isnan(avg) else 'N/A'}s</b></p>
                    <p style="color:#ccc;">Positions Gained: <b>{gained:+d}</b></p>
                    <p style="color:#ccc;">Compounds: <b>{compounds}</b></p>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # TRACK CONDITIONS
    # =====================================================
    st.subheader("🌦️ TRACK CONDITIONS")
    try:
        weather = session.weather_data
        wf = go.Figure()
        wf.add_trace(go.Scatter(x=weather["Time"], y=weather["AirTemp"], mode="lines", name="Air Temp"))
        wf.add_trace(go.Scatter(x=weather["Time"], y=weather["TrackTemp"], mode="lines", name="Track Temp"))
        wf.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                          plot_bgcolor="#0a0a0a", font_color="white", title="Track Temperature Analysis")
        st.plotly_chart(wf, use_container_width=True)
    except Exception:
        st.info("Weather data not available for this session.")

    # =====================================================
    # AI RACE INSIGHTS
    # =====================================================
    if model_loaded:
        st.subheader("🤖 AI RACE INSIGHTS")
        top = latest.sort_values("OvertakeProbability", ascending=False).iloc[0]
        bottom = latest.sort_values("OvertakeProbability").iloc[0]
        leader_drv = latest.sort_values("Position").iloc[0]["Driver"]

        st.success(f"🔥 {top['Driver']} has the highest overtake probability at {round(top['OvertakeProbability'],2)}%")
        st.info(f"🛞 {bottom['Driver']} currently has the lowest overtake probability at {round(bottom['OvertakeProbability'],2)}%")
        st.warning(f"🏁 Current race leader: {leader_drv}")

    # =====================================================
    # LIVE LEADERBOARD
    # =====================================================
    st.subheader("🏆 LIVE LEADERBOARD")
    for _, row in latest.sort_values("Position").iterrows():
        prob_txt = f"| Overtake Probability: {round(row['OvertakeProbability'],2)}%" if model_loaded else ""
        st.markdown(f"""
        <div style="background:#151515;padding:10px;border-radius:10px;
                    border-left:5px solid #ff1e00;margin-bottom:10px;">
            <h4 style="color:white;">P{int(row['Position'])} — {row['Driver']}</h4>
            <p style="color:#bbb;">Tyre: {row['Compound']} | Tyre Life: {int(row['TyreLife'])} {prob_txt}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<center><h4 style='color:#ff1e00'>Formula 1 Analytics Dashboard</h4></center>",
                unsafe_allow_html=True)

with tab_predict:
    render_prediction_tab()
