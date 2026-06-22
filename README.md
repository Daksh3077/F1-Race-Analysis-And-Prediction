🏎️ F1 Race Analytics


Analyze past races. Predict the next one. All in one dashboard.



Live App → f1-race-analysis3077.streamlit.app


What is this?

F1 Race Analytics is a personal project I built out of a mix of obsession with Formula 1 and wanting to actually apply machine learning to something I care about. It started as a race analysis tool, and I kept adding features until it could predict upcoming race results too.

It pulls real F1 data using the FastF1 library, runs it through trained ML models, and serves everything through a clean Streamlit dashboard — dark themed, obviously.


Features

📊 Live Race Analysis Tab

Pick any season (2018–2026) and any Grand Prix from the sidebar, and the app loads the full race data for that event.


Overtake probability — a trained classifier predicts which driver is most likely to overtake on the next lap, based on tyre life, lap delta, position trend, and pace
Position changes — lap-by-lap position chart for all 20 (now 22) drivers
Tyre degradation — scatter plot showing how each compound degrades over stint length
Pit stop strategy timeline — visual breakdown of when each driver pitted and on which compound
Sector performance — average S1/S2/S3 times per driver, grouped bar chart
Fastest lap comparison — who set the quickest lap and by how much
Telemetry speed trace — pick any driver, see their fastest lap speed through every corner
Driver vs Driver mode — head-to-head on lap times, position, tyre strategy, sector times, telemetry overlay, and a summary card
Weather conditions — air vs track temperature across the race
Live leaderboard — final standings with tyre info and overtake probability


🔮 Predict Upcoming Race Tab

Before each race weekend, the app predicts the full finishing order for all 22 drivers.


Uses 2019–2025 historical race results to train a Gradient Boosting model
Features include: grid position, last-5-race finishing average, team form, circuit history, DNF rate, season points
If qualifying has already happened, it uses the real grid; otherwise it estimates grid from recent qualifying pace
Shows a ranked finishing table, bar chart, and podium prediction cards



Tech Stack

ToolWhat it doesFastF1Downloads official F1 timing, telemetry, and results datascikit-learnGradient Boosting models for overtake probability + race predictionStreamlitThe entire frontend and deploymentPlotlyAll the interactive chartsPandas / NumPyData processing and feature engineeringJoblibModel serialization (.pkl files)
