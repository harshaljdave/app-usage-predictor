# Save this as a .py file first, then we convert. 
# Create notebooks/01_data_exploration.py

import sqlite3
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# CELL 1: Setup & Load Data
# ============================================================
DB_PATH = "../usage_synthetic.db"  # Change to "usage.db" for real data

conn = sqlite3.connect(DB_PATH)

# Load all events
df = pd.read_sql_query("""
    SELECT 
        id,
        timestamp,
        app_id,
        event_type,
        datetime(timestamp, 'unixepoch', 'localtime') as readable_time
    FROM app_events
    ORDER BY timestamp
""", conn)

print(f"Total events: {len(df)}")
print(f"Date range: {df['readable_time'].iloc[0]} → {df['readable_time'].iloc[-1]}")
print(f"Unique apps: {df['app_id'].nunique()}")
df.head(10)


# ============================================================
# CELL 2: App Usage Distribution
# ============================================================
app_counts = df['app_id'].value_counts()

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Usage Distribution", "Event Counts"),
    specs=[[{"type": "pie"}, {"type": "bar"}]]
)

fig.add_trace(go.Pie(
    labels=app_counts.index,
    values=app_counts.values,
    hole=0.3
), row=1, col=1)

fig.add_trace(go.Bar(
    x=app_counts.index,
    y=app_counts.values,
    marker_color='#636EFA',
    showlegend=False
), row=1, col=2)

fig.update_layout(
    title="App Usage Distribution",
    height=450,
    width=1100,
    template='plotly_white'
)
fig.update_yaxes(title_text="Event Count", row=1, col=2)
fig.show()


# ============================================================
# CELL 3: Temporal Patterns
# ============================================================
df['hour'] = df['timestamp'].apply(lambda ts: time.localtime(ts).tm_hour)
df['day'] = df['timestamp'].apply(lambda ts: time.localtime(ts).tm_wday)
df['day_name'] = df['day'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})

# Events per hour
hourly = df.groupby('hour').size().reset_index(name='count')

fig = px.bar(
    hourly, x='hour', y='count',
    title="Events by Hour of Day",
    labels={'hour': 'Hour', 'count': 'Event Count'},
    color_discrete_sequence=['#636EFA'],
    template='plotly_white'
)
fig.update_xaxes(range=[-0.5, 23.5], dtick=1)
fig.show()


# ============================================================
# CELL 4: Usage Heatmap (Hour × Day)
# ============================================================
days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
pivot = df.pivot_table(
    index='day_name', columns='hour',
    aggfunc='size', fill_value=0
)
pivot = pivot.reindex(days_order)

fig = px.imshow(
    pivot.values,
    x=[f"{h:02d}:00" for h in range(24)],
    y=days_order,
    color_continuous_scale='YlOrRd',
    title="Usage Heatmap (Hour × Day)",
    labels=dict(x="Hour", y="Day", color="Events"),
    template='plotly_white',
    height=400, width=900
)
fig.update_xaxes(tickangle=45)
fig.show()


# ============================================================
# CELL 5: App-Specific Patterns
# ============================================================
app_hourly = df.groupby(['app_id', 'hour']).size().reset_index(name='count')

fig = px.line(
    app_hourly, x='hour', y='count', color='app_id',
    title="App Usage by Hour",
    labels={'hour': 'Hour', 'count': 'Events', 'app_id': 'App'},
    template='plotly_white',
    height=450, width=900
)
fig.update_xaxes(range=[-0.5, 23.5], dtick=1)
fig.show()


# ============================================================
# CELL 6: Session Analysis
# ============================================================
from data_processing.preprocessing import extract_sessions

extract_sessions(conn)

sessions_df = pd.read_sql_query("""
    SELECT 
        session_id,
        start_time,
        end_time,
        apps,
        (end_time - start_time) as duration_secs,
        LENGTH(apps) - LENGTH(REPLACE(apps, ',', '')) + 1 as num_apps
    FROM app_sessions
""", conn)

sessions_df['duration_mins'] = sessions_df['duration_secs'] / 60

print(f"Total sessions: {len(sessions_df)}")
print(f"Avg session length: {sessions_df['duration_mins'].mean():.1f} minutes")
print(f"Avg apps per session: {sessions_df['num_apps'].mean():.1f}")

fig = make_subplots(rows=1, cols=2, subplot_titles=("Session Duration", "Apps per Session"))

fig.add_trace(go.Histogram(
    x=sessions_df['duration_mins'],
    marker_color='#636EFA',
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Histogram(
    x=sessions_df['num_apps'],
    marker_color='#EF553B',
    showlegend=False
), row=1, col=2)

fig.update_layout(
    title="Session Statistics",
    height=400, width=1000,
    template='plotly_white'
)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_xaxes(title_text="Duration (minutes)", row=1, col=1)
fig.update_xaxes(title_text="Number of Apps", row=1, col=2)
fig.show()


# ============================================================
# CELL 7: Transition Matrix
# ============================================================
transitions = {}
for _, row in sessions_df.iterrows():
    apps = row['apps'].split(',')
    for i in range(len(apps) - 1):
        key = (apps[i], apps[i+1])
        transitions[key] = transitions.get(key, 0) + 1

# Build matrix
all_apps = sorted(df['app_id'].unique())
matrix = pd.DataFrame(0, index=all_apps, columns=all_apps)

for (src, dst), count in transitions.items():
    if src in matrix.index and dst in matrix.columns:
        matrix.loc[src, dst] = count

# Normalize by row (transition probabilities)
row_sums = matrix.sum(axis=1)
matrix_norm = matrix.div(row_sums, axis=0).fillna(0)

fig = px.imshow(
    matrix_norm.values,
    x=all_apps,
    y=all_apps,
    color_continuous_scale='Blues',
    title="App Transition Probabilities",
    labels=dict(x="Next App", y="Current App", color="P(next|current)"),
    template='plotly_white',
    height=500, width=600
)
fig.update_traces(text=matrix_norm.values.round(2), texttemplate="%{text}")
fig.show()


# ============================================================
# CELL 8: Summary Stats
# ============================================================
print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print(f"Total events:          {len(df)}")
print(f"Unique apps:           {df['app_id'].nunique()}")
print(f"Total sessions:        {len(sessions_df)}")
print(f"Avg session duration:  {sessions_df['duration_mins'].mean():.1f} min")
print(f"Avg apps/session:      {sessions_df['num_apps'].mean():.1f}")
print(f"\nApp frequencies:")
for app, count in app_counts.items():
    print(f"  {app:<20} {count:>5} ({count/len(df)*100:.1f}%)")

conn.close()