import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import sqlite3
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hybrid import HybridPredictor
from models.tcn_model import load_tcn
from models.embeddings import AppEmbeddings
from models.association_rules import AssociationRuleMiner
from inference.multi_horizon import predict_multi_horizon, get_horizon_label

# -------------------------
# Config
# -------------------------
DB_PATH = "usage.db"   # Change to "usage.db" for real data
MODELS_DIR = Path("outputs/models")
REFRESH_MS = 5000                # 5 second refresh

# -------------------------
# Load Models (once at startup)
# -------------------------
print("Loading models...")
tcn_model, vocab = load_tcn(MODELS_DIR / "tcn_model.pt")
inv_vocab = {i: app for app, i in vocab.items()}

emb_model = AppEmbeddings(vocab)
emb_model.load(MODELS_DIR / "app_embeddings.pkl")

rule_miner = AssociationRuleMiner()
rule_miner.load(MODELS_DIR / "association_rules.pkl")

hybrid = HybridPredictor(tcn_model, vocab, emb_model, rule_miner)
print(f"✓ Models loaded. Vocab: {list(vocab.keys())}")
print(f"✓ Dashboard starting on http://127.0.0.1:8050")

# -------------------------
# Data helpers
# -------------------------
def get_recent_window(conn):
    """Build 24-bucket usage window from most recent data in DB"""
    cur = conn.cursor()
    cur.execute("SELECT MAX(timestamp) FROM app_events")
    row = cur.fetchone()
    max_ts = row[0] if row and row[0] else None

    if not max_ts:
        return np.zeros((24, len(vocab)))

    window_start = max_ts - (24 * 1800)  # 12 hours before latest event

    cur.execute("""
        SELECT timestamp, app_id FROM app_events
        WHERE timestamp >= ? AND timestamp <= ?
    """, (window_start, max_ts))

    window = np.zeros((24, len(vocab)))
    for ts, app in cur.fetchall():
        bucket_offset = (ts - window_start) // 1800
        if 0 <= bucket_offset < 24 and app in vocab:
            window[bucket_offset, vocab[app]] += 1

    return window


def get_current_app(conn):
    """Most recent app from events"""
    cur = conn.cursor()
    cur.execute("SELECT app_id FROM app_events ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else "—"


def get_recent_apps(conn, n=5):
    """Last N unique apps in order"""
    cur = conn.cursor()
    cur.execute("SELECT app_id FROM app_events ORDER BY timestamp DESC LIMIT 30")
    seen = []
    for row in cur.fetchall():
        if row[0] not in seen:
            seen.append(row[0])
        if len(seen) >= n:
            break
    return seen


def get_recent_events(conn, n=8):
    """Last N events with timestamps"""
    cur = conn.cursor()
    cur.execute("SELECT timestamp, app_id FROM app_events ORDER BY timestamp DESC LIMIT ?", (n,))
    return cur.fetchall()


# -------------------------
# App Layout
# -------------------------
app = dash.Dash(__name__, title="App Usage Predictor")

# Shared styles
CARD_STYLE = {
    "background": "#fff",
    "borderRadius": "12px",
    "padding": "22px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    "marginBottom": "16px"
}

HEADER_STYLE = {
    "color": "#555",
    "fontSize": "12px",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "marginBottom": "12px",
    "fontWeight": "600"
}

app.layout = html.Div([
    # Top bar
    html.Div([
        html.Div([
            html.H1("⚡ App Usage Predictor",
                     style={"margin": "0", "fontSize": "22px", "fontWeight": "700", "color": "#fff"}),
            html.Span(f"Hybrid ML · {len(vocab)} apps · Refreshes every {REFRESH_MS//1000}s",
                      style={"color": "#aaa", "fontSize": "13px"})
        ]),
        html.Div(id="last-updated", style={"color": "#666", "fontSize": "12px", "textAlign": "right"})
    ], style={
        "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
        "padding": "16px 28px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center"
    }),

    # Body
    html.Div([
        # Row 1: Current app + Live prediction
        html.Div([
            # Current app
            html.Div([
                html.Div("Currently Active", style=HEADER_STYLE),
                html.Div(id="current-app-display", style={
                    "fontSize": "32px",
                    "fontWeight": "700",
                    "color": "#636EFA"
                }),
                html.Div(id="recent-trail", style={
                    "marginTop": "10px",
                    "fontSize": "12px",
                    "color": "#999"
                })
            ], style={**CARD_STYLE, "flex": "0 0 240px"}),

            # Live prediction
            html.Div([
                html.Div("Next App · Top 5", style=HEADER_STYLE),
                dcc.Graph(id="live-prediction-chart",
                          style={"height": "180px"},
                          config={"displayModeBar": False})
            ], style={**CARD_STYLE, "flex": "1", "marginLeft": "16px"})
        ], style={"display": "flex"}),

        # Row 2: Multi-horizon
        html.Div([
            html.Div("Multi-Horizon Predictions", style=HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Span("30 min", style={"color": "#636EFA", "fontWeight": "600"}),
                    html.Span(" · next bucket", style={"color": "#999", "fontSize": "12px"})
                ], style={"display": "inline-block", "marginRight": "20px"}),
                html.Div([
                    html.Span("1.5 h", style={"color": "#EF553B", "fontWeight": "600"}),
                    html.Span(" · 3 buckets", style={"color": "#999", "fontSize": "12px"})
                ], style={"display": "inline-block", "marginRight": "20px"}),
                html.Div([
                    html.Span("3 h", style={"color": "#00CC96", "fontWeight": "600"}),
                    html.Span(" · 6 buckets", style={"color": "#999", "fontSize": "12px"})
                ], style={"display": "inline-block"})
            ], style={"marginBottom": "8px"}),
            dcc.Graph(id="multi-horizon-chart",
                      style={"height": "230px"},
                      config={"displayModeBar": False})
        ], style=CARD_STYLE),

        # Row 3: Prediction accuracy + Recent activity (side by side)
        html.Div([
            # Accuracy card
            html.Div([
                html.Div("Prediction Accuracy", style=HEADER_STYLE),
                html.Div(id="accuracy-display", style={"fontSize": "28px", "fontWeight": "700", "color": "#00CC96"}),
                html.Div(id="accuracy-detail", style={"fontSize": "12px", "color": "#999", "marginTop": "4px"}),
                dcc.Graph(id="accuracy-history-chart",
                          style={"height": "120px", "marginTop": "10px"},
                          config={"displayModeBar": False})
            ], style={**CARD_STYLE, "flex": "0 0 320px", "marginBottom": "0"}),

            # Recent activity
            html.Div([
                html.Div("Recent Activity", style=HEADER_STYLE),
                html.Div(id="activity-timeline")
            ], style={**CARD_STYLE, "flex": "1", "marginLeft": "16px", "marginBottom": "0"})
        ], style={"display": "flex"}),

    ], style={"padding": "20px", "maxWidth": "1100px", "margin": "0 auto", "background": "#f4f5f7", "minHeight": "calc(100vh - 70px)"}),

    # Auto-refresh
    dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0)
], style={"fontFamily": "'Segoe UI', system-ui, sans-serif", "margin": 0, "background": "#f4f5f7"})


# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("last-updated", "children"),
    Output("current-app-display", "children"),
    Output("recent-trail", "children"),
    Output("live-prediction-chart", "figure"),
    Output("multi-horizon-chart", "figure"),
    Output("accuracy-display", "children"),
    Output("accuracy-detail", "children"),
    Output("accuracy-history-chart", "figure"),
    Output("activity-timeline", "children"),
    Input("interval", "n_intervals")
)
def update_all(n):
    conn = sqlite3.connect(DB_PATH)

    # --- Data ---
    current_app = get_current_app(conn)
    recent_apps = get_recent_apps(conn, 5)
    recent_window = get_recent_window(conn)
    recent_events = get_recent_events(conn, 8)

    context = {
        'recent_window': recent_window,
        'recent_apps': recent_apps,
        'timestamp': int(time.time()),
        'usage_stats': {}
    }

    # --- 1. Live prediction (top 5, horizontal bar) ---
    predictions = hybrid.predict(context, top_k=5)
    pred_apps  = [p[0] for p in predictions]
    pred_scores = [p[1] for p in predictions]

    # Log prediction to DB
    import json
    cur = conn.cursor()
    bucket_id = int(time.time()) // 1800
    cur.execute("""
        INSERT INTO model_predictions (timestamp, bucket_id, model_name, predicted_apps, scores)
        VALUES (?, ?, ?, ?, ?)
    """, (
        int(time.time()),
        bucket_id,
        'hybrid',
        json.dumps(pred_apps),
        json.dumps(pred_scores)
    ))

    # Backfill actual: find most recent prediction with no actual_app, set it
    if current_app and current_app != "—":
        cur.execute("""
            UPDATE model_predictions
            SET actual_app = ?
            WHERE id = (
                SELECT id FROM model_predictions
                WHERE actual_app IS NULL
                  AND id != (SELECT MAX(id) FROM model_predictions)
                ORDER BY timestamp DESC
                LIMIT 1
            )
        """, (current_app,))

    conn.commit()
    bar_colors = ['#636EFA'] + ['#c5c8ff'] * (len(pred_apps) - 1)

    fig_live = go.Figure(go.Bar(
        x=pred_scores,
        y=pred_apps,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{s:.2f}" for s in pred_scores],
        textposition='outside',
        textfont=dict(size=12, color="#444")
    ))
    fig_live.update_layout(
        margin=dict(l=90, r=50, t=5, b=5),
        xaxis=dict(range=[0, 1.2], showgrid=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=13, color="#333")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25
    )

    # --- 2. Multi-horizon (grouped bar, top 3 per horizon) ---
    multi = predict_multi_horizon(tcn_model, recent_window, vocab, horizons=[1, 3, 6])

    horizon_labels = {1: "30 min", 3: "1.5 h", 6: "3 h"}
    horizon_colors = {1: "#636EFA", 3: "#EF553B", 6: "#00CC96"}

    # Collect all apps that appear in any horizon's top 3
    all_apps = []
    for h in [1, 3, 6]:
        for app, _ in multi[h][:3]:
            if app not in all_apps:
                all_apps.append(app)

    fig_multi = go.Figure()
    for h in [1, 3, 6]:
        top3_dict = {app: score for app, score in multi[h][:3]}
        fig_multi.add_trace(go.Bar(
            name=horizon_labels[h],
            x=all_apps,
            y=[top3_dict.get(app, 0) for app in all_apps],
            marker_color=horizon_colors[h],
            marker_line=dict(width=0)
        ))

    fig_multi.update_layout(
        barmode='group',
        margin=dict(l=20, r=20, t=5, b=25),
        yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#eee", showline=False, tickformat=".1f"),
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=13, color="#333")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,  # Legend handled manually above
        bargap=0.2,
        bargroupgap=0.05
    )

    # --- 3. Activity timeline ---
    app_colors = {
        'chrome': '#4285F4', 'vscode': '#007ACC', 'terminal': '#2ecc71',
        'slack': '#611f69', 'spotify': '#1DB954', 'discord': '#5865F2',
        'git': '#F1430A'
    }

    timeline_items = []
    for ts, app in recent_events:
        t_str = time.strftime("%H:%M", time.localtime(ts))
        color = app_colors.get(app.split(":")[0], "#636EFA")

        timeline_items.append(html.Div([
            html.Span(t_str, style={"color": "#999", "fontSize": "12px", "width": "50px", "display": "inline-block"}),
            html.Span("●", style={"color": color, "marginRight": "8px", "fontSize": "10px"}),
            html.Span(app, style={"color": "#333", "fontSize": "14px", "fontWeight": "500"})
        ], style={"padding": "5px 0", "borderBottom": "1px solid #f0f0f0"}))

    # --- Outputs ---
    # --- 4. Prediction accuracy from logged predictions ---
    cur = conn.cursor()
    cur.execute("""
        SELECT predicted_apps, actual_app FROM model_predictions
        WHERE actual_app IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 50
    """)
    logged = cur.fetchall()

    hit1_list = []
    hit3_list = []
    for pred_json, actual in logged:
        preds = json.loads(pred_json)
        hit1_list.append(1 if (len(preds) > 0 and preds[0] == actual) else 0)
        hit3_list.append(1 if actual in preds[:3] else 0)

    if hit1_list:
        hit1_pct = f"{np.mean(hit1_list) * 100:.0f}%"
        hit3_pct = f"{np.mean(hit3_list) * 100:.0f}%"
        accuracy_detail = f"Hit@1: {hit1_pct}  ·  Hit@3: {hit3_pct}  ·  {len(hit1_list)} predictions logged"

        # Rolling accuracy (window of 10)
        window = 10
        rolling_hit3 = [np.mean(hit3_list[max(0,i-window):i+1]) for i in range(len(hit3_list))]
        rolling_hit3.reverse()  # oldest first

        fig_acc = go.Figure(go.Scatter(
            x=list(range(len(rolling_hit3))),
            y=[v * 100 for v in rolling_hit3],
            mode='lines',
            line=dict(color='#00CC96', width=2),
            fill='tonexty',
            fillcolor='rgba(0,204,150,0.1)'
        ))
        fig_acc.update_layout(
            margin=dict(l=30, r=10, t=0, b=15),
            yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#eee", showline=False, ticksuffix="%", tickfont=dict(size=10)),
            xaxis=dict(showgrid=False, showline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    else:
        hit1_pct = "—"
        accuracy_detail = "No predictions logged yet"
        fig_acc = go.Figure()
        fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    timestamp_str = time.strftime("%H:%M:%S", time.localtime())
    trail_str = " → ".join(recent_apps[:4]) if recent_apps else "—"

    conn.close()

    return (
        f"Updated {timestamp_str}",
        current_app,
        f"Recent: {trail_str}",
        fig_live,
        fig_multi,
        hit1_pct,
        accuracy_detail,
        fig_acc,
        timeline_items
    )


if __name__ == "__main__":
    app.run(debug=False, port=8050)


