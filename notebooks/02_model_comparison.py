# notebooks/02_model_comparison.py

import sqlite3
import numpy as np
import torch
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import FrequencyBaseline
from models.tcn_model import load_tcn
from models.embeddings import AppEmbeddings
from models.association_rules import AssociationRuleMiner
from models.hybrid import HybridPredictor
from evaluation.metrics import EvaluationMetrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CELL 1: Setup & Load Models
# ============================================================
DB_PATH = "../usage_synthetic.db"
MODELS_DIR = Path("../outputs/models")

conn = sqlite3.connect(DB_PATH)

# Load all models
baseline = FrequencyBaseline()
baseline.load(MODELS_DIR / "baseline_model.pkl")

tcn_model, vocab = load_tcn(MODELS_DIR / "tcn_model.pt")
inv_vocab = {i: app for app, i in vocab.items()}

emb_model = AppEmbeddings(vocab)
emb_model.load(MODELS_DIR / "app_embeddings.pkl")

rule_miner = AssociationRuleMiner()
rule_miner.load(MODELS_DIR / "association_rules.pkl")

# Create hybrid variants
hybrid_equal = HybridPredictor(tcn_model, vocab, emb_model, rule_miner,
                               weights={'tcn': 0.33, 'emb': 0.33, 'rules': 0.34})
hybrid_optimized = HybridPredictor(tcn_model, vocab, emb_model, rule_miner,
                                   weights={'tcn': 0.5, 'emb': 0.3, 'rules': 0.2})

print("✓ All models loaded")
print(f"Vocabulary: {list(vocab.keys())}")


# ============================================================
# CELL 2: TCN Prediction Breakdown
# ============================================================
# Show what TCN sees for different time windows

from data_processing.preprocessing import aggregate_to_buckets
from data_processing.feature_engineering import build_tcn_dataset

buckets = aggregate_to_buckets(conn)
X, Y = build_tcn_dataset(buckets, vocab, window=24)

# Get TCN predictions for 5 random test samples
np.random.seed(42)
sample_indices = np.random.choice(range(len(X) - 100, len(X)), size=5, replace=False)

fig = make_subplots(rows=5, cols=1, subplot_titles=[f"Sample {i+1}" for i in range(5)])

for plot_idx, idx in enumerate(sample_indices):
    # Get prediction
    with torch.no_grad():
        pred = tcn_model(torch.FloatTensor(X[idx:idx+1])).squeeze().numpy()
    
    apps = list(vocab.keys())
    
    fig.add_trace(go.Bar(
        x=apps, y=pred,
        marker_color='#636EFA',
        showlegend=False,
        text=[f"{p:.2f}" for p in pred],
        textposition='outside'
    ), row=plot_idx+1, col=1)

fig.update_layout(
    title="TCN Predictions (5 Samples)",
    height=900,
    width=800,
    template='plotly_white'
)
fig.update_yaxes(title_text="Score", range=[0, 1.2])
fig.show()


# ============================================================
# CELL 3: Embedding Similarity Matrix
# ============================================================
apps = list(vocab.keys())
sim_matrix = np.zeros((len(apps), len(apps)))

for i, app1 in enumerate(apps):
    for j, app2 in enumerate(apps):
        sim_matrix[i][j] = emb_model.similarity(app1, app2)

fig = go.Figure(data=go.Heatmap(
    z=sim_matrix,
    x=apps, y=apps,
    colorscale='RdBu_r',
    zmid=0,
    text=sim_matrix.round(2),
    texttemplate="%{text}",
    colorbar=dict(title="Cosine Similarity")
))

fig.update_layout(
    title="App Embedding Similarity Matrix",
    height=500, width=600,
    template='plotly_white'
)
fig.show()


# ============================================================
# CELL 4: Association Rules Breakdown
# ============================================================
# Group rules by consequent (target app)
rules_by_app = defaultdict(list)
for rule in rule_miner.rules:
    rules_by_app[rule['rhs']].append(rule)

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=list(rules_by_app.keys())[:6]
)

for idx, (app, rules) in enumerate(list(rules_by_app.items())[:6]):
    row = idx // 3 + 1
    col = idx % 3 + 1
    
    # Top 3 rules for this app
    top_rules = sorted(rules, key=lambda x: -x['confidence'])[:3]
    labels = [f"{{{', '.join(r['lhs'])}}}" for r in top_rules]
    confs = [r['confidence'] for r in top_rules]
    
    fig.add_trace(go.Bar(
        x=confs, y=labels,
        orientation='h',
        marker_color='#EF553B',
        showlegend=False,
        text=[f"{c:.2f}" for c in confs],
        textposition='outside'
    ), row=row, col=col)

fig.update_layout(
    title="Top Rules Per App (Confidence)",
    height=600, width=1100,
    template='plotly_white'
)
fig.update_xaxes(range=[0, 1.3], title_text="Confidence")
fig.show()


# ============================================================
# CELL 5: Side-by-Side Prediction Comparison
# ============================================================
# Compare predictions from each model for same contexts

test_contexts = [
    {'hour': 9, 'day': 0, 'last_app': 'chrome', 'label': 'Monday 9AM after Chrome'},
    {'hour': 14, 'day': 2, 'last_app': 'vscode', 'label': 'Wednesday 2PM after VSCode'},
    {'hour': 20, 'day': 4, 'last_app': 'terminal', 'label': 'Friday 8PM after Terminal'},
    {'hour': 11, 'day': 5, 'last_app': 'spotify', 'label': 'Saturday 11AM after Spotify'},
]

for ctx in test_contexts:
    print(f"\n📍 {ctx['label']}")
    print("-" * 55)
    
    # Baseline prediction
    baseline_preds = baseline.predict(ctx, top_k=3)
    print(f"  Baseline:  {[(app, f'{s:.2f}') for app, s in baseline_preds]}")
    
    # Hybrid prediction
    hybrid_ctx = {
        'recent_window': np.random.rand(24, len(vocab)),  # Simplified
        'recent_apps': [ctx['last_app']],
        'timestamp': int(time.time()),
        'usage_stats': {}
    }
    hybrid_preds = hybrid_optimized.predict(hybrid_ctx, top_k=3)
    print(f"  Hybrid:    {[(app, f'{s:.2f}') for app, s in hybrid_preds]}")


# ============================================================
# CELL 6: Model Score Contribution
# ============================================================
# Show how each model contributes to hybrid score

sample_apps = list(vocab.keys())
hybrid_ctx = {
    'recent_window': np.random.rand(24, len(vocab)),
    'recent_apps': ['vscode', 'terminal'],
    'timestamp': int(time.time()),
    'usage_stats': {}
}

# Get individual scores
with torch.no_grad():
    tcn_scores = tcn_model(torch.FloatTensor(hybrid_ctx['recent_window'][None, :, :])).squeeze().numpy()

tcn_dict = {inv_vocab[i]: tcn_scores[i] for i in range(len(tcn_scores))}
emb_scores = {app: emb_model.similarity_to_set(app, hybrid_ctx['recent_apps']) for app in vocab}
rule_scores = {app: rule_miner.get_rule_score(hybrid_ctx['recent_apps'], app) for app in vocab}

# Weighted contributions
w = {'tcn': 0.5, 'emb': 0.3, 'rules': 0.2}
tcn_contrib  = {app: w['tcn'] * tcn_dict[app] for app in vocab}
emb_contrib  = {app: w['emb'] * emb_scores[app] for app in vocab}
rule_contrib = {app: w['rules'] * rule_scores[app] for app in vocab}

apps = list(vocab.keys())

fig = go.Figure(data=[
    go.Bar(name='TCN (α=0.5)', x=apps, y=[tcn_contrib[a] for a in apps], marker_color='#636EFA'),
    go.Bar(name='Embedding (β=0.3)', x=apps, y=[emb_contrib[a] for a in apps], marker_color='#EF553B'),
    go.Bar(name='Rules (γ=0.2)', x=apps, y=[rule_contrib[a] for a in apps], marker_color='#00CC96'),
])

fig.update_layout(
    title="Hybrid Score Breakdown by Model Component<br><i>Context: after [vscode, terminal]</i>",
    barmode='stack',
    yaxis_title="Score Contribution",
    template='plotly_white',
    height=450, width=800
)
fig.show()


# ============================================================
# CELL 7: Ablation Results Summary
# ============================================================
results = {
    'Baseline':         {'Hit@1': 0.277, 'Hit@3': 0.755, 'Hit@5': 0.979, 'MRR': 0.529},
    'Hybrid (equal)':   {'Hit@1': 0.372, 'Hit@3': 0.819, 'Hit@5': 0.979, 'MRR': 0.597},
    'Hybrid (TCN-heavy)':{'Hit@1': 0.372, 'Hit@3': 0.819, 'Hit@5': 0.979, 'MRR': 0.597},
    'Hybrid (optimized)':{'Hit@1': 0.372, 'Hit@3': 0.819, 'Hit@5': 0.979, 'MRR': 0.597},
}

models = list(results.keys())
metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

fig = go.Figure()
for metric, color in zip(metrics, colors):
    fig.add_trace(go.Bar(
        name=metric,
        x=models,
        y=[results[m][metric] for m in models],
        marker_color=color
    ))

# Add baseline reference line
fig.add_shape(
    type="line", x0=-0.5, x1=3.5,
    y0=results['Baseline']['Hit@1'], y1=results['Baseline']['Hit@1'],
    line=dict(color="gray", width=1, dash="dash")
)

fig.update_layout(
    title="Ablation Study: Model Performance Comparison",
    barmode='group',
    yaxis_title="Score",
    yaxis_range=[0, 1.1],
    template='plotly_white',
    height=500, width=900
)
fig.show()

print("\n✓ All model comparisons complete")
conn.close()