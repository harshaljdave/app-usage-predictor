import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# -------------------------------
# 1. Model Comparison Bar Chart
# -------------------------------
def plot_model_comparison(results, save_path=None):
    """Bar chart comparing all model variants
    
    results: dict from ablation.py
    {model_name: {hit@1, hit@3, hit@5, mrr}}
    """
    models = list(results.keys())
    hit1 = [results[m]['hit@1'] for m in models]
    hit3 = [results[m]['hit@3'] for m in models]
    hit5 = [results[m]['hit@5'] for m in models]
    mrr  = [results[m]['mrr'] for m in models]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hit@K Accuracy", "Mean Reciprocal Rank"))
    
    # Hit@K grouped bar
    fig.add_trace(go.Bar(name='Hit@1', x=models, y=hit1, marker_color='#636EFA'), row=1, col=1)
    fig.add_trace(go.Bar(name='Hit@3', x=models, y=hit3, marker_color='#EF553B'), row=1, col=1)
    fig.add_trace(go.Bar(name='Hit@5', x=models, y=hit5, marker_color='#00CC96'), row=1, col=1)
    
    # MRR bar
    fig.add_trace(go.Bar(name='MRR', x=models, y=mrr, marker_color='#AB63FA', showlegend=True), row=1, col=2)
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        barmode='group',
        height=500,
        width=1100,
        yaxis_title="Accuracy",
        yaxis2_title="MRR Score",
        yaxis_range=[0, 1.1],
        yaxis2_range=[0, 1.1],
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    return fig


# -------------------------------
# 2. Training Curves
# -------------------------------
def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot TCN training and validation loss
    
    train_losses: list of train losses per epoch
    val_losses: list of val losses per epoch
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        marker=dict(size=6),
        line=dict(color='#636EFA', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        marker=dict(size=6),
        line=dict(color='#EF553B', width=2)
    ))
    
    # Mark best val loss
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    
    fig.add_trace(go.Scatter(
        x=[best_epoch], y=[best_loss],
        mode='markers',
        name=f'Best Val Loss (epoch {best_epoch})',
        marker=dict(size=14, color='#00CC96', symbol='star')
    ))
    
    fig.update_layout(
        title="TCN Training Curves",
        xaxis_title="Epoch",
        yaxis_title="BCE Loss",
        template='plotly_white',
        height=400,
        width=800
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    return fig


# -------------------------------
# 3. Embedding Visualization (t-SNE)
# -------------------------------
def plot_embeddings(emb_model, save_path=None):
    """2D t-SNE visualization of app embeddings"""
    from sklearn.manifold import TSNE
    
    apps = list(emb_model.vocab.keys())
    vectors = np.array([emb_model.W_in[emb_model.vocab[app]] for app in apps])
    
    # t-SNE to 2D
    if len(apps) > 2:
        perplexity = min(5, len(apps) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(vectors)
    else:
        coords = vectors[:, :2]
    
    # Color by category
    categories = {
        'development': ['vscode', 'terminal', 'git'],
        'communication': ['slack', 'discord', 'chrome:work:gmail'],
        'media': ['spotify', 'chrome:personal:youtube'],
        'browser': ['chrome', 'firefox']
    }
    
    color_map = {}
    for cat, app_list in categories.items():
        for app in app_list:
            color_map[app] = cat
    
    colors = [color_map.get(app, 'other') for app in apps]
    
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        text=apps,
        color=colors,
        title="App Embeddings (t-SNE 2D)",
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Category'},
        template='plotly_white',
        height=500,
        width=700
    )
    
    fig.update_traces(textposition='top center', marker_size=12)
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    return fig


# -------------------------------
# 4. Usage Heatmap
# -------------------------------
def plot_usage_heatmap(conn, save_path=None):
    """Hour-of-day × Day-of-week usage heatmap"""
    cur = conn.cursor()
    cur.execute("SELECT timestamp, app_id FROM app_events")
    events = cur.fetchall()
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = [f"{h:02d}:00" for h in range(24)]
    
    # Build heatmap matrix
    matrix = np.zeros((7, 24))
    for ts, app in events:
        t = time.localtime(ts)
        matrix[t.tm_wday][t.tm_hour] += 1
    
    fig = px.imshow(
        matrix,
        x=hours,
        y=days,
        color_continuous_scale='YlOrRd',
        title="App Usage Heatmap (Hour × Day)",
        labels=dict(x="Hour", y="Day", color="Events"),
        template='plotly_white',
        height=400,
        width=900
    )
    
    fig.update_xaxes(tickangle=45)
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    return fig


# -------------------------------
# 5. App Distribution
# -------------------------------
def plot_app_distribution(conn, save_path=None):
    """Pie + bar chart of app usage distribution"""
    cur = conn.cursor()
    cur.execute("""
        SELECT app_id, COUNT(*) as cnt 
        FROM app_events 
        GROUP BY app_id 
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    
    apps = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Usage Distribution", "Event Counts"),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Pie chart
    fig.add_trace(go.Pie(
        labels=apps,
        values=counts,
        hole=0.3
    ), row=1, col=1)
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=apps,
        y=counts,
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
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    return fig


# -------------------------------
# 6. Association Rules Network
# -------------------------------
def plot_rules_summary(rules, save_path=None):
    """Bar chart of top association rules by confidence"""
    if not rules:
        print("⚠️  No rules to plot")
        return None
    
    # Sort by confidence
    sorted_rules = sorted(rules, key=lambda x: -x['confidence'])[:15]
    
    labels = [f"{{{', '.join(r['lhs'])}}} → {r['rhs']}" for r in sorted_rules]
    confidences = [r['confidence'] for r in sorted_rules]
    supports = [r['support'] for r in sorted_rules]
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Confidence", "Support"))
    
    fig.add_trace(go.Bar(
        x=labels, y=confidences,
        marker_color='#EF553B',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=labels, y=supports,
        marker_color='#00CC96',
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        title="Top Association Rules",
        height=600,
        width=1000,
        template='plotly_white',
        yaxis_title="Confidence",
        yaxis2_title="Support Count"
    )
    
    fig.update_xaxes(tickangle=45)
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    
    return fig


# -------------------------------
# Main: Generate all visualizations
# -------------------------------
if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="usage_synthetic.db")
    parser.add_argument("--models", default="outputs/models")
    parser.add_argument("--output", default="outputs/figures")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models)
    
    print("=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    # 1. Model comparison (hardcoded from ablation results)
    print("\n[1/6] Model comparison...")
    results = {
        'baseline':          {'hit@1': 0.277, 'hit@3': 0.755, 'hit@5': 0.979, 'mrr': 0.529},
        'hybrid_equal':      {'hit@1': 0.372, 'hit@3': 0.819, 'hit@5': 0.979, 'mrr': 0.597},
        'hybrid_tcn_heavy':  {'hit@1': 0.372, 'hit@3': 0.819, 'hit@5': 0.979, 'mrr': 0.597},
        'hybrid_optimized':  {'hit@1': 0.372, 'hit@3': 0.819, 'hit@5': 0.979, 'mrr': 0.597},
    }
    plot_model_comparison(results, output_dir / "model_comparison.html")
    
    # 2. Training curves (from TCN output)
    print("\n[2/6] Training curves...")
    train_losses = [0.6315, 0.5157, 0.4725, 0.4491, 0.4304, 0.4137, 0.3978, 0.3827,
                    0.3690, 0.3612, 0.3535, 0.3458, 0.3387, 0.3348, 0.3308, 0.3268,
                    0.3231, 0.3211, 0.3191, 0.3171]
    val_losses   = [0.5633, 0.5160, 0.5014, 0.4990, 0.5006, 0.5039, 0.5091, 0.5132,
                    0.5156, 0.5185, 0.5216, 0.5246, 0.5265, 0.5282, 0.5298, 0.5318,
                    0.5329, 0.5339, 0.5348, 0.5358]
    plot_training_curves(train_losses, val_losses, output_dir / "training_curves.html")
    
    # 3. Embeddings
    print("\n[3/6] Embedding visualization...")
    from models.embeddings import AppEmbeddings
    from data_processing.preprocessing import build_vocab
    
    conn = sqlite3.connect(args.db)
    vocab = build_vocab(conn, min_count=10)
    emb = AppEmbeddings(vocab)
    emb.load(models_dir / "app_embeddings.pkl")
    plot_embeddings(emb, output_dir / "embeddings_tsne.html")
    
    # 4. Usage heatmap
    print("\n[4/6] Usage heatmap...")
    plot_usage_heatmap(conn, output_dir / "usage_heatmap.html")
    
    # 5. App distribution
    print("\n[5/6] App distribution...")
    plot_app_distribution(conn, output_dir / "app_distribution.html")
    
    # 6. Association rules
    print("\n[6/6] Association rules...")
    from models.association_rules import AssociationRuleMiner
    miner = AssociationRuleMiner()
    miner.load(models_dir / "association_rules.pkl")
    plot_rules_summary(miner.rules, output_dir / "association_rules.html")
    
    conn.close()
    
    print("\n" + "=" * 50)
    print(f"✓ All visualizations saved to {output_dir}")
    print("=" * 50)