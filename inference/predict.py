#!/usr/bin/env python3
"""
Standalone prediction CLI - test models without running dashboard
"""
import argparse
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


def get_recent_window(conn, window_size=24):
    """Build recent usage window from DB"""
    cur = conn.cursor()
    cur.execute("SELECT MAX(timestamp) FROM app_events")
    row = cur.fetchone()
    max_ts = row[0] if row and row[0] else None

    if not max_ts:
        print("⚠️  No events in database")
        return None, None

    window_start = max_ts - (window_size * 1800)
    cur.execute("""
        SELECT timestamp, app_id FROM app_events
        WHERE timestamp >= ? AND timestamp <= ?
    """, (window_start, max_ts))

    from data_processing.preprocessing import build_vocab
    vocab = build_vocab(conn, min_count=3)
    
    window = np.zeros((window_size, len(vocab)))
    for ts, app in cur.fetchall():
        bucket_offset = (ts - window_start) // 1800
        if 0 <= bucket_offset < window_size and app in vocab:
            window[bucket_offset, vocab[app]] += 1

    return window, vocab


def get_recent_apps(conn, n=5):
    cur = conn.cursor()
    cur.execute("SELECT app_id FROM app_events ORDER BY timestamp DESC LIMIT 30")
    seen = []
    for row in cur.fetchall():
        if row[0] not in seen:
            seen.append(row[0])
        if len(seen) >= n:
            break
    return seen


def main():
    parser = argparse.ArgumentParser(description="Predict next app usage")
    parser.add_argument("--db", default="usage.db", help="Database path")
    parser.add_argument("--models", default="outputs/models", help="Models directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions")
    parser.add_argument("--multi-horizon", action="store_true", help="Show multi-horizon predictions")
    args = parser.parse_args()

    models_dir = Path(args.models)
    
    # Load models
    print("Loading models...")
    tcn_model, vocab = load_tcn(models_dir / "tcn_model.pt")
    emb_model = AppEmbeddings(vocab)
    emb_model.load(models_dir / "app_embeddings.pkl")
    rule_miner = AssociationRuleMiner()
    rule_miner.load(models_dir / "association_rules.pkl")
    hybrid = HybridPredictor(tcn_model, vocab, emb_model, rule_miner)
    print(f"✓ Models loaded ({len(vocab)} apps)\n")

    # Load data
    conn = sqlite3.connect(args.db)
    recent_window, db_vocab = get_recent_window(conn)
    
    if recent_window is None:
        conn.close()
        return

    recent_apps = get_recent_apps(conn)
    
    # Build context
    context = {
        'recent_window': recent_window,
        'recent_apps': recent_apps,
        'timestamp': int(time.time()),
        'usage_stats': {}
    }

    # Predict
    predictions = hybrid.predict(context, top_k=args.top_k)
    
    print(f"📍 Context: Recent apps = {', '.join(recent_apps[:3])}")
    print(f"\n🔮 Top-{args.top_k} Predictions:\n")
    
    for i, (app, score) in enumerate(predictions, 1):
        bar = "█" * int(score * 20)
        print(f"  {i}. {app:<25} {score:.3f}  {bar}")

    # Multi-horizon
    if args.multi_horizon:
        print(f"\n⏱️  Multi-Horizon Predictions:\n")
        multi = predict_multi_horizon(tcn_model, recent_window, vocab, horizons=[1, 3, 6])
        
        for step, label in [(1, "30 min"), (3, "1.5 h"), (6, "3 h")]:
            print(f"  {label}:")
            for app, score in multi[step][:3]:
                print(f"    • {app:<20} {score:.3f}")
            print()

    conn.close()


if __name__ == "__main__":
    main()