import sqlite3
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.preprocessing import aggregate_to_buckets, extract_sessions, build_vocab
from data_processing.feature_engineering import build_tcn_dataset
from models.baseline import FrequencyBaseline
from models.tcn_model import TCN, train_tcn, save_tcn
from models.embeddings import AppEmbeddings
from models.association_rules import AssociationRuleMiner


def train_all_models(db_path, output_dir, min_app_count=10):
    """Train all models on dataset"""
    
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    conn = sqlite3.connect(db_path)
    
    # Step 1: Preprocessing
    print("\n[1/6] Preprocessing data...")
    buckets = aggregate_to_buckets(conn)
    extract_sessions(conn)
    vocab = build_vocab(conn, min_count=min_app_count)
    
    if len(vocab) == 0:
        print("❌ Empty vocabulary. Lower min_app_count or collect more data.")
        return
    
    # Step 2: Train baseline
    print("\n[2/6] Training baseline model...")
    baseline = FrequencyBaseline()
    baseline.train(conn)
    baseline.save(output_dir / "baseline_model.pkl")
    print("✓ Baseline saved")
    
    # Step 3: Prepare TCN data
    print("\n[3/6] Building TCN dataset...")
    X, Y = build_tcn_dataset(buckets, vocab, window=24)
    
    if len(X) < 50:
        print("⚠️  Very few samples for TCN. Results may be poor.")
    
    # Train/val split (80/20 temporal)
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Step 4: Train TCN
    print("\n[4/6] Training TCN...")
    tcn = TCN(num_apps=len(vocab))
    tcn = train_tcn(tcn, X_train, Y_train, X_val, Y_val, epochs=20)
    save_tcn(tcn, vocab, output_dir / "tcn_model.pt")
    print("✓ TCN saved")
    
    # Step 5: Train embeddings
    print("\n[5/6] Training embeddings...")
    cur = conn.cursor()
    cur.execute("SELECT apps FROM app_sessions")
    sessions = [row[0].split(',') for row in cur.fetchall()]
    
    emb = AppEmbeddings(vocab, dim=16, lr=0.01)
    for epoch in range(5):
        for session in sessions:
            emb.train_session(session, window=2)
        print(f"  Epoch {epoch+1}/5")
    
    emb.save(output_dir / "app_embeddings.pkl")
    print("✓ Embeddings saved")
    
    # Step 6: Mine rules
    print("\n[6/6] Mining association rules...")
    miner = AssociationRuleMiner(min_support=5, min_confidence=0.6)
    miner.train(sessions)
    miner.save(output_dir / "association_rules.pkl")
    print("✓ Rules saved")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total samples: {len(X)}")
    print(f"Sessions: {len(sessions)}")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--db", default="usage_synthetic.db", help="Database path")
    parser.add_argument("--output", default="outputs/models", help="Output directory")
    parser.add_argument("--min-count", type=int, default=10, help="Min app count")
    
    args = parser.parse_args()
    
    train_all_models(args.db, args.output, args.min_count)