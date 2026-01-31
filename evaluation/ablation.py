import sqlite3
import numpy as np
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.preprocessing import build_vocab
from data_processing.feature_engineering import build_tcn_dataset, RollingStats
from models.baseline import FrequencyBaseline
from models.tcn_model import load_tcn
from models.embeddings import AppEmbeddings
from models.association_rules import AssociationRuleMiner
from models.hybrid import HybridPredictor
from evaluation.metrics import EvaluationMetrics


def evaluate_model(model, model_name, test_data, vocab):
    """Evaluate a model on test data
    
    test_data: list of (context, actual_app) tuples
    """
    print(f"\nEvaluating {model_name}...")
    metrics = EvaluationMetrics()
    
    for context, actual in test_data:
        try:
            if model_name == 'baseline':
                predictions = model.predict(context, top_k=5)
            else:  # hybrid or variants
                predictions = model.predict(context, top_k=5)
            
            if actual in vocab:
                metrics.add_prediction(predictions, actual)
        except Exception as e:
            print(f"  Error on sample: {e}")
            continue
    
    results = metrics.summarize()
    print(f"  Evaluated on {metrics.count()} samples")
    
    return results


def prepare_test_data(conn, vocab, buckets, train_split=0.8):
    """Prepare test contexts and ground truth"""
    bucket_ids = sorted(buckets.keys())
    split_idx = int(train_split * len(bucket_ids))
    test_bucket_ids = bucket_ids[split_idx:]
    
    test_data = []
    rolling_stats = RollingStats()
    
    # Build rolling stats on train data
    for b_id in bucket_ids[:split_idx]:
        rolling_stats.update(buckets[b_id], b_id)
    
    # Create test samples
    for i, b_id in enumerate(test_bucket_ids[24:]):  # Need window history
        # Get recent window for TCN
        window_start = test_bucket_ids[i]
        recent_window = np.zeros((24, len(vocab)))
        
        for t in range(24):
            w_id = test_bucket_ids[i + t]
            for app, count in buckets.get(w_id, {}).items():
                if app in vocab:
                    recent_window[t, vocab[app]] = count
        
        # Get recent apps (last 5 unique)
        recent_apps = []
        for w_id in reversed(test_bucket_ids[i:i+24]):
            for app in buckets.get(w_id, {}).keys():
                if app not in recent_apps:
                    recent_apps.append(app)
                if len(recent_apps) >= 5:
                    break
            if len(recent_apps) >= 5:
                break
        
        # Build context
        context = {
            'recent_window': recent_window,
            'recent_apps': recent_apps,
            'timestamp': b_id * 1800,  # BUCKET_SIZE
            'usage_stats': {},
            'hour': time.localtime(b_id * 1800).tm_hour,
            'day': time.localtime(b_id * 1800).tm_wday,
            'last_app': recent_apps[0] if recent_apps else None
        }
        
        # Add usage stats
        for app in vocab:
            context['usage_stats'][app] = rolling_stats.get_stats(app, b_id)
        
        # Ground truth: most used app in next bucket
        next_bucket = buckets.get(test_bucket_ids[i + 24], {})
        if next_bucket:
            actual_app = max(next_bucket.items(), key=lambda x: x[1])[0]
            test_data.append((context, actual_app))
        
        # Update rolling stats
        rolling_stats.update(buckets[b_id], b_id)
    
    return test_data


def run_ablation_study(db_path, models_dir):
    """Compare all model variants"""
    
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    models_dir = Path(models_dir)
    
    # Load data
    from data_processing.preprocessing import aggregate_to_buckets
    buckets = aggregate_to_buckets(conn)
    vocab = build_vocab(conn, min_count=10)
    
    # Prepare test data
    print("\nPreparing test data...")
    test_data = prepare_test_data(conn, vocab, buckets, train_split=0.8)
    print(f"Test samples: {len(test_data)}")
    
    # Load models
    print("\nLoading models...")
    
    baseline = FrequencyBaseline()
    baseline.load(models_dir / "baseline_model.pkl")
    
    tcn_model, _ = load_tcn(models_dir / "tcn_model.pt")
    
    emb_model = AppEmbeddings(vocab)
    emb_model.load(models_dir / "app_embeddings.pkl")
    
    rule_miner = AssociationRuleMiner()
    rule_miner.load(models_dir / "association_rules.pkl")
    
    # Create model variants
    models = {
        'baseline': baseline,
        'hybrid_equal': HybridPredictor(tcn_model, vocab, emb_model, rule_miner, 
                                        weights={'tcn': 0.33, 'emb': 0.33, 'rules': 0.34}),
        'hybrid_tcn_heavy': HybridPredictor(tcn_model, vocab, emb_model, rule_miner,
                                            weights={'tcn': 0.7, 'emb': 0.15, 'rules': 0.15}),
        'hybrid_optimized': HybridPredictor(tcn_model, vocab, emb_model, rule_miner,
                                            weights={'tcn': 0.5, 'emb': 0.3, 'rules': 0.2}),
    }
    
    # Evaluate all
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, model_name, test_data, vocab)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} {'MRR':<10}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['hit@1']:<10.3f} {metrics['hit@3']:<10.3f} "
              f"{metrics['hit@5']:<10.3f} {metrics['mrr']:<10.3f}")
    
    conn.close()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="usage_synthetic.db")
    parser.add_argument("--models", default="outputs/models")
    
    args = parser.parse_args()
    
    run_ablation_study(args.db, args.models)