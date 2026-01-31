import time
import math
import numpy as np
from collections import deque

# Constants (copied from settings)
BUCKET_SIZE = 30 * 60
MAX_ROLLING_BUCKETS = 672


def time_features(timestamp):
    """Extract cyclic time features"""
    t = time.localtime(timestamp)
    
    hour = t.tm_hour + t.tm_min / 60
    day = t.tm_wday
    
    return {
        'hour_sin': math.sin(2 * math.pi * hour / 24),
        'hour_cos': math.cos(2 * math.pi * hour / 24),
        'day_sin': math.sin(2 * math.pi * day / 7),
        'day_cos': math.cos(2 * math.pi * day / 7),
        'is_weekend': 1.0 if day >= 5 else 0.0
    }


class RollingStats:
    """Efficient rolling window statistics"""
    
    def __init__(self, max_buckets=MAX_ROLLING_BUCKETS):
        self.history = deque(maxlen=max_buckets)
        self.last_seen = {}
    
    def update(self, bucket_apps, bucket_id):
        """Add new bucket to history"""
        self.history.append(bucket_apps)
        for app in bucket_apps:
            self.last_seen[app] = bucket_id
    
    def get_stats(self, app, current_bucket):
        """Compute usage statistics for app"""
        count_24h = sum(h.get(app, 0) for h in list(self.history)[-48:])
        count_7d = sum(h.get(app, 0) for h in list(self.history)[-336:])
        count_14d = sum(h.get(app, 0) for h in self.history)
        
        last = self.last_seen.get(app)
        since_last = None if last is None else current_bucket - last
        
        return {
            'count_24h': count_24h,
            'count_7d': count_7d,
            'count_14d': count_14d,
            'since_last_use': since_last
        }


def build_tcn_dataset(buckets, vocab, window=24):
    """Build dataset for TCN training"""
    bucket_ids = sorted(buckets.keys())
    num_apps = len(vocab)
    
    X, Y = [], []
    
    for i in range(len(bucket_ids) - window):
        x = np.zeros((window, num_apps))
        for t in range(window):
            b_id = bucket_ids[i + t]
            for app, count in buckets[b_id].items():
                if app in vocab:
                    x[t, vocab[app]] = count
        
        y = np.zeros(num_apps)
        next_bucket = buckets[bucket_ids[i + window]]
        for app in next_bucket:
            if app in vocab:
                y[vocab[app]] = 1.0
        
        X.append(x)
        Y.append(y)
    
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    import sqlite3
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_processing.preprocessing import aggregate_to_buckets, build_vocab
    
    print("=== Feature Engineering Test ===")
    
    conn = sqlite3.connect("usage_synthetic.db")
    
    buckets = aggregate_to_buckets(conn)
    vocab = build_vocab(conn)
    
    ts = int(time.time())
    feats = time_features(ts)
    print(f"\nTime features for now: {feats}")
    
    stats = RollingStats()
    for b_id in sorted(list(buckets.keys())[:100]):
        stats.update(buckets[b_id], b_id)
    
    test_app = list(vocab.keys())[0]
    app_stats = stats.get_stats(test_app, max(buckets.keys()))
    print(f"\nStats for '{test_app}': {app_stats}")
    
    X, Y = build_tcn_dataset(buckets, vocab, window=24)
    print(f"\nTCN dataset shape: X={X.shape}, Y={Y.shape}")
    
    conn.close()