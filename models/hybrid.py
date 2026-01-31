import numpy as np
import torch
import time
import pickle
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tcn_model import TCN, load_tcn
from models.embeddings import AppEmbeddings
from models.association_rules import AssociationRuleMiner

def time_features(timestamp):
    """Extract cyclic time features"""
    import math
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


def build_context_vector(
    app_id,
    tcn_scores,
    emb_model,
    rule_miner,
    recent_apps,
    time_feats,
    usage_stats
):
    """Build fixed-length context vector for one app
    
    Returns 12-dimensional vector:
    [tcn, emb, rule, hour_sin, hour_cos, day_sin, day_cos, weekend, 
     count_24h, count_7d, count_14d, since_last]
    """
    # Model scores
    tcn_score = tcn_scores.get(app_id, 0.0)
    emb_score = emb_model.similarity_to_set(app_id, recent_apps) if recent_apps else 0.0
    rule_score = rule_miner.get_rule_score(recent_apps, app_id)
    
    # Assemble vector
    context = [
        float(tcn_score),
        float(emb_score),
        float(rule_score),
        time_feats['hour_sin'],
        time_feats['hour_cos'],
        time_feats['day_sin'],
        time_feats['day_cos'],
        time_feats['is_weekend'],
        usage_stats.get('count_24h', 0),
        usage_stats.get('count_7d', 0),
        usage_stats.get('count_14d', 0),
        usage_stats.get('since_last_use', 0) or 0
    ]
    
    return np.array(context)


class HybridPredictor:
    """Hybrid model combining TCN + Embeddings + Rules"""
    
    def __init__(self, tcn_model, vocab, emb_model, rule_miner, weights=None):
        self.tcn = tcn_model
        self.vocab = vocab
        self.inv_vocab = {i: app for app, i in vocab.items()}
        self.emb = emb_model
        self.rules = rule_miner
        
        # Fusion weights
        if weights is None:
            self.weights = {'tcn': 0.5, 'emb': 0.3, 'rules': 0.2}
        else:
            self.weights = weights
    
    def get_tcn_scores(self, recent_window):
        """Get TCN predictions from recent window
        
        recent_window: (window, num_apps) numpy array
        """
        with torch.no_grad():
            x = torch.FloatTensor(recent_window[None, :, :])
            scores = self.tcn(x).squeeze().numpy()
        
        # Convert to dict
        return {self.inv_vocab[i]: scores[i] for i in range(len(scores))}
    
    def predict(self, context_state, top_k=5):
        """Predict top-K apps
        
        context_state = {
            'recent_window': np.array,  # (window, num_apps) for TCN
            'recent_apps': list,         # Recent app names
            'timestamp': int,            # Current time
            'usage_stats': dict          # {app: {count_24h, count_7d, ...}}
        }
        """
        # Get TCN scores
        tcn_scores = self.get_tcn_scores(context_state['recent_window'])
        
        # Get time features
        time_feats = time_features(context_state['timestamp'])
        
        # Score all apps
        app_scores = {}
        for app in self.vocab:
            # Get usage stats for this app
            stats = context_state.get('usage_stats', {}).get(app, {
                'count_24h': 0,
                'count_7d': 0,
                'count_14d': 0,
                'since_last_use': None
            })
            
            # Build context vector
            ctx = build_context_vector(
                app,
                tcn_scores,
                self.emb,
                self.rules,
                context_state['recent_apps'],
                time_feats,
                stats
            )
            
            # Fused score: weighted sum of first 3 components
            score = (
                self.weights['tcn'] * ctx[0] +
                self.weights['emb'] * ctx[1] +
                self.weights['rules'] * ctx[2]
            )
            
            app_scores[app] = score
        
        # Return top-K
        return sorted(app_scores.items(), key=lambda x: -x[1])[:top_k]


if __name__ == "__main__":
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=== Hybrid Model Test ===")
    
    # Load models
    tcn_model, vocab = load_tcn("tcn_model.pt")
    
    emb_model = AppEmbeddings(vocab)
    emb_model.load("app_embeddings.pkl")
    
    rule_miner = AssociationRuleMiner()
    rule_miner.load("association_rules.pkl")
    
    print("✓ All models loaded")
    
    # Create hybrid
    hybrid = HybridPredictor(tcn_model, vocab, emb_model, rule_miner)
    
    # Test prediction
    num_apps = len(vocab)
    recent_window = np.random.rand(24, num_apps)  # Dummy data
    
    context = {
        'recent_window': recent_window,
        'recent_apps': ['vscode', 'terminal'],
        'timestamp': int(time.time()),
        'usage_stats': {
            'chrome': {'count_24h': 10, 'count_7d': 50, 'count_14d': 100, 'since_last_use': 5},
            'vscode': {'count_24h': 8, 'count_7d': 40, 'count_14d': 80, 'since_last_use': 1}
        }
    }
    
    predictions = hybrid.predict(context, top_k=5)
    
    print("\nHybrid predictions:")
    for app, score in predictions:
        print(f"  {app}: {score:.3f}")
    
    print("\n✓ Hybrid model working!")