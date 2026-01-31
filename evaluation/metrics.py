import numpy as np


def hit_at_k(predictions, actual, k=3):
    """Check if actual is in top-K predictions"""
    pred_apps = [p[0] for p in predictions[:k]]
    return 1.0 if actual in pred_apps else 0.0


def mean_reciprocal_rank(predictions, actual):
    """MRR: 1/rank if found, else 0"""
    pred_apps = [p[0] for p in predictions]
    try:
        rank = pred_apps.index(actual) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def precision_at_k(predictions, actual, k=3):
    """Precision@K (binary relevance)"""
    return 1.0 / k if hit_at_k(predictions, actual, k) else 0.0


class EvaluationMetrics:
    """Track multiple metrics across predictions"""
    
    def __init__(self):
        self.metrics = {
            'hit@1': [],
            'hit@3': [],
            'hit@5': [],
            'mrr': [],
            'precision@3': []
        }
    
    def add_prediction(self, predictions, actual):
        """Record one prediction
        
        predictions: list of (app, score) tuples
        actual: ground truth app
        """
        self.metrics['hit@1'].append(hit_at_k(predictions, actual, k=1))
        self.metrics['hit@3'].append(hit_at_k(predictions, actual, k=3))
        self.metrics['hit@5'].append(hit_at_k(predictions, actual, k=5))
        self.metrics['mrr'].append(mean_reciprocal_rank(predictions, actual))
        self.metrics['precision@3'].append(precision_at_k(predictions, actual, k=3))
    
    def summarize(self):
        """Compute mean metrics"""
        return {k: np.mean(v) if v else 0.0 for k, v in self.metrics.items()}
    
    def count(self):
        """Number of predictions"""
        return len(self.metrics['hit@1'])


if __name__ == "__main__":
    # Test metrics
    metrics = EvaluationMetrics()
    
    # Perfect prediction
    metrics.add_prediction([('chrome', 0.9), ('vscode', 0.7)], 'chrome')
    
    # Hit@3 but not Hit@1
    metrics.add_prediction([('slack', 0.8), ('terminal', 0.6), ('chrome', 0.5)], 'chrome')
    
    # Miss
    metrics.add_prediction([('vscode', 0.9), ('terminal', 0.8)], 'spotify')
    
    print("Test metrics:")
    for k, v in metrics.summarize().items():
        print(f"  {k}: {v:.3f}")