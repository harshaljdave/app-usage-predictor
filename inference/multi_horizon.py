import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def predict_multi_horizon(tcn_model, recent_window, vocab, horizons=[1, 3, 6]):
    """Autoregressive multi-horizon prediction.

    Predicts multiple steps ahead by feeding each prediction back
    as input for the next step. Uses sigmoid threshold (0.5) to
    binarize predictions before feeding back.

    Args:
        tcn_model: Trained TCN model
        recent_window: np.array (24, num_apps) - recent usage history
        vocab: dict {app_name: index}
        horizons: list of steps to return predictions for

    Returns:
        dict: {step: [(app, score), ...]} sorted by score descending
    """
    inv_vocab = {i: app for app, i in vocab.items()}
    max_horizon = max(horizons)
    window = recent_window.copy()

    results = {}

    with torch.no_grad():
        for step in range(1, max_horizon + 1):
            input_tensor = torch.FloatTensor(window[None, :, :])
            pred = tcn_model(input_tensor).squeeze().numpy()

            if step in horizons:
                ranked = sorted(
                    [(inv_vocab[i], float(pred[i])) for i in range(len(pred))],
                    key=lambda x: -x[1]
                )
                results[step] = ranked

            # Autoregressive: binarize prediction, shift window, append
            next_bucket = (pred > 0.5).astype(float)
            window = np.vstack([window[1:], next_bucket[None, :]])

    return results


def get_horizon_label(step, bucket_size_mins=30):
    """Convert bucket steps to human-readable time labels"""
    total_mins = step * bucket_size_mins
    if total_mins < 60:
        return f"Next {total_mins}min"
    hours = total_mins // 60
    mins = total_mins % 60
    if mins == 0:
        return f"Next {hours}h"
    return f"Next {hours}h {mins}min"