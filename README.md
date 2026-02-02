# App Usage Prediction System

A hybrid machine learning system that learns laptop usage patterns and predicts which application you'll use next. Built for Linux/X11 with live prediction dashboard and online learning.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

Predicts next-app usage by combining three complementary ML approaches:
- **Temporal Convolutional Network (TCN)** — Short-term sequential patterns (12h memory)
- **Skip-gram embeddings** — Long-term co-usage relationships
- **Association rules** — Stable workflow habits with explainability

Self-collected dataset on Fedora/KDE with enhanced window title parsing (e.g., `chrome:work:gmail`, `terminal:project-dir`).

## 📊 Performance

Evaluated on synthetic data (101 test samples, 6 apps):

| Model | Hit@1 | Hit@3 | Hit@5 | MRR |
|-------|-------|-------|-------|-----|
| Baseline (frequency) | 34.7% | 81.2% | 97.0% | 0.598 |
| **Hybrid (optimized)** | 32.7% | **85.1%** | 99.0% | 0.583 |

**Key insight:** Hybrid leads on Hit@3 (+4.8%), the metric that matters for real usage. Baseline competitive on Hit@1 due to limited vocabulary — hybrid expected to differentiate clearly with real data (more apps, varied patterns).

## 🏗️ Architecture
```
┌─────────────────────────────────┐
│   X11 Events + Window Titles    │
│   chrome:work:github            │
│   terminal:app-predictor        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│    Feature Engineering          │
│  • 30-min time buckets          │
│  • Session extraction           │
│  • Cyclic time features         │
│  • Rolling statistics           │
└────────────┬────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌───────┐┌──────┐┌────────┐
│  TCN  ││ Emb. ││ Rules  │
│ 12h   ││ Co-  ││Stable  │
│memory ││usage ││habits  │
└───┬───┘└──┬───┘└───┬────┘
    │       │        │
    └───────┼────────┘
            ▼
    ┌──────────────┐
    │ Weighted     │
    │ Fusion       │
    │ α=0.5 β=0.3  │
    │ γ=0.2        │
    └──────────────┘
```

### Model Components

**TCN (Temporal Convolutional Network)**
- 3 dilated causal conv blocks [1, 2, 4]
- 7,430 parameters
- Input: 24 buckets (12 hours) × num_apps
- Output: Next-bucket probability distribution

**App Embeddings** (Skip-gram)
- 16-dimensional dense vectors
- Trained per-session (online updates)
- Captures "VSCode and Terminal are similar"

**Association Rules** (FP-Growth)
- Mined 63 itemsets, 122 rules
- Example: `{VSCode, Terminal} → Chrome (conf=0.81)`
- Weekly re-mining (stable patterns)

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/app-usage-predictor
cd app-usage-predictor

# Create environment (Python 3.12)
conda create -n pattern python=3.12
conda activate pattern

# Install dependencies
pip install -r requirements.txt

# Install X11 tools (Fedora/RHEL)
sudo dnf install xdotool xprintidle

# Or on Ubuntu/Debian
sudo apt install xdotool xprintidle
```

### Data Collection

**Start logger** (requires X11 session):
```bash
bash scripts/collect_data.sh start
```

**Check status:**
```bash
bash scripts/collect_data.sh status
```

Logger captures:
- Application focus changes
- Window titles → `chrome:profile:domain`, `terminal:directory`
- Idle/lock detection (auto-pauses)
- **Online embedding updates** at session end

**Recommended:** Run for 2-3 weeks for robust training data.

### Training
```bash
# Train all models on collected data
bash scripts/train_models.sh

# Or use synthetic data for testing
bash scripts/train_models.sh --synthetic
```

Output:
```
[1/6] Preprocessing data...
✓ Aggregated 2961 events into 621 buckets
✓ Vocabulary size: 6 apps

[2/6] Training baseline...
[3/6] Training TCN...
[4/6] Training embeddings...
[5/6] Mining association rules...
[6/6] Complete
```

### Evaluation
```bash
# Run ablation study + generate visualizations
bash scripts/evaluate.sh

# Visualizations saved to outputs/figures/
```

## 📊 Live Dashboard

Run the interactive dashboard:
```bash
python inference/dashboard.py
```

Open `http://127.0.0.1:8050` in your browser.

**Features:**
- **Live Top-5 predictions** with confidence bars
- **Multi-horizon predictions** (next 30min / 1.5h / 3h)
- **Prediction accuracy tracking** with rolling history
- **Recent activity timeline**
- Auto-refreshes every 5 seconds

**Note:** Edit `DB_PATH` in `dashboard.py` to switch between `usage.db` (real) and `usage_synthetic.db` (test).

## 💻 Standalone Prediction

Test models from command line without dashboard:
```bash
# Quick prediction
python inference/predict.py

# With multi-horizon
python inference/predict.py --multi-horizon

# On synthetic data
python inference/predict.py --db usage_synthetic.db --top-k 3
```

Output:
```
Loading models...
✓ Models loaded (6 apps)

📍 Context: Recent apps = vscode, terminal, chrome

🔮 Top-5 Predictions:

  1. chrome                     0.905  ████████████████████
  2. vscode                     0.889  █████████████████
  3. terminal                   0.803  ████████████████
  4. slack                      0.762  ███████████████
  5. spotify                    0.552  ███████████
```

## 📖 Usage Example
```python
from models.hybrid import HybridPredictor
from models.tcn_model import load_tcn
from models.embeddings import AppEmbeddings
from models.association_rules import AssociationRuleMiner
import numpy as np

# Load models
tcn_model, vocab = load_tcn("outputs/models/tcn_model.pt")
emb_model = AppEmbeddings(vocab)
emb_model.load("outputs/models/app_embeddings.pkl")
rule_miner = AssociationRuleMiner()
rule_miner.load("outputs/models/association_rules.pkl")

hybrid = HybridPredictor(tcn_model, vocab, emb_model, rule_miner)

# Prepare context
context = {
    'recent_window': recent_usage_matrix,  # (24, num_apps)
    'recent_apps': ['vscode', 'terminal'],
    'timestamp': int(time.time()),
    'usage_stats': {}  # Rolling statistics
}

# Get predictions
predictions = hybrid.predict(context, top_k=5)

for app, score in predictions:
    print(f"{app}: {score:.3f}")
```

## 📁 Project Structure
```
app-usage-predictor/
├── data_collection/
│   ├── logger_x11.py          # Main logger (with online embeddings)
│   ├── logger_test.py         # Synthetic data generator
│   └── db_core.py             # Database schemas
│
├── data_processing/
│   ├── preprocessing.py       # Bucketing, sessions, vocab
│   └── feature_engineering.py # Time features, rolling stats
│
├── models/
│   ├── baseline.py            # Frequency-based baseline
│   ├── tcn_model.py           # Temporal CNN
│   ├── embeddings.py          # Skip-gram embeddings
│   ├── association_rules.py   # FP-Growth rules
│   └── hybrid.py              # Combined predictor
│
├── training/
│   └── train_all.py           # Master training pipeline
│
├── evaluation/
│   ├── metrics.py             # Hit@K, MRR, Precision
│   ├── ablation.py            # Model comparison
│   └── visualize.py           # Generate 6 Plotly charts
│
├── inference/
│   ├── predict.py             # CLI prediction tool
│   ├── dashboard.py           # Live Dash dashboard
│   └── multi_horizon.py       # Autoregressive predictions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results_visualization.ipynb
│
├── scripts/
│   ├── collect_data.sh        # Start/stop/status logger
│   ├── train_models.sh        # Run training pipeline
│   └── evaluate.sh            # Run evaluation + viz
│
└── outputs/
    ├── models/                # Trained weights (.pt, .pkl)
    ├── figures/               # 6 HTML visualizations
    └── logs/                  # Logger output
```

## 🔬 Technical Details

### Data Processing
- **Time bucketing:** 30-minute intervals reduce noise
- **Sessionization:** Idle gaps >15 minutes define boundaries
- **Features:** Cyclic time encoding (sin/cos), rolling statistics (24h/7d/14d), recency

### Model Hyperparameters

**TCN:**
- Hidden dim: 32, Kernel: 3, Dilations: [1, 2, 4]
- Receptive field: 12 hours
- Training: Adam (lr=1e-3), 20 epochs
- Converges at epoch 4 (validation loss plateaus)

**Embeddings:**
- Dimension: 16, Window: 2 (skip-gram)
- Learning rate: 0.01
- **Online updates:** Per-session at idle threshold

**Association Rules:**
- Min support: 5, Min confidence: 0.6
- Max itemset size: 3

**Fusion Weights (optimized):**
- TCN: 0.5, Embeddings: 0.3, Rules: 0.2

### Performance
- **Model size:** ~15MB total
- **Training time:** ~2 minutes (CPU)
- **Memory footprint:** ~80MB runtime
- **Inference latency:** <100ms (target)

### Online Learning Strategy / Periodic Retraining

**What updates online:**
- ✅ **Embeddings** — per-session updates in logger (cheap, no forgetting risk)
- ✅ **Predictions** — logged to `model_predictions` table for accuracy tracking

**What doesn't:**
- ❌ **TCN** — periodic retrain (weekly) avoids catastrophic forgetting
- ❌ **Rules** — weekly/monthly re-mining (stable patterns)

## ⚠️ Limitations

**Granularity:**
- Tracks application-level focus, not URLs or terminal commands
- Window title parsing adds context (e.g., `chrome:work:gmail`) but not full content
- Background processes ignored (e.g., Spotify while coding)

**Platform:**
- Requires X11 session (Wayland restricts window introspection)
- Linux-specific (uses xdotool, xprintidle)
- Could adapt for Windows/macOS/Android with platform-specific APIs

**Architecture:**
- Logger couples data collection with embedding training (convenience over purity)
- For proffesional/production: separate `embedding_updater.py` daemon recommended

**Dataset:**
- Synthetic data has limited diversity (6 apps, repetitive patterns)
- Real data collection ongoing (need 2-3 weeks for robust evaluation)

**Model:**
- Vocabulary limited to frequently used apps (min_count=10)
- Cold start problem for new apps
- No temporal drift adaptation yet (planned)


## 🔮 Possible Future Work
- Contextual bandits for active recommendations
- User feedback collection (click/dismiss)
- Browser extension for URL-level tracking
- Shell integration for command-level context
- Cross-device learning (laptop + phone)
- Federated learning for privacy

## 🛠️ Development

**Run tests:**
```bash
pytest tests/
```

**Generate synthetic data:**
```bash
python data_collection/logger_test.py
```

**Inspect database:**
```bash
sqlite3 usage.db "SELECT COUNT(*) FROM app_events"
sqlite3 usage.db "SELECT app_id, COUNT(*) FROM app_events GROUP BY app_id ORDER BY COUNT(*) DESC"
```

**Verify setup:**
```bash
bash scripts/verify_setup.sh
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Attribution

If you use or build upon this work, please consider crediting the original author:

**Harshal Dave**  
GitHub: [harshaljdave/app-usage-predictor](https://github.com/harshaljdave/app-usage-predictor)

For academic use:
```bibtex
@misc{dave2026apppredictor,
  author = {Dave, Harshal},
  title = {Hybrid App Usage Prediction System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/harshaljdave/app-usage-predictor}
}
```
---

**Status:** Core system complete · Real data collection ongoing · Dashboard live

**Tech Stack:** Python 3.12 · PyTorch · SQLite · Dash · Plotly · X11

**Author:** [Harshal Dave] · M.Tech Student · [SVNIT]