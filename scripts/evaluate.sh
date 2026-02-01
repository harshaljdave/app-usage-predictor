#!/bin/bash
# scripts/evaluate.sh
# Run evaluation and ablation study
#
# Usage:
#   bash scripts/evaluate.sh              # Uses real data
#   bash scripts/evaluate.sh --synthetic  # Uses synthetic data

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Determine database
if [ "$1" == "--synthetic" ]; then
    DB="usage_synthetic.db"
    echo "📦 Using synthetic data"
else
    DB="usage.db"
    echo "📦 Using real data"
fi

# Check database
if [ ! -f "$DB" ]; then
    echo "❌ Database not found: $DB"
    exit 1
fi

# Check models exist
if [ ! -d "outputs/models" ] || [ -z "$(ls outputs/models/ 2>/dev/null)" ]; then
    echo "❌ No trained models found"
    echo "   Run: bash scripts/train_models.sh"
    exit 1
fi

echo ""
echo "📊 Running ablation study..."
echo "================================="

python evaluation/ablation.py --db "$DB" --models outputs/models

echo ""
echo "================================="
echo "📈 Generating visualizations..."

python evaluation/visualize.py --db "$DB" --models outputs/models --output outputs/figures

echo ""
echo "✓ Evaluation complete"
echo "📂 Figures saved to: outputs/figures/"
ls outputs/figures/