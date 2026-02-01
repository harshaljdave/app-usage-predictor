#!/bin/bash
# scripts/train_models.sh
# Train all models on collected data
#
# Usage:
#   bash scripts/train_models.sh              # Uses real data
#   bash scripts/train_models.sh --synthetic  # Uses synthetic data

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

# Check database exists
if [ ! -f "$DB" ]; then
    echo "❌ Database not found: $DB"
    if [ "$1" == "--synthetic" ]; then
        echo "   Run: python data_collection/logger_test.py"
    else
        echo "   Run: bash scripts/collect_data.sh start"
    fi
    exit 1
fi

# Check event count
COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM app_events" 2>/dev/null || echo "0")
echo "📊 Total events in database: $COUNT"

if [ "$COUNT" -lt 100 ]; then
    echo "⚠️  Very few events. Results may be unreliable."
    echo "   Consider collecting more data before training."
    read -p "   Continue anyway? (y/N): " reply
    if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
        exit 0
    fi
fi

echo ""
echo "🏋️  Starting training pipeline..."
echo "================================="

python training/train_all.py --db "$DB" --output outputs/models

echo ""
echo "================================="
echo "✓ Training complete"
echo "📂 Models saved to: outputs/models/"
ls -lh outputs/models/