#!/bin/bash
# Quick smoke test - verify everything works after clone

echo "🔍 Verifying App Usage Predictor setup..."
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi
echo "✓ Python found: $(python --version)"

# Check dependencies
echo "✓ Checking dependencies..."
python -c "import torch, numpy, pandas, sklearn, plotly, dash" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All Python packages installed"
else
    echo "❌ Missing packages. Run: pip install -r requirements.txt"
    exit 1
fi

# Check X11 tools
if ! command -v xdotool &> /dev/null; then
    echo "⚠️  xdotool not found (needed for logger)"
fi

# Check structure
echo "✓ Checking project structure..."
for dir in data_collection models training evaluation; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing directory: $dir"
        exit 1
    fi
done
echo "✓ Project structure OK"

# Generate test data if needed
if [ ! -f "usage_synthetic.db" ]; then
    echo "📦 Generating synthetic data..."
    python data_collection/logger_test.py
fi

# Quick train test
echo "🏋️  Testing training pipeline..."
python training/train_all.py --db usage_synthetic.db --output outputs/models 2>&1 | tail -5

if [ -f "outputs/models/tcn_model.pt" ]; then
    echo "✓ Models trained successfully"
else
    echo "❌ Training failed"
    exit 1
fi

# Dashboard test (just import, don't run)
python -c "import inference.dashboard" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Dashboard imports OK"
else
    echo "⚠️  Dashboard import failed"
fi

echo ""
echo "================================"
echo "✅ Setup verified!"
echo "================================"
echo "Next steps:"
echo "  1. Start logger: bash scripts/collect_data.sh start"
echo "  2. View dashboard: python inference/dashboard.py"
echo "  3. See README.md for full documentation"