#!/bin/bash
# Script to activate the DSP virtual environment
# Usage: source activate_env.sh

echo "🔧 Activating DSP virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated!"
echo "📦 Python location: $(which python)"
echo "📦 Pip location: $(which pip)"
echo ""
echo "To deactivate, run: deactivate"
