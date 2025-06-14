#!/bin/bash

# LeRobot Environment Activation Script
# Usage: source activate.sh (NOT ./activate.sh)

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ This script must be sourced, not executed!"
    echo "Usage: source activate.sh"
    echo "   or: . activate.sh"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the LeRobot project directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e ."
    return 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Verify python is available
if ! command -v python &> /dev/null; then
    echo "⚠️  Creating python alias..."
    alias python=python3
fi

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ LeRobot environment activated!"
    echo "📁 Project directory: $(pwd)"
    echo "🐍 Python version: $(python --version)"
    echo "🤖 LeRobot version: $(python -c "import lerobot; print('0.1.0')" 2>/dev/null || echo 'Not installed')"
    echo ""
    echo "🚀 Quick commands:"
    echo "  cd examples/          # Go to examples directory"
    echo "  python 1_load_lerobot_dataset.py  # Run first example"
    echo "  python lerobot/scripts/visualize_dataset.py --repo-id lerobot/pusht --episode-index 0"
    echo ""
    echo "📚 Documentation: https://github.com/huggingface/lerobot"
else
    echo "❌ Failed to activate virtual environment"
    return 1
fi 