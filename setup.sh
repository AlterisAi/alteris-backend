#!/bin/bash
# Loom Setup Script
# Run this once on a new Mac to install all dependencies.
# Usage: bash setup.sh

set -e

echo "=== Loom Setup ==="
echo ""

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "[1/4] Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to path for Apple Silicon Macs
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
else
    echo "[1/4] Homebrew already installed."
fi

# 2. Install Python 3
if ! command -v python3 &> /dev/null || ! python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    echo "[2/4] Installing Python 3..."
    brew install python@3.13
else
    echo "[2/4] Python 3 already installed: $(python3 --version)"
fi

# 3. Install Loom package
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[3/4] Installing Loom from ${SCRIPT_DIR}..."
pip3 install -e "$SCRIPT_DIR"

# Verify
if python3 -c "import loom" 2>/dev/null; then
    echo "  Loom installed successfully."
else
    echo "  ERROR: Loom import failed. Check the output above."
    exit 1
fi

# 4. Create ~/.loom directory
mkdir -p ~/.loom
echo "[4/4] Created ~/.loom data directory."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Open LoomApp.app"
echo "  2. Grant Full Disk Access in System Settings > Privacy & Security"
echo "  3. The app will guide you through API keys and profile setup"
echo ""
