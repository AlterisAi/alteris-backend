#!/bin/bash
# Build a distributable Alteris DMG containing the macOS app + bundled CLI.
#
# Prerequisites:
#   - .venv exists in repo root (python3 -m venv .venv && pip install -e .)
#   - PyInstaller installed in .venv (pip install pyinstaller)
#   - Xcode with AlterisApp project at ~/Code/alteris-app/AlterisApp/
#
# Usage:
#   cd ~/Code/alteris-backend
#   bash scripts/dist/build_dmg.sh          # builds Alteris-<version>.dmg
#   bash scripts/dist/build_dmg.sh 1.2      # override version

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_REPO="${APP_REPO:-$HOME/Code/alteris-app/AlterisApp}"
VERSION="${1:-1.1}"
BUILD_DIR="$REPO_ROOT/build"
DMG_STAGING="$BUILD_DIR/dmg-staging"
DMG_OUT="$REPO_ROOT/Alteris-${VERSION}.dmg"

echo "=== Alteris DMG Build (v${VERSION}) ==="
echo "  Backend: $REPO_ROOT"
echo "  App:     $APP_REPO"
echo "  Output:  $DMG_OUT"
echo ""

# ── Step 1: Build CLI binary with PyInstaller ──────────────────────
echo "[1/4] Building CLI binary..."

if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"

pyinstaller \
    "$REPO_ROOT/scripts/dist/alteris-cli.spec" \
    --noconfirm \
    --distpath "$BUILD_DIR/dist" \
    --workpath "$BUILD_DIR/pyinstaller" \
    2>&1 | tail -5

if [ ! -f "$BUILD_DIR/dist/alteris-cli/alteris-cli" ]; then
    echo "ERROR: PyInstaller build failed — no binary produced."
    exit 1
fi
echo "  CLI binary OK"

# ── Step 2: Build macOS app with Xcode ─────────────────────────────
echo "[2/4] Building AlterisApp (Release)..."

if [ ! -d "$APP_REPO/AlterisApp.xcodeproj" ]; then
    echo "ERROR: Xcode project not found at $APP_REPO/AlterisApp.xcodeproj"
    exit 1
fi

xcodebuild \
    -project "$APP_REPO/AlterisApp.xcodeproj" \
    -scheme AlterisApp \
    -configuration Release \
    -derivedDataPath "$BUILD_DIR/xcode" \
    SWIFT_TREAT_WARNINGS_AS_ERRORS=NO \
    2>&1 | tail -5

APP_PATH="$BUILD_DIR/xcode/Build/Products/Release/AlterisApp.app"
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: Xcode build failed — no .app produced."
    exit 1
fi
echo "  AlterisApp.app OK"

# ── Step 3: Assemble DMG staging ───────────────────────────────────
echo "[3/4] Assembling DMG staging..."

rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"

# Copy app
cp -R "$APP_PATH" "$DMG_STAGING/AlterisApp.app"

# Bundle CLI into app
CLI_DEST="$DMG_STAGING/AlterisApp.app/Contents/Resources/alteris-cli"
cp -R "$BUILD_DIR/dist/alteris-cli" "$CLI_DEST"

# Applications symlink for drag-to-install
ln -s /Applications "$DMG_STAGING/Applications"

# Include setup.sh
cp "$REPO_ROOT/setup.sh" "$DMG_STAGING/setup.sh" 2>/dev/null || true

echo "  Staging ready"

# ── Step 4: Create DMG ─────────────────────────────────────────────
echo "[4/4] Creating DMG..."

[ -f "$DMG_OUT" ] && rm "$DMG_OUT"

hdiutil create \
    -volname "Alteris" \
    -srcfolder "$DMG_STAGING" \
    -ov \
    -format UDZO \
    "$DMG_OUT" \
    2>&1 | tail -3

DMG_SIZE=$(du -h "$DMG_OUT" | cut -f1)
echo ""
echo "=== Done: $DMG_OUT ($DMG_SIZE) ==="

# ── Quick sanity check ─────────────────────────────────────────────
echo ""
echo "Sanity check — bundled CLI:"
"$CLI_DEST/alteris-cli" --help 2>&1 | head -3 || echo "(help output failed — check manually)"
