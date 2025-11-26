#!/bin/bash
# Script to archive old folders from before restructuring
# This preserves them for reference but moves them out of the way

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$BASE_DIR/archive"

echo "=========================================="
echo "Archiving Old Folders"
echo "=========================================="
echo ""
echo "This script will move old folders to archive/ directory"
echo "They will be preserved for reference but won't clutter the workspace"
echo ""

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Archive Mountain Car folder
if [ -d "$BASE_DIR/mountain-car" ]; then
    echo "Archiving mountain-car/ folder..."
    mv "$BASE_DIR/mountain-car" "$ARCHIVE_DIR/" 2>/dev/null || echo "  Already archived or not found"
    echo "✓ mountain-car/ archived"
fi

# Archive Pendulum LQR folder
if [ -d "$BASE_DIR/pendulum-lqr" ]; then
    echo "Archiving pendulum-lqr/ folder..."
    mv "$BASE_DIR/pendulum-lqr" "$ARCHIVE_DIR/" 2>/dev/null || echo "  Already archived or not found"
    echo "✓ pendulum-lqr/ archived"
fi

# Archive Pendulum Cartesian folder
if [ -d "$BASE_DIR/pendulum-cartesian" ]; then
    echo "Archiving pendulum-cartesian/ folder..."
    mv "$BASE_DIR/pendulum-cartesian" "$ARCHIVE_DIR/" 2>/dev/null || echo "  Already archived or not found"
    echo "✓ pendulum-cartesian/ archived"
fi

# Archive Humanoid folder
if [ -d "$BASE_DIR/Humanoid" ]; then
    echo "Archiving Humanoid/ folder..."
    mv "$BASE_DIR/Humanoid" "$ARCHIVE_DIR/" 2>/dev/null || echo "  Already archived or not found"
    echo "✓ Humanoid/ archived"
fi

# Archive Cartpole PyBullet folder
if [ -d "$BASE_DIR/cartpole-pybullet" ]; then
    echo "Archiving cartpole-pybullet/ folder..."
    mv "$BASE_DIR/cartpole-pybullet" "$ARCHIVE_DIR/" 2>/dev/null || echo "  Already archived or not found"
    echo "✓ cartpole-pybullet/ archived"
fi

echo ""
echo "=========================================="
echo "Archive Complete!"
echo "=========================================="
echo ""
echo "Old folders have been moved to: $ARCHIVE_DIR"
echo ""
echo "Archived folders:"
ls -1 "$ARCHIVE_DIR" 2>/dev/null || echo "  (none)"
echo ""
echo "You can now use the new restructured codebase:"
echo "  - New notebooks: notebooks/<system>/model.ipynb"
echo "  - Config files: configs/notebook/<system>_viz.yaml"
echo "  - Training scripts: src/training/train.py"
echo ""

