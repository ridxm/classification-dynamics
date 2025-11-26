#!/bin/bash
# Script to archive old notebooks to archive/ directory
# This preserves them for reference but moves them out of the way

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$BASE_DIR/archive"

echo "=========================================="
echo "Archiving Old Notebooks"
echo "=========================================="
echo ""
echo "This script will move old notebooks to archive/ directory"
echo "They will be preserved for reference but won't clutter the workspace"
echo ""

# Create archive directory structure
mkdir -p "$ARCHIVE_DIR/mountain-car"
mkdir -p "$ARCHIVE_DIR/pendulum-lqr"
mkdir -p "$ARCHIVE_DIR/pendulum-cartesian"
mkdir -p "$ARCHIVE_DIR/Humanoid"
mkdir -p "$ARCHIVE_DIR/cartpole-pybullet"

# Archive Mountain Car notebooks
if [ -d "$BASE_DIR/mountain-car" ]; then
    echo "Archiving Mountain Car notebooks..."
    if [ -f "$BASE_DIR/mountain-car/model.ipynb" ]; then
        mv "$BASE_DIR/mountain-car/model.ipynb" "$ARCHIVE_DIR/mountain-car/" 2>/dev/null || echo "  Already archived or not found"
    fi
    if [ -f "$BASE_DIR/mountain-car/model-500.ipynb" ]; then
        mv "$BASE_DIR/mountain-car/model-500.ipynb" "$ARCHIVE_DIR/mountain-car/" 2>/dev/null || echo "  Already archived or not found"
    fi
    echo "✓ Mountain Car notebooks archived"
fi

# Archive Pendulum LQR notebooks
if [ -d "$BASE_DIR/pendulum-lqr" ]; then
    echo "Archiving Pendulum LQR notebooks..."
    if [ -f "$BASE_DIR/pendulum-lqr/model.ipynb" ]; then
        mv "$BASE_DIR/pendulum-lqr/model.ipynb" "$ARCHIVE_DIR/pendulum-lqr/" 2>/dev/null || echo "  Already archived or not found"
    fi
    if [ -f "$BASE_DIR/pendulum-lqr/model-500.ipynb" ]; then
        mv "$BASE_DIR/pendulum-lqr/model-500.ipynb" "$ARCHIVE_DIR/pendulum-lqr/" 2>/dev/null || echo "  Already archived or not found"
    fi
    echo "✓ Pendulum LQR notebooks archived"
fi

# Archive Pendulum Cartesian notebooks
if [ -d "$BASE_DIR/pendulum-cartesian" ]; then
    echo "Archiving Pendulum Cartesian notebooks..."
    if [ -f "$BASE_DIR/pendulum-cartesian/model.ipynb" ]; then
        mv "$BASE_DIR/pendulum-cartesian/model.ipynb" "$ARCHIVE_DIR/pendulum-cartesian/" 2>/dev/null || echo "  Already archived or not found"
    fi
    if [ -f "$BASE_DIR/pendulum-cartesian/model-500.ipynb" ]; then
        mv "$BASE_DIR/pendulum-cartesian/model-500.ipynb" "$ARCHIVE_DIR/pendulum-cartesian/" 2>/dev/null || echo "  Already archived or not found"
    fi
    echo "✓ Pendulum Cartesian notebooks archived"
fi

# Archive Humanoid notebooks
if [ -d "$BASE_DIR/Humanoid" ]; then
    echo "Archiving Humanoid notebooks..."
    if [ -f "$BASE_DIR/Humanoid/model.ipynb" ]; then
        mv "$BASE_DIR/Humanoid/model.ipynb" "$ARCHIVE_DIR/Humanoid/" 2>/dev/null || echo "  Already archived or not found"
    fi
    echo "✓ Humanoid notebooks archived"
fi

# Archive Cartpole notebooks
if [ -d "$BASE_DIR/cartpole-pybullet" ]; then
    echo "Archiving Cartpole notebooks..."
    if [ -f "$BASE_DIR/cartpole-pybullet/model-500.ipynb" ]; then
        mv "$BASE_DIR/cartpole-pybullet/model-500.ipynb" "$ARCHIVE_DIR/cartpole-pybullet/" 2>/dev/null || echo "  Already archived or not found"
    fi
    if [ -f "$BASE_DIR/cartpole-pybullet/model-1000.ipynb" ]; then
        mv "$BASE_DIR/cartpole-pybullet/model-1000.ipynb" "$ARCHIVE_DIR/cartpole-pybullet/" 2>/dev/null || echo "  Already archived or not found"
    fi
    echo "✓ Cartpole notebooks archived"
fi

echo ""
echo "=========================================="
echo "Archive Complete!"
echo "=========================================="
echo ""
echo "Old notebooks have been moved to: $ARCHIVE_DIR"
echo ""
echo "You can now use the new config-driven notebooks in:"
echo "  notebooks/mountain_car/example_usage.ipynb"
echo "  notebooks/pendulum_lqr/example_usage.ipynb"
echo "  notebooks/pendulum_cartesian/example_usage.ipynb"
echo "  notebooks/humanoid/example_usage.ipynb"
echo ""
echo "All parameters are in config files:"
echo "  configs/notebook/mountain_car_viz.yaml"
echo "  configs/notebook/pendulum_lqr_viz.yaml"
echo "  configs/notebook/pendulum_cartesian_viz.yaml"
echo "  configs/notebook/humanoid_viz.yaml"
echo ""

