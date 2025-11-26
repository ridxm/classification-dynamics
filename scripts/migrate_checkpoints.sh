#!/bin/bash
# Script to migrate model checkpoints to outputs/legacy/

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Migrating model checkpoints to outputs/legacy/..."

# Create legacy directories
mkdir -p "$BASE_DIR/outputs/mountain_car_classifier/legacy"
mkdir -p "$BASE_DIR/outputs/pendulum_lqr_classifier/legacy"
mkdir -p "$BASE_DIR/outputs/pendulum_cartesian_classifier/legacy"
mkdir -p "$BASE_DIR/outputs/humanoid_classifier/legacy"

# Mountain Car
if [ -f "$BASE_DIR/mountain-car/mountain_car_classifier.pth" ]; then
    echo "Moving Mountain Car checkpoint..."
    mv "$BASE_DIR/mountain-car/mountain_car_classifier.pth" "$BASE_DIR/outputs/mountain_car_classifier/legacy/"
fi

# Pendulum LQR
if [ -f "$BASE_DIR/pendulum-lqr/pendulum_lqr_classifier.pth" ]; then
    echo "Moving Pendulum LQR checkpoint..."
    mv "$BASE_DIR/pendulum-lqr/pendulum_lqr_classifier.pth" "$BASE_DIR/outputs/pendulum_lqr_classifier/legacy/"
fi

# Pendulum Cartesian
if [ -f "$BASE_DIR/pendulum-cartesian/pendulum_cartesian_classifier.pth" ]; then
    echo "Moving Pendulum Cartesian checkpoint..."
    mv "$BASE_DIR/pendulum-cartesian/pendulum_cartesian_classifier.pth" "$BASE_DIR/outputs/pendulum_cartesian_classifier/legacy/"
fi

# Humanoid
if [ -f "$BASE_DIR/Humanoid/humanoid_classifier_cnn.pth" ]; then
    echo "Moving Humanoid checkpoint..."
    mv "$BASE_DIR/Humanoid/humanoid_classifier_cnn.pth" "$BASE_DIR/outputs/humanoid_classifier/legacy/"
fi

# Cartpole (if exists)
if [ -f "$BASE_DIR/cartpole-pybullet/trajectory_classifier_cnn.pth" ]; then
    echo "Moving Cartpole checkpoint..."
    mkdir -p "$BASE_DIR/outputs/cartpole_classifier/legacy"
    mv "$BASE_DIR/cartpole-pybullet/trajectory_classifier_cnn.pth" "$BASE_DIR/outputs/cartpole_classifier/legacy/"
fi

echo "Checkpoint migration complete!"

