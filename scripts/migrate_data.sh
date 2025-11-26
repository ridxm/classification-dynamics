#!/bin/bash
# Script to migrate data files from old structure to new structure
# NOTE: All data is now stored in the shared directory:
# /common/users/shared/pracsys/genMoPlan/data_trajectories
# This script is kept for reference but data migration is not needed.

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "Data Migration Script"
echo "=========================================="
echo ""
echo "NOTE: All data files are stored in the shared directory:"
echo "  /common/users/shared/pracsys/genMoPlan/data_trajectories"
echo ""
echo "The following directories are used:"
echo "  - mountain_car_power_0p0008"
echo "  - pendulum_lqr_50k"
echo "  - pendulum_cartesian_50k"
echo "  - humanoid_get_up"
echo ""
echo "All data modules and configs have been updated to use these paths."
echo "No local data migration is needed."
echo ""
echo "If you have local data files you want to archive, you can move them to:"
echo "  $BASE_DIR/archive/"
echo ""
echo "=========================================="

