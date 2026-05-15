#!/usr/bin/env bash
set -euo pipefail

# Prepare Unitree robot-description assets for Task 3 platform evidence.
# This script does not run Isaac Sim or Isaac Lab.
# It downloads public robot-description files outside the project repo.

EXTERNAL_ROOT="${1:-$HOME/Desktop/COGAR/external}"
mkdir -p "$EXTERNAL_ROOT"

cd "$EXTERNAL_ROOT"

echo "[INFO] External asset root: $EXTERNAL_ROOT"

if [ ! -d g1_description ]; then
  echo "[INFO] Cloning Unitree G1 description repository..."
  git clone https://github.com/isri-aist/g1_description.git
else
  echo "[INFO] g1_description already exists."
fi

echo
echo "[INFO] Checking G1 URDF/MJCF files..."
find "$EXTERNAL_ROOT/g1_description" \
  \( -iname "*.urdf" -o -iname "*.xml" -o -iname "*.mjcf" \) \
  | sort

echo
echo "[INFO] Expected useful files include:"
echo "  g1_description/urdf/g1_23dof.urdf"
echo "  g1_description/urdf/g1_29dof.urdf"

echo
echo "[INFO] These assets are intended for later import into Isaac Sim using the URDF importer."
echo "[INFO] Do not commit external robot assets into the benchmark repository."
