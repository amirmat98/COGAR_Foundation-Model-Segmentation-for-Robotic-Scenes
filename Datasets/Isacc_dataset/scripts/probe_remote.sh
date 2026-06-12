#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -a
else
  cat /etc/os-release
fi

echo
echo "== Kernel =="
uname -a

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "== Docker =="
if command -v docker >/dev/null 2>&1; then
  docker --version
  echo
  echo "Docker images:"
  docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" || true
else
  echo "docker not found"
fi

echo
echo "== Isaac Sim Launchers =="
for candidate in \
  isaacsim \
  "$HOME/isaacsim/python.sh" \
  "$HOME/isaac-sim/python.sh" \
  "/opt/isaac-sim/python.sh" \
  "/isaac-sim/python.sh"; do
  if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
    echo "found: $candidate"
  fi
done

echo
echo "== Disk =="
df -h .
