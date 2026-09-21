#!/usr/bin/env bash
#
# deploy/update_services.sh — CNCM I-745 Digital Twin (v4.0.0)
#
# Pull the latest code, refresh dependencies, run the test gate, and restart
# both systemd services. The deploy ABORTS if the test suite fails so a broken
# build never reaches the live API / dashboard.
#
# Usage:  ./deploy/update_services.sh
#
set -euo pipefail

PROJECT_DIR="/home/nsdeshmukh306/digital-twin"
VENV_DIR="/home/nsdeshmukh306/digital-twin-env"

cd "$PROJECT_DIR"

echo "==> [1/6] git pull origin main"
git pull origin main

echo "==> [2/6] activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> [3/6] installing dependencies"
pip install -r "$PROJECT_DIR/requirements.txt"

echo "==> [4/6] running test gate (pytest tests/ -v)"
if ! pytest tests/ -v; then
    echo "!! Tests FAILED — aborting deploy. Services left untouched." >&2
    exit 1
fi

echo "==> [5/6] restarting services"
sudo systemctl restart digital-twin-api
sudo systemctl restart digital-twin-dashboard

echo "==> [6/6] waiting for API health"
for _ in $(seq 1 15); do
    if curl -sf --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -s --max-time 5 http://localhost:8000/health || true
echo ""

echo "Deploy complete at $(date)"
