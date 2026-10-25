#!/usr/bin/env bash
# Launch the Streamlit dashboard, building any missing pipeline outputs first.
#
# Usage (from anywhere):
#   bash scripts/run_dashboard.sh            # check, build what's missing, launch
#   bash scripts/run_dashboard.sh --check    # only report what's present / missing
#   bash scripts/run_dashboard.sh --rebuild  # re-run every pipeline step, then launch
#
# Stop the dashboard with Ctrl+C in this terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
STREAMLIT="$ROOT/.venv/bin/streamlit"
MODE="${1:-}"

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
miss() { printf "  \033[33m✗\033[0m %s\n" "$1"; }

echo "IDX Exchange AVM dashboard — $ROOT"

# 1. Python environment
if [[ ! -x "$PY" || ! -x "$STREAMLIT" ]]; then
  echo "Python environment not found. Creating .venv and installing requirements (one time, a few minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
ok "Python environment (.venv)"

# 2. Inputs that must be downloaded by hand
n_raw=$(ls data/raw/CRMLSSold*.csv 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n_raw" -eq 0 ]]; then
  miss "No CRMLSSold*.csv files in data/raw/ — download them from the IDX FTP server first."
  exit 1
fi
ok "$n_raw monthly CRMLS files in data/raw/"

DISTRICTS="data/external/school_districts_2425.geojson"
if [[ ! -s "$DISTRICTS" ]]; then
  if [[ "$MODE" == "--check" ]]; then
    miss "School district boundaries missing ($DISTRICTS)"
  else
    echo "  Downloading CA School District Areas 2024-25 (~45 MB, public data from data.ca.gov)..."
    mkdir -p data/external
    curl -sSL -o "$DISTRICTS" \
      "https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
    ok "School district boundaries downloaded"
  fi
else
  ok "School district boundaries"
fi

# 3. Pipeline outputs (each step only runs if its outputs are missing, or with --rebuild)
step() {  # step <label> <marker file> <script>
  local label="$1" marker="$2" script="$3"
  if [[ -s "$marker" && "$MODE" != "--rebuild" ]]; then
    ok "$label"
  elif [[ "$MODE" == "--check" ]]; then
    miss "$label (run: .venv/bin/python $script)"
  else
    echo "  Running $label: $script"
    "$PY" -W ignore "$script"
    ok "$label"
  fi
}
step "Week 3 cleaned data + split"       data/processed/split_metadata.json  scripts/run_preprocessing.py
step "Week 4 baseline model"             models/baseline_linear.joblib       scripts/run_baseline.py
step "Weeks 5-6 model comparison (~10 min)" models/best_model.joblib         scripts/run_models.py

[[ "$MODE" == "--check" ]] && exit 0

# 4. Launch on the first free port from 8501
PORT=8501
while lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT + 1)); done
URL="http://localhost:$PORT"
echo
echo "Starting dashboard at $URL  (press Ctrl+C to stop)"
echo "The Price Estimator page takes ~15 s the first time while the 350 MB model loads."
( sleep 4; open "$URL" 2>/dev/null || true ) &
# localhost only: the dashboard shows aggregates of confidential IDX data, so keep it off the network.
exec "$STREAMLIT" run app/app.py --server.port "$PORT" --server.address localhost --server.headless true
