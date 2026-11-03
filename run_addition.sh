#!/bin/bash
# addition.py: starting model (0..3 or 0..9) x variante x 3 seeds. Needs the plage models indexed in modeles/ (indexer_modeles.py).
# Usage: ./run_addition.sh [parallel jobs, default 3]. Job lines (jobs/addition_jobs.txt): start variante seed
cd "$(dirname "$0")"; mkdir -p logs/addition
xargs -P "${1:-3}" -L 1 bash -c '
  [ -f results/addition/D$0_$1_seed$2/metrics.json ] && exit 0
  uv run python addition.py --depart $0 --variante $1 --seed $2 > logs/addition/D$0_$1_$2.log 2>&1
' < jobs/addition_jobs.txt
