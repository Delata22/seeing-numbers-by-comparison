#!/bin/bash
# expressions.py: starting model (0..3 or 0..9) x variante x 3 seeds. Needs the plage models indexed in modeles/ (indexer_modeles.py).
# Usage: ./run_expressions.sh [parallel jobs, default 3]. Job lines (jobs/expressions_jobs.txt): start variante seed
cd "$(dirname "$0")"; mkdir -p logs/expressions
xargs -P "${1:-3}" -L 1 bash -c '
  [ -f results/expressions/D$0_$1_seed$2/metrics.json ] && exit 0
  uv run python expressions.py --depart $0 --variante $1 --seed $2 > logs/expressions/D$0_$1_$2.log 2>&1
' < jobs/expressions_jobs.txt
