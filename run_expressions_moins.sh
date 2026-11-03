#!/bin/bash
# expressions_moins.py: starting model (0..3 or 0..9) x init x 3 seeds. Needs the plage models indexed in modeles/ (indexer_modeles.py).
# Usage: ./run_expressions_moins.sh [parallel jobs, default 3]. Job lines (jobs/expressions_moins_jobs.txt): start init seed
cd "$(dirname "$0")"; mkdir -p logs/expressions_moins
xargs -P "${1:-3}" -L 1 bash -c '
  [ -f results/expressions_moins/D$0_$1_seed$2/metrics.json ] && exit 0
  uv run python expressions_moins.py --depart $0 --init $1 --seed $2 > logs/expressions_moins/D$0_$1_$2.log 2>&1
' < jobs/expressions_moins_jobs.txt
