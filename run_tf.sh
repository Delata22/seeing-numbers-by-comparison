#!/bin/bash
# Transformer vs MLP comparison head, ranges 0..N with "same" (plage.py --tete_type). Usage: ./run_tf.sh [parallel jobs, default 4]
# Job lines (jobs/plage_jobs_tf.txt): N head_type seed
cd "$(dirname "$0")"; mkdir -p logs/plage
xargs -P "${1:-4}" -L 1 bash -c '
  uv run python plage.py --n_max $0 --zero --tete_type $1 --seed $2 > logs/plage/tf_$1_N0-$0_$2.log 2>&1
' < jobs/plage_jobs_tf.txt
