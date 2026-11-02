#!/bin/bash
# Comparison on 1 to 9 objects, 64-dim latent (compare.py): mix/uni datasets x conditions x 3 seeds, then plots.
# Conditions: B (held-out pairs), A (all equalities seen), B2s (B, pairs in both orders), Ase (A, no "same" class).
# Usage: ./run_compare.sh [parallel jobs, default 2]
cd "$(dirname "$0")"; mkdir -p logs
for d in mix uni; do for s in 0 1 2; do
  echo "${d}_B_$s --data $d --condition B --seed $s"
  echo "${d}_A_$s --data $d --condition A --seed $s"
  echo "${d}_B2s_$s --data $d --condition B --deux_sens --seed $s"
  echo "${d}_Ase_$s --data $d --condition A --sans_egal --seed $s"
done; done | xargs -P "${1:-2}" -L 1 bash -c 'log=$0; uv run python compare.py "$@" > logs/$log.log 2>&1'
uv run python plots.py mix > logs/resume_mix.txt 2>&1
uv run python plots.py uni > logs/resume_uni.txt 2>&1
