#!/bin/bash
# 1-100 series (cent.py), 192x192 images. Needs data/cent_X.npy (generate_cent.py).
# Usage: ./run_cent.sh [parallel jobs, default 2]
# jobs/cent_jobs.txt lines: distribution seed eg|se. Runs without "same" ("se") are then redone with the fix
# (lr 2e-4, gradient clipping 1) listed in jobs/cent_jobs_fix.txt (distribution seed); the reported numbers use the fixed runs.
cd "$(dirname "$0")"; mkdir -p logs/cent
xargs -P "${1:-2}" -L 1 bash -c '
  opt=""; [ "$2" = se ] && opt="--sans_egal"
  uv run python cent.py --distrib $0 --seed $1 $opt --plafond 8000 >> logs/cent/$0_$2_$1.log 2>&1
' < jobs/cent_jobs.txt
xargs -P "${1:-2}" -L 1 bash -c '
  uv run python cent.py --distrib $0 --seed $1 --sans_egal --plafond 8000 --lr 2e-4 --clip 1 > logs/cent/fix_$0_se_$1.log 2>&1
' < jobs/cent_jobs_fix.txt
uv run python resume_cent.py
