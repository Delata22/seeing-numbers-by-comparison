#!/bin/bash
# Encoder size reduction (reduction.py). Usage: ./run_reduction.sh jobs/reduction_jobs_a.txt [parallel jobs, default 4]
# Each job line: width_divisor dim_z seed eg|se [head neurons, default 128]
cd "$(dirname "$0")"; mkdir -p logs/reduction
xargs -P "${2:-4}" -L 1 bash -c '
  L=$0; Z=$1; S=$2; E=$3; T=${4:-128}
  suf=""; [ "$T" != 128 ] && suf="_t$T"
  [ -f results/reduction/L${L}_z${Z}_${E}${suf}_seed${S}/metrics.json ] && exit 0
  opt=""; [ "$E" = se ] && opt="--sans_egal"
  uv run python reduction.py --largeur $L --dim_z $Z --seed $S --tete $T $opt > logs/reduction/L${L}_z${Z}_${E}${suf}_${S}.log 2>&1
' < "$1"
