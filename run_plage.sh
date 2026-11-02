#!/bin/bash
# Training range 1..N or 0..N (plage.py). Usage: ./run_plage.sh jobs/plage_jobs.txt [parallel jobs, default 4]
# Each job line: N width_divisor dim_z head seed eg|se [zero]
cd "$(dirname "$0")"; mkdir -p logs/plage
xargs -P "${2:-4}" -L 1 bash -c '
  N=$0; L=$1; Z=$2; T=$3; S=$4; E=$5; ZE=$6
  nom=N${N}_L${L}_z${Z}_t${T}_${E}_seed${S}; oz=""
  [ "$ZE" = zero ] && { nom=N0-${N}_L${L}_z${Z}_t${T}_${E}_seed${S}; oz=--zero; }
  [ -f results/plage/$nom/metrics.json ] && exit 0
  opt=""; [ "$E" = se ] && opt="--sans_egal"
  uv run python plage.py --n_max $N --largeur $L --dim_z $Z --tete $T --seed $S $opt $oz > logs/plage/$nom.log 2>&1
' < "$1"
