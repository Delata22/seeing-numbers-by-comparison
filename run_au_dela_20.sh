#!/bin/bash
# Above 20. Needs results/expressions/ (run_expressions.sh), indexed models in modeles/, and data/train_21a30_aire.npz.
# Usage: ./run_au_dela_20.sh [parallel jobs, default 2]
cd "$(dirname "$0")"; mkdir -p logs/au_dela_20_B logs/au_dela_20_1bis logs/au_dela_20_1bis_long
# Part A: sums read on the image line up to 100
uv run python au_dela_20.py > logs/au_dela_20_run.log 2>&1
# Part B: fine-tuning on images 0..30, including the longer-comparison control
cat jobs/au_dela_20_B_jobs.txt jobs/au_dela_20_B_jobs_controle.txt | xargs -P "${1:-2}" -L 1 bash -c '
  [ -f results/au_dela_20_B/D$0_$1_seed$2.json ] && exit 0
  uv run python au_dela_20_B.py --depart $0 --type $1 --seed $2 > logs/au_dela_20_B/D$0_$1_$2.log 2>&1
'
# Test 1: image of n vs [a]+[b] before / after encoder-only fine-tuning
uv run python au_dela_20_test1.py > logs/au_dela_20_test1_run.log 2>&1
# Test 1bis: mixed fine-tuning, brick trained with the encoder
for v in general succ; do for s in 0 1 2; do for d in 3 9; do echo "$d $v $s"; done; done; done | xargs -P "${1:-2}" -L 1 bash -c '
  [ -f results/au_dela_20_1bis/D$0_$1_seed$2.json ] && exit 0
  uv run python au_dela_20_1bis.py --depart $0 --variante $1 --seed $2 > logs/au_dela_20_1bis/D$0_$1_$2.log 2>&1
'
uv run python resume_1bis.py > logs/resume_1bis.txt
# Long continuation (9000 more steps) of the 6 "general" models
printf "%s\n" "3 0" "9 0" "3 1" "9 1" "3 2" "9 2" | xargs -P "${1:-2}" -L 1 bash -c '
  [ -f results/au_dela_20_1bis_long/D$0_general_seed$1.json ] && exit 0
  uv run python au_dela_20_1bis.py --depart $0 --variante general --seed $1 --depuis results/au_dela_20_1bis \
    --sortie results/au_dela_20_1bis_long --plafond 9000 --sans_arret --journal 500 > logs/au_dela_20_1bis_long/D$0_$1.log 2>&1
'
uv run python resume_1bis_long.py > logs/resume_1bis_long.txt
