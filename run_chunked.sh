#!/usr/bin/env bash
# Chunked runner: small chunks, waits for the GPU to cool between chunks.
# Safe to stop with Ctrl+C and to re-run: finished runs are skipped (results are saved after every run).
#
#   bash run_chunked.sh controls    # checks that the main result is not an artefact
#   bash run_chunked.sh diversity   # pretrain on heat only / heat+advection, then transfer
#   bash run_chunked.sh plots       # figures and tables
#
# Environment variables:  MAX_TEMP=75 (pause above this GPU temperature, deg C)   DRY_RUN=1 (print only)
set -e
MAX_TEMP=${MAX_TEMP:-75}

cool() {
  [ -n "$DRY_RUN" ] && return 0
  while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n1)
    if [ "$t" -le "$MAX_TEMP" ]; then break; fi
    echo "   GPU at ${t} C - waiting until it is at or below ${MAX_TEMP} C ..."
    sleep 30
  done
}

step() {
  cool
  echo ">>> $*"
  if [ -z "$DRY_RUN" ]; then "$@"; fi
}

pretrain_once() {            # pretrain_once NAME ARGS...  (skips if the checkpoint already exists)
  name=$1; shift
  if [ -f "ckpt/${name}.pt" ]; then echo "(ckpt/${name}.pt exists - skipping pretraining)"; return 0; fi
  step python run_pretrain.py --name "$name" "$@"
}

controls() {
  # Control B: scratch with a 3x larger step budget (6000 vs 2000), best lr only.
  for seed in 0 1 2; do
    step python run_transfer.py --ckpts --steps 6000 --lrs 1e-3 --sizes 64 256 --seeds $seed \
         --out results/scratch_long.json
  done
  # Control A: a higher learning rate (3e-3) for scratch and pretrained.
  for seed in 0 1 2; do
    step python run_transfer.py --ckpts pre_all --steps 2000 --lrs 3e-3 --sizes 64 256 --seeds $seed \
         --out results/lr3e3.json
  done
}

diversity() {
  pretrain_once pre_heat --pdes heat
  pretrain_once pre_heat_advection --pdes heat advection
  for seed in 0 1 2; do
    step python run_transfer.py --ckpts pre_all pre_heat pre_heat_advection --sizes 64 256 --seeds $seed
  done
}

plots() {
  python plot_results.py --arms scratch pre_all --out-prefix results/main
  python plot_results.py --arms scratch pre_heat pre_heat_advection pre_all --out-prefix results/diversity
  python plot_results.py --arms scratch pre_all_n250 pre_all_n1000 pre_all pre_all_n4000 --out-prefix results/scale
  [ -f results/scratch_long.json ] && python plot_results.py --results results/scratch_long.json \
         --arms scratch --out-prefix results/scratch_long
  [ -f results/lr3e3.json ] && python plot_results.py --results results/lr3e3.json \
         --arms scratch pre_all --out-prefix results/lr3e3
  true
}

case "${1:-}" in
  controls) controls ;;
  diversity) diversity ;;
  plots) plots ;;
  *) echo "usage: bash run_chunked.sh {controls|diversity|plots}"; exit 1 ;;
esac
