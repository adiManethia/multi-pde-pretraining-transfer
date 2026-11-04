#!/usr/bin/env bash
# Full experiment (fine-tuning sizes N = 16, 64, 256). Stage 1 gives the main result; stage 2 adds the ablations.
set -e

# ---- Stage 1: main result (pretrain once, then transfer curve) ----
python run_pretrain.py --name pre_all --pdes heat advection reaction_diffusion
python run_transfer.py --ckpts pre_all
python plot_results.py --arms scratch pre_all --out-prefix results/main

# ---- Stage 2a: pretraining diversity ablation ----
python run_pretrain.py --name pre_heat --pdes heat
python run_pretrain.py --name pre_heat_advection --pdes heat advection
python run_transfer.py --ckpts pre_all pre_heat pre_heat_advection
python plot_results.py --arms scratch pre_heat pre_heat_advection pre_all --out-prefix results/diversity

# ---- Stage 2b: pretraining data-scale ablation ----
python run_pretrain.py --name pre_all_n250  --pdes heat advection reaction_diffusion --n-per-pde 250
python run_pretrain.py --name pre_all_n1000 --pdes heat advection reaction_diffusion --n-per-pde 1000
python run_pretrain.py --name pre_all_n4000 --pdes heat advection reaction_diffusion --n-per-pde 4000
python run_transfer.py --ckpts pre_all pre_all_n250 pre_all_n1000 pre_all_n4000
python plot_results.py --arms scratch pre_all_n250 pre_all_n1000 pre_all pre_all_n4000 --out-prefix results/scale
