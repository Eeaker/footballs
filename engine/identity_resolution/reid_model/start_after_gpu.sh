#!/usr/bin/env bash
# Run only after the user has enabled the GPU.
set -euo pipefail
cd /root/autodl-tmp/football_reid_20260908
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs
.venv/bin/python -u train_adapt.py --run 2>&1 | tee runs/console.log
