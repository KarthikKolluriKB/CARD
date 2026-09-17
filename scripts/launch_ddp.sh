#!/usr/bin/env bash
# Usage: bash scripts/launch_ddp.sh <config.yaml> [train.py arguments]
# Effective batch = batch_size x grad_accum x GPUs (paper: 4 x 8 x 2 = 64).

set -euo pipefail

CONFIG=${1:?"Usage: $0 <config.yaml> [train.py arguments]"}

export DDP_BACKEND=${DDP_BACKEND:-nccl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-2}" \
    --standalone \
    train.py \
    --config "${CONFIG}" \
    "${@:2}"
