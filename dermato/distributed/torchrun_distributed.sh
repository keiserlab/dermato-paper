#!/bin/bash
HOST=127.0.0.1
PORT=38242
NNODES=1
NPROC=3

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,3 \
uv run torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --rdzv_endpoint=$HOST:$PORT src/saliency_overlap_v2.py