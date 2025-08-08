#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7

vllm serve "Qwen/Qwen2.5-32B-Instruct" \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 2 \
  --port 8001
