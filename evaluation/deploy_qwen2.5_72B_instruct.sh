#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve "Qwen/Qwen2.5-72B-Instruct" \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 4 \
  --port 8001
