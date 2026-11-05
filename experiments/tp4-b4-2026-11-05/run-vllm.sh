#!/usr/bin/env bash
source /etc/bashrc >/dev/null 2>&1 || true
set -euo pipefail
root=/home/galbloch/itay-tp4-b4-20261105
cd "$root/spyre-inference"
export PATH="$root/bin:$PWD/.venv/bin:/home/galbloch/trackh/venv/bin:$PATH"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME=/home/galbloch HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SPYRE_DEVICES=0,1,2,3 AIU_WORLD_SIZE=4 SPYRE_NUM_CPUS=32
export SPYRE_UPDATE_THREAD_CONFIG=0 TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_WAIT_POLICY=PASSIVE
export TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_CACHE_DIR="$root/inductor-cache"
unset SPYRE_KERNEL_CACHE RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
bench="$root/sglang-spyre/experiments/tp4-b4-2026-11-05/vllm_batch_bench.py"
for spec in 'granite ibm-granite/granite-3.3-8b-instruct' 'gemma google/gemma-4-26B-A4B-it'; do
    read -r name model <<< "$spec"
    printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
    if .venv/bin/python -u "$bench" --model "$model" --dataset-cache /home/galbloch/datasets/cais___mmlu --output "$root/$name-vllm-tp4-b4.json" > "$root/$name-vllm-tp4-b4.log" 2>&1; then
        printf 'PASS %s\n' "$name"
    else
        rc=$?
        printf 'FAIL %s exit=%s\n' "$name" "$rc"
        exit "$rc"
    fi
done
printf 'COMPLETE %s\n' "$(date -u +%FT%TZ)"
