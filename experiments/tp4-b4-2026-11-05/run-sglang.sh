#!/usr/bin/env bash
source /etc/bashrc >/dev/null 2>&1 || true
set -euo pipefail
root=${SPYRE_BENCH_ROOT:-/home/galbloch/itay-tp4-b4-20261105}
cd "$root/sglang-tp4"
export PATH="$root/bin:$root/sglang-env/bin:$PATH"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME=/home/galbloch HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SPYRE_DEVICES=0,1,2,3 AIU_WORLD_SIZE=4 SPYRE_NUM_CPUS=32
export SPYRE_UPDATE_THREAD_CONFIG=0 TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export TOKENIZERS_PARALLELISM=false SGLANG_FORCE_DEVICE_SPYRE=1 VLLM_PLUGINS=
export TORCHINDUCTOR_CACHE_DIR="$root/sglang-inductor-cache"
unset SPYRE_KERNEL_CACHE RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
mode=${1:-micro}
output="$root/$mode-sglang-tp4-b4.json"
case "$mode" in
    micro) model=ibm-ai-platform/micro-g3.3-8b-instruct-1b; extra=(--context-length 512 --max-new-tokens 2 --iterations 1 --log-level info) ;;
    micro-tp1)
        model=ibm-ai-platform/micro-g3.3-8b-instruct-1b
        export SPYRE_DEVICES=0 AIU_WORLD_SIZE=1
        output="$root/micro-sglang-tp1-b4.json"
        extra=(--tp-size 1 --context-length 512 --max-new-tokens 2 --iterations 1 --log-level info)
        ;;
    granite) model=ibm-granite/granite-3.3-8b-instruct; extra=() ;;
    gemma) model=google/gemma-4-26B-A4B-it; extra=() ;;
    *) printf 'Unknown model selection: %s\n' "$mode" >&2; exit 2 ;;
esac
printf 'START sglang %s %s\n' "$mode" "$(date -u +%FT%TZ)"
exec "$root/sglang-env/bin/python" -u experiments/tp4-b4-2026-11-05/sglang_batch_bench.py \
    --model "$model" --dataset-cache /home/galbloch/datasets/cais___mmlu \
    --output "$output" "${extra[@]}"
