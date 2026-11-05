#!/usr/bin/env bash
set -euo pipefail
root=${SPYRE_BENCH_ROOT:-/home/galbloch/itay-tp4-b4-20261105}
cd "$root/sglang-tp4"
stage=${1:-all}
if [[ "$stage" != all && "$stage" != sglang && "$stage" != vllm ]]; then
    echo 'Usage: run-tp4-validation.sh [all|sglang|vllm] [sglang modes...]' >&2
    exit 2
fi
shift "$(( $# > 0 ? 1 : 0 ))"
if [[ "$stage" != vllm ]]; then
sglang_modes=("$@")
if [[ ${#sglang_modes[@]} == 0 ]]; then sglang_modes=(micro micro-tp1 granite gemma); fi
for mode in "${sglang_modes[@]}"; do
    log="$root/$mode-sglang-tp4-b4.log"
    if [[ "$mode" == micro-tp1 ]]; then log="$root/micro-sglang-tp1-b4.log"; fi
    printf 'START sglang %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    bash experiments/tp4-b4-2026-11-05/run-sglang.sh "$mode" > "$log" 2>&1
    printf 'PASS sglang %s %s\n' "$mode" "$(date -u +%FT%TZ)"
done
fi
if [[ "$stage" != sglang ]]; then
for name in granite gemma; do
    printf 'START vllm %s %s\n' "$name" "$(date -u +%FT%TZ)"
    (
        set +u
        source /etc/bashrc >/dev/null 2>&1 || true
        set -u
        export PATH="$root/bin:$root/spyre-inference/.venv/bin:/home/galbloch/trackh/venv/bin:$PATH"
        export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
        export HF_HOME=/home/galbloch HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
        export SPYRE_DEVICES=0,1,2,3 AIU_WORLD_SIZE=4 SPYRE_NUM_CPUS=32
        export SPYRE_UPDATE_THREAD_CONFIG=0 TORCHINDUCTOR_COMPILE_THREADS=1
        export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
        export TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=1
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export TORCHINDUCTOR_CACHE_DIR="$root/inductor-cache"
        unset SPYRE_KERNEL_CACHE RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE VLLM_PLUGINS SGLANG_FORCE_DEVICE_SPYRE
        cd "$root/spyre-inference"
        extra=()
        model=ibm-granite/granite-3.3-8b-instruct
        if [[ "$name" == gemma ]]; then model=google/gemma-4-26B-A4B-it; extra=(--text-only); fi
        "$root/spyre-inference/.venv/bin/python" -u "$root/sglang-tp4/experiments/tp4-b4-2026-11-05/vllm_batch_bench.py" \
            --model "$model" --dataset-cache /home/galbloch/datasets/cais___mmlu \
            --output "$root/$name-vllm-tp4-b4-matched.json" "${extra[@]}"
    ) > "$root/$name-vllm-tp4-b4-matched.log" 2>&1
    printf 'PASS vllm %s %s\n' "$name" "$(date -u +%FT%TZ)"
done
fi
printf 'COMPLETE %s\n' "$(date -u +%FT%TZ)"
