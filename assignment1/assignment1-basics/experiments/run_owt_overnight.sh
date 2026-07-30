#!/usr/bin/env bash
# OWT overnight: Q0 tokenize → Q1/Q2 四卡并行（SwanLab → cs336-owt）
set -euo pipefail

ROOT=/data1/wcz/projects/Myllm/assignment1/assignment1-basics
RUNS=/data1/wcz/projects/Myllm-runs/experiments
TOK=/data1/wcz/datasets/myllm/tokenized
cd "$ROOT"

# 避免坏代理挡 SwanLab
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

echo "=== Q0 tokenize valid ==="
uv run --no-sync python scripts/run_tokenize.py --corpus owt --splits valid

echo "=== Q0 tokenize train（墙钟可能数小时）==="
uv run --no-sync python scripts/run_tokenize.py --corpus owt --splits train

test -f "$TOK/owt_train.npy" && test -f "$TOK/owt_valid.npy"
echo "=== Q0 done: $(ls -lh "$TOK"/owt_*.npy) ==="

# 公共 SwanLab：独立项目，四跑同 group 好对比
export SWANLAB_PROJ_NAME=cs336-owt
export SWANLAB_DESCRIPTION='CS336 A1 OpenWebText · Q1/Q2'
export SWANLAB_GROUP=owt-hparam
export SWANLAB_DATASET_TAG=owt
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

COMMON=(
  uv run --no-sync python scripts/run_train.py
  --train-data "$TOK/owt_train.npy"
  --valid-data "$TOK/owt_valid.npy"
  --vocab-size 32000
  --steps 20000
  --warmup 200
  --context-length 256
  --d-model 512
  --num-layers 4
  --num-heads 16
  --d-ff 1344
  --seed 0
)

launch() {
  local gpu="$1" run_id="$2" tags="$3" bs="$4" max_lr="$5" min_lr="$6"
  local out="$RUNS/$run_id"
  mkdir -p "$out"
  echo "=== launch GPU$gpu $run_id ==="
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SWANLAB_EXP_NAME="$run_id"
    export SWANLAB_TAGS="$tags"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
    "${COMMON[@]}" \
      --batch-size "$bs" \
      --max-lr "$max_lr" \
      --min-lr "$min_lr" \
      --out-dir "$out" \
      2>&1 | tee "$out/train_stdout.log"
    echo "EXIT=$?" | tee -a "$out/train_stdout.log"
  ) &
}

# GPU0 Q1 ba · GPU1–3 Q2 grid
launch 0 owt_ba_20k          "q1,ba-protocol,baseline" 32 3e-4 3e-5
launch 1 owt_lr6e4_20k       "q2,lr6e4"                32 6e-4 6e-5
launch 2 owt_bs64_20k        "q2,bs64"                 64 3e-4 3e-5
launch 3 owt_bs64_lr6e4_20k  "q2,bs64,lr6e4"           64 6e-4 6e-5

echo "=== Q1/Q2 four jobs launched; waiting ==="
wait
echo "=== all train jobs finished ==="
