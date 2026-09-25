#!/bin/bash
# batch_size_sweep.sh

# 基础模型超参数
TRAIN_DATA="output/ts_train.bin"
VAL_DATA="output/ts_val.bin"
VOCAB_SIZE=10000
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
CONTEXT_LENGTH=256

# 固定总 Token 预算：20,480,000 tokens (保证公平对比)
TOTAL_TOKENS=20480000

# 测试的 batch size 列表
# 从 1 到极限大（可先测到 256/512，看显卡显存是否支持）
batch_sizes=(1 4 16 64 128 256 512)

# 基于之前 lr sweep 结果的最佳基准 LR
BASE_LR="3e-3"

LOG_BASE_DIR="log/batch_sweep"
mkdir -p "$LOG_BASE_DIR"

echo "=========================================================="
echo "Starting Batch Size Variation Experiments"
echo "Target Tokens per Run: $TOTAL_TOKENS"
echo "=========================================================="

for bs in "${batch_sizes[@]}"
do
    exp_name="bs_${bs}"
    exp_dir="${LOG_BASE_DIR}/${exp_name}"
    ckpt_dir="checkpoints/batch_sweep/${exp_name}"

    # 动态计算对应的 steps: TOTAL_TOKENS / (bs * CONTEXT_LENGTH)
    max_iters=$(( TOTAL_TOKENS / (bs * CONTEXT_LENGTH) ))

    # 针对 batch=1 步数过多（80,000步）做上限截断保护（防止跑太久）
    if [ $max_iters -gt 25000 ]; then
        max_iters=25000
    fi

    # 预热步数设为总步数的 5% ~ 10%
    warmup_iters=$(( max_iters / 10 ))
    if [ $warmup_iters -lt 50 ]; then
        warmup_iters=50
    fi

    # 评估间隔自适应
    eval_interval=$(( max_iters / 10 ))
    if [ $eval_interval -lt 25 ]; then
        eval_interval=25
    fi

    echo ""
    echo "=========================================================="
    echo "Testing Batch Size: $bs | Max Iters: $max_iters | Warmup: $warmup_iters"
    echo "=========================================================="

    if [ -f "${exp_dir}/metrics.jsonl" ]; then
        echo "[SKIP] Experiment $exp_name already completed. Skipping..."
        continue
    fi

    rm -rf "$exp_dir" "$ckpt_dir"
    mkdir -p "$exp_dir" "$ckpt_dir"

    python cs336_basics/train/training_loop.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --vocab_size "$VOCAB_SIZE" \
        --d_model "$D_MODEL" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --d_ff "$D_FF" \
        --context_length "$CONTEXT_LENGTH" \
        --batch_size "$bs" \
        --max_iters "$max_iters" \
        --warmup_iters "$warmup_iters" \
        --cosine_cycle_iters "$max_iters" \
        --eval_interval "$eval_interval" \
        --learning_rate "$BASE_LR" \
        --checkpoint_dir "$ckpt_dir" \
        --experiment_name "$exp_name" 2>&1 | tee "${exp_dir}/train.log"

    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[OOM or CRASH] Batch size $bs failed with exit code $exit_code."
        echo "diverged_or_oom: true" > "${exp_dir}/oom.flag"
        # 如果达到显存上限 OOM 了，记录下来
        if grep -iq "CUDA out of memory" "${exp_dir}/train.log"; then
            echo ">> Reached GPU Memory Limit at Batch Size: $bs !"
        fi
    fi
done

echo "=========================================================="
echo "Batch Size Sweep Completed!"
echo "=========================================================="