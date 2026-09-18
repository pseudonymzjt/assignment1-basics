#!/bin/bash
# lr_sweep.sh - 高效扫参版

# 学习率列表：覆盖低速收敛、黄金区间、边缘稳定性和发散点
learning_rates=(1e-5 3e-5 1e-4 3e-4 6e-4 1e-3 3e-3 6e-3 1e-2)

# 模型与训练配置
TRAIN_DATA="output/ts_train.bin"
VAL_DATA="output/ts_val.bin"
VOCAB_SIZE=10000
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
CONTEXT_LENGTH=256

# 扫参优化参数：
BATCH_SIZE=32        # 从 16 提升到 32，大幅提升显卡利用率，吞吐翻倍
MAX_ITERS=20000       # 缩短为 20000 步（约 15~20 分钟一个 run）
WARMUP_ITERS=2000     # 2000 步预热
EVAL_INTERVAL=500    # 每 500 步评估一次验证集
LOG_INTERVAL=50      # 每 50 步打印一次日志

LOG_BASE_DIR="log"
mkdir -p "$LOG_BASE_DIR"

echo "=========================================================="
echo "Starting Fast LR Sweep (Max Iters: $MAX_ITERS, Batch Size: $BATCH_SIZE)"
echo "Target: Deliverables for 7.2.3 (Learning curves & Divergence)"
echo "=========================================================="

for lr in "${learning_rates[@]}"
do
    exp_name="lr_sweep_lr${lr}"
    exp_dir="${LOG_BASE_DIR}/${exp_name}"
    ckpt_dir="checkpoints/${exp_name}"
    
    echo ""
    echo "=========================================================="
    echo "Running: $exp_name (LR = $lr)"
    echo "=========================================================="

    # 如果已有完整跑完的有效结果（且达标），跳过
    if [ -f "${exp_dir}/metrics.jsonl" ]; then
        echo "[SKIP] $exp_name already has completed metrics. Skipping..."
        continue
    fi

    # 清理之前失败或被中断的不完整目录
    rm -rf "$exp_dir" "$ckpt_dir"
    mkdir -p "$exp_dir" "$ckpt_dir"

    # 启动训练：屏幕实时打印，并同步存入 train.log
    python cs336_basics/train/training_loop.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --vocab_size "$VOCAB_SIZE" \
        --d_model "$D_MODEL" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --d_ff "$D_FF" \
        --context_length "$CONTEXT_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --max_iters "$MAX_ITERS" \
        --warmup_iters "$WARMUP_ITERS" \
        --cosine_cycle_iters "$MAX_ITERS" \
        --eval_interval "$EVAL_INTERVAL" \
        --log_interval "$LOG_INTERVAL" \
        --learning_rate "$lr" \
        --checkpoint_dir "$ckpt_dir" \
        --experiment_name "$exp_name" 2>&1 | tee "${exp_dir}/train.log"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo "[DIVERGED] Run with LR=$lr diverged or crashed (Code: $exit_code)."
        echo "diverged: true" > "${exp_dir}/diverged.flag"
    else
        echo "[COMPLETED] Run with LR=$lr finished."
    fi
done

echo ""
echo "=========================================================="
echo "All sweep runs finished! Generating analysis plot..."
echo "=========================================================="

# 跑完自动生成对比图表
python cs336_basics/train/analyze_lr_sweep.py