#!/bin/bash
# run_swiglu_ablation.sh

TRAIN_DATA="output/ts_train.bin"
VAL_DATA="output/ts_val.bin"
VOCAB_SIZE=10000
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
CONTEXT_LENGTH=256
BATCH_SIZE=32
MAX_ITERS=8000
WARMUP_ITERS=500
LR=3e-3

echo "=========================================================="
echo "Starting SwiGLU vs SiLU Ablation Experiment"
echo "=========================================================="

# 1. 运行 SiLU 基准模型 (d_ff = 4 * d_model = 2048)
echo "[1/2] Training SiLU FFN Model (d_ff=2048)..."
python cs336_basics/train/training_loop.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --vocab_size "$VOCAB_SIZE" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --d_ff 2048 \
    --ffn_type "silu" \
    --context_length "$CONTEXT_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --max_iters "$MAX_ITERS" \
    --warmup_iters "$WARMUP_ITERS" \
    --cosine_cycle_iters "$MAX_ITERS" \
    --learning_rate "$LR" \
    --experiment_name "ablation_silu_dff2048"

# 2. 运行 SwiGLU 对照模型 (d_ff = 1344)
# (如果前面 baseline_rope_lr3e-3 已经跑过 8000 步且就是默认的 SwiGLU，也可以直接复用)
echo "[2/2] Training SwiGLU Model (d_ff=1344)..."
python cs336_basics/train/training_loop.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --vocab_size "$VOCAB_SIZE" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --d_ff 1344 \
    --ffn_type "swiglu" \
    --context_length "$CONTEXT_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --max_iters "$MAX_ITERS" \
    --warmup_iters "$WARMUP_ITERS" \
    --cosine_cycle_iters "$MAX_ITERS" \
    --learning_rate "$LR" \
    --experiment_name "baseline_swiglu_dff1344"

echo "=========================================================="
echo "All done! Generating comparison plot..."
echo "=========================================================="