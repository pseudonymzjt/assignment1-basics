#!/bin/bash
# run_nope_ablation.sh

TRAIN_DATA="output/ts_train.bin"
VAL_DATA="output/ts_val.bin"
VOCAB_SIZE=10000
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
CONTEXT_LENGTH=256
BATCH_SIZE=32
MAX_ITERS=8000
WARMUP_ITERS=500
LR=3e-3

echo "=========================================================="
echo "Starting RoPE vs NoPE (Position Embedding Ablation)"
echo "=========================================================="

# 1. 运行 NoPE (无位置编码) 模型
echo "[1/2] Training NoPE model (without position embeddings)..."
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
    --learning_rate "$LR" \
    --no_rope \
    --experiment_name "ablation_nope_lr3e-3"

# 2. 运行 RoPE 基线模型 (如果已有 8000 步的 log/lr_sweep_lr3e-3 可以直接复用)
echo "[2/2] Training Baseline RoPE model..."
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
    --learning_rate "$LR" \
    --experiment_name "baseline_rope_lr3e-3"

echo "=========================================================="
echo "Experiments finished! Ready to plot RoPE vs NoPE curves."
echo "=========================================================="