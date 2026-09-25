#!/bin/bash
# run_norm_ablation.sh

# 1. 在前一步的最优学习率 3e-3 下，无 Norm 训练（预期：发生梯度爆炸/发散 Divergence）
python cs336_basics/train/training_loop.py \
    --train_data "output/ts_train.bin" \
    --val_data "output/ts_val.bin" \
    --vocab_size 10000 --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 --context_length 256 \
    --batch_size 32 --max_iters 4000 --warmup_iters 500 --cosine_cycle_iters 8000 \
    --learning_rate 3e-3 \
    --norm_type none \
    --experiment_name "ablation_no_norm_lr3e-3"

# 2. 探究更小的学习率是否能稳定（测试 1e-5 和 3e-5）
for lr in 1e-5 3e-5
do
    python cs336_basics/train/training_loop.py \
        --train_data "output/ts_train.bin" \
        --val_data "output/ts_val.bin" \
        --vocab_size 10000 --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 --context_length 256 \
        --batch_size 32 --max_iters 8000 --warmup_iters 500 --cosine_cycle_iters 8000 \
        --learning_rate $lr \
        --norm_type none \
        --experiment_name "ablation_no_norm_lr${lr}"
done