cd /root/autodl-tmp

python cs336_basics/train/training_loop.py \
    --train_data "output/owt_train.bin" \
    --val_data "output/owt_val.bin" \
    --vocab_size 35000 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 2048 \
    --context_length 512 \
    --batch_size 32 \
    --max_iters 50 \
    --log_interval 10 \
    --learning_rate 6e-4