# 1. 开始训练（本地日志）
python cs336_basics/train/training_loop.py \
  --train_data output/ts_train.bin \
  --val_data output/ts_val.bin \
  --vocab_size 10000 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --context_length 256 \
  --batch_size 16 \
  --max_iters 50000 \
  --experiment_name "default_tinystories"

# 2. 开始训练（使用W&B）
python train.py \
  --train_data data/train.bin \
  --val_data data/val.bin \
  --vocab_size 50257 \
  --d_model 768 \
  --num_layers 12 \
  --experiment_name "large_768d_12L" \
  --use_wandb \
  --wandb_project "my-transformer-project"

# 3. 分析实验结果
python cs336_basics/train/analyze_log.py log/default_tinystories --plot

# 4. 比较多个实验
python analyze_logs.py log/default_tinystories
python analyze_logs.py log/large_768d_12L