# training_loop.py
import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.attention import cross_entropy
from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.optimizer import learning_rate_schedule
from cs336_basics.transformer import Transformer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for cloud logging.")


class ExperimentLogger:
    """实验日志记录器，支持本地文件和W&B"""
    
    def __init__(self, log_dir: str, use_wandb: bool = False, wandb_project: str | None = None, 
                 wandb_name: str | None = None, config: dict | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # 本地日志文件
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.log_file = self.log_dir / "train.log"
        
        # 内存中保存metrics用于绘图
        self.metrics_history = []
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "transformer-training",
                name=wandb_name,
                config=config
            )
        
        # 保存配置
        if config:
            with open(self.log_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
    
    def log(self, metrics: dict, step: int, commit: bool = True):
        """记录metrics"""
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        
        # 添加到历史记录
        self.metrics_history.append(metrics.copy())
        
        # 写入本地文件
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log(metrics, step=step, commit=commit)
    
    def log_text(self, text: str):
        """记录文本日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {text}\n"
        
        print(text)  # 也打印到控制台
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
    def save_plot(self):
        """保存loss曲线图"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                return
            
            # 提取数据
            train_losses = [(m['step'], m.get('train/loss')) 
                           for m in self.metrics_history if 'train/loss' in m]
            val_losses = [(m['step'], m.get('eval/val_loss')) 
                         for m in self.metrics_history if 'eval/val_loss' in m]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss vs steps
            if train_losses:
                steps, losses = zip(*train_losses)
                ax1.plot(steps, losses, label='Train Loss', alpha=0.7)
            if val_losses:
                steps, losses = zip(*val_losses)
                ax1.plot(steps, losses, label='Val Loss', marker='o')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss vs Training Steps')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss vs wall-clock time
            if train_losses:
                times = [m['wall_time'] for m in self.metrics_history if 'train/loss' in m]
                losses = [m['train/loss'] for m in self.metrics_history if 'train/loss' in m]
                ax2.plot(times, losses, label='Train Loss', alpha=0.7)
            if val_losses:
                times = [m['wall_time'] for m in self.metrics_history if 'eval/val_loss' in m]
                losses = [m['eval/val_loss'] for m in self.metrics_history if 'eval/val_loss' in m]
                ax2.plot(times, losses, label='Val Loss', marker='o')
            ax2.set_xlabel('Wall-clock Time (hours)')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss vs Wall-clock Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'loss_curves.png', dpi=150)
            plt.close()
            
            self.log_text("Saved loss curves to loss_curves.png")
        except ImportError:
            self.log_text("matplotlib not available, skipping plot generation")
    
    def finish(self):
        """结束日志记录"""
        self.save_plot()
        if self.use_wandb:
            wandb.finish()


def load_memmap_dataset(path, dtype=np.uint16):
    """使用memmap加载数据集"""
    print(f"Loading {path}...")
    data = np.memmap(path, dtype=dtype, mode='r')
    print(f"  Size: {len(data):,} tokens")
    print(f"  Range: [{data.min()}, {data.max()}]")
    return data


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, 
                  eval_iters, device):
    """评估损失"""
    model.eval()
    losses = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            total_loss += loss.item()
        
        losses[split] = total_loss / eval_iters
    
    model.train()
    return losses


def train(
    # 模型超参数
    vocab_size,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    context_length,
    rope_theta=10000.0,
    # 数据
    train_data_path="data/train.bin",
    val_data_path="data/val.bin",
    # 训练超参数
    batch_size=32,
    max_iters=100000,
    learning_rate=3e-4,
    weight_decay=0.1,
    warmup_iters=2000,
    cosine_cycle_iters=None,
    min_learning_rate=None,
    # 日志和保存
    eval_interval=500,
    eval_iters=100,
    save_interval=5000,
    log_interval=100,
    checkpoint_dir="./checkpoints",
    log_dir="./log",
    resume_from=None,
    device="cuda",
    # 实验跟踪
    use_wandb=False,
    wandb_project="transformer-training",
    experiment_name=None,
):
    """主训练循环"""
    torch.set_float32_matmul_precision('high')
    # 设置默认值
    if cosine_cycle_iters is None:
        cosine_cycle_iters = max_iters
    if min_learning_rate is None:
        min_learning_rate = learning_rate * 0.1
    if experiment_name is None:
        experiment_name = f"d{d_model}_l{num_layers}_h{num_heads}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建目录
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置字典
    config = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'd_ff': d_ff,
        'context_length': context_length,
        'rope_theta': rope_theta,
        'batch_size': batch_size,
        'max_iters': max_iters,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'warmup_iters': warmup_iters,
        'cosine_cycle_iters': cosine_cycle_iters,
        'min_learning_rate': min_learning_rate,
        'train_data_path': train_data_path,
        'val_data_path': val_data_path,
        'experiment_name': experiment_name,
    }
    
    # 初始化logger
    logger = ExperimentLogger(
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=experiment_name,
        config=config
    )
    
    logger.log_text("="*70)
    logger.log_text("Starting Training Experiment")
    logger.log_text(f"Experiment: {experiment_name}")
    logger.log_text("="*70)
    
    # 加载数据集（memory-mapped）
    logger.log_text("Loading datasets...")
    train_data = load_memmap_dataset(train_data_path, dtype=np.uint16)
    val_data = load_memmap_dataset(val_data_path, dtype=np.uint16)
    
    # 验证数据范围
    assert train_data.max() < vocab_size, \
        f"Train data max token {train_data.max()} >= vocab_size {vocab_size}"
    assert val_data.max() < vocab_size, \
        f"Val data max token {val_data.max()} >= vocab_size {vocab_size}"
    
    # 创建模型
    logger.log_text("Initializing model...")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        weights=None,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        norm_type=args.norm_type,         # <-- 传入
        norm_position=args.norm_position, # <-- 传入
        use_rope=(not args.no_rope),
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 恢复训练
    start_iter = 0
    if resume_from:
        start_iter = load_checkpoint(resume_from, model, optimizer)
        logger.log_text(f"Resumed from iteration {start_iter}")
    
    # 训练统计
    num_params = sum(p.numel() for p in model.parameters())
    logger.log_text(f"Model parameters: {num_params:,}")
    logger.log_text(f"Batch size: {batch_size}")
    logger.log_text(f"Context length: {context_length}")
    logger.log_text(f"Iterations: {start_iter} -> {max_iters}")
    logger.log_text(f"Learning rate: {learning_rate} -> {min_learning_rate}")
    logger.log_text(f"Device: {device}")
    logger.log_text("="*70)
    
    model.train()
    model = torch.compile(model)

    training_start_time = time.time()
    last_log_time = training_start_time
    
    for iteration in range(start_iter, max_iters):
        iter_start_time = time.time()
        
        # 获取学习率
        lr = learning_rate_schedule(
            iteration, 
            learning_rate, 
            min_learning_rate, 
            warmup_iters, 
            cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 获取batch
        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        
        # 前向传播
        logits = model(inputs)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 裁剪模型全部参数的梯度的 L2 范数，阈值设为 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # ==============================================================

        # 检测是否发散 (NaN / Inf)，遇到发散优雅退出
        if torch.isnan(loss) or torch.isinf(loss):
            logger.log_text(f"\n[DIVERGENCE DETECTED] Loss became {loss.item()} at iteration {iteration}! Exiting early.")
            break
        optimizer.step()
        
        # 计算时间
        iter_time = time.time() - iter_start_time
        elapsed = time.time() - training_start_time
        wall_time_hours = elapsed / 3600
        
        # 日志记录
        if iteration % log_interval == 0:
            tokens_per_sec = batch_size * context_length / iter_time
            
            log_msg = (f"iter {iteration:6d} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s | "
                      f"time {wall_time_hours:.2f}h")
            logger.log_text(log_msg)
            
            # 记录metrics
            logger.log({
                'train/loss': loss.item(),
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/iter_time': iter_time,
                'wall_time': wall_time_hours,
            }, step=iteration)
        
        # 评估
        if iteration % eval_interval == 0 and iteration > 0:
            logger.log_text("\n" + "="*70)
            logger.log_text(f"Evaluation at iteration {iteration}")
            
            eval_start = time.time()
            losses = estimate_loss(
                model, train_data, val_data, 
                batch_size, context_length, eval_iters, device
            )
            eval_time = time.time() - eval_start
            
            logger.log_text(f"  Train loss: {losses['train']:.4f}")
            logger.log_text(f"  Val loss:   {losses['val']:.4f}")
            logger.log_text(f"  Eval time:  {eval_time:.2f}s")
            logger.log_text("="*70 + "\n")
            
            # 记录评估metrics
            logger.log({
                'eval/train_loss': losses['train'],
                'eval/val_loss': losses['val'],
                'eval/eval_time': eval_time,
                'wall_time': wall_time_hours,
            }, step=iteration)
            
            # 保存图表
            logger.save_plot()
        
        # 保存checkpoint
        if iteration % save_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f"ckpt_iter_{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            logger.log_text(f"Saved checkpoint: {checkpoint_path}")
    
    # 保存最终模型
    final_path = checkpoint_dir / "ckpt_final.pt"
    save_checkpoint(model, optimizer, max_iters, final_path)
    
    total_time = time.time() - training_start_time
    logger.log_text("\n" + "="*70)
    logger.log_text("Training Complete!")
    logger.log_text(f"Total time: {total_time/3600:.2f} hours")
    logger.log_text(f"Final checkpoint: {final_path}")
    logger.log_text("="*70)
    
    # 最终评估
    logger.log_text("\nFinal Evaluation...")
    final_losses = estimate_loss(
        model, train_data, val_data, 
        batch_size, context_length, eval_iters, device
    )
    logger.log_text(f"Final Train Loss: {final_losses['train']:.4f}")
    logger.log_text(f"Final Val Loss:   {final_losses['val']:.4f}")
    
    # 记录最终metrics
    logger.log({
        'final/train_loss': final_losses['train'],
        'final/val_loss': final_losses['val'],
        'final/total_time_hours': total_time / 3600,
    }, step=max_iters)
    
    # 完成日志记录
    logger.finish()
    
    logger.log_text(f"\nExperiment logs saved to: {log_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    
    # 数据
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.bin)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data (.bin)")
    
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--min_learning_rate", type=float, default=None)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)
    
    # 日志和保存
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--resume_from", type=str, default=None)
    
    # 实验跟踪
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-training")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 消融实验
    parser.add_argument("--norm_type", type=str, default="rmsnorm", choices=["rmsnorm", "none"], help="Type of normalization")
    parser.add_argument("--norm_position", type=str, default="pre", choices=["pre", "post"], help="Position of normalization: pre or post")
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE (implement NoPE)")
    
    args = parser.parse_args()
    
    # 开始训练
    train(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        min_learning_rate=args.min_learning_rate,
        cosine_cycle_iters=args.cosine_cycle_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        experiment_name=args.experiment_name,
    )