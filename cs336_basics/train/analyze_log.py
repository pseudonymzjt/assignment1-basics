# analyze_logs.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_file):
    """从jsonl文件加载metrics"""
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics

def plot_experiment(log_dir, save_path=None):
    """绘制实验曲线"""
    log_dir = Path(log_dir)
    metrics_file = log_dir / "metrics.jsonl"
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    metrics = load_metrics(metrics_file)
    
    # 提取数据
    train_data = [(m['step'], m['train/loss'], m.get('wall_time', 0)) 
                  for m in metrics if 'train/loss' in m]
    eval_data = [(m['step'], m['eval/train_loss'], m['eval/val_loss'], m.get('wall_time', 0))
                 for m in metrics if 'eval/val_loss' in m]
    
    if not train_data:
        print("No training data found")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Training loss vs steps (smoothed)
    steps, losses, _ = zip(*train_data)
    window = min(100, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        axes[0, 0].plot(smoothed_steps, smoothed, label='Train Loss (smoothed)', linewidth=2)
    axes[0, 0].plot(steps, losses, alpha=0.3, label='Train Loss (raw)')
    if eval_data:
        eval_steps, eval_train, eval_val, _ = zip(*eval_data)
        axes[0, 0].plot(eval_steps, eval_train, 'o-', label='Train Loss (eval)', markersize=4)
        axes[0, 0].plot(eval_steps, eval_val, 's-', label='Val Loss', markersize=4)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress: Loss vs Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Loss vs wall-clock time
    steps, losses, times = zip(*train_data)
    times = np.array(times)
    axes[0, 1].plot(times, losses, alpha=0.3)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smoothed_times = times[window-1:]
        axes[0, 1].plot(smoothed_times, smoothed, linewidth=2, label='Train Loss')
    if eval_data:
        eval_steps, eval_train, eval_val, eval_times = zip(*eval_data)
        eval_times = np.array(eval_times)
        axes[0, 1].plot(eval_times, eval_val, 's-', label='Val Loss', markersize=4)
    axes[0, 1].set_xlabel('Wall-clock Time (hours)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Progress: Loss vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning rate schedule
    lr_data = [(m['step'], m['train/lr']) for m in metrics if 'train/lr' in m]
    if lr_data:
        lr_steps, lrs = zip(*lr_data)
        axes[1, 0].plot(lr_steps, lrs)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Tokens per second
    tok_data = [(m['step'], m['train/tokens_per_sec']) 
                for m in metrics if 'train/tokens_per_sec' in m]
    if tok_data:
        tok_steps, tok_rates = zip(*tok_data)
        axes[1, 1].plot(tok_steps, tok_rates)
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Tokens/sec')
        axes[1, 1].set_title('Training Throughput')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 显示平均throughput
        avg_throughput = np.mean(tok_rates)
        axes[1, 1].axhline(y=avg_throughput, color='r', linestyle='--', 
                          label=f'Avg: {avg_throughput:.0f} tok/s')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.savefig(log_dir / 'analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot to {log_dir / 'analysis.png'}")
    
    plt.show()

def print_summary(log_dir):
    """打印实验摘要"""
    log_dir = Path(log_dir)
    
    # 加载配置
    config_file = log_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(f"Experiment: {config.get('experiment_name', 'unknown')}")
        print(f"\nModel Configuration:")
        print(f"  d_model: {config.get('d_model')}")
        print(f"  num_layers: {config.get('num_layers')}")
        print(f"  num_heads: {config.get('num_heads')}")
        print(f"  d_ff: {config.get('d_ff')}")
        print(f"  context_length: {config.get('context_length')}")
        print(f"\nTraining Configuration:")
        print(f"  batch_size: {config.get('batch_size')}")
        print(f"  learning_rate: {config.get('learning_rate')}")
        print(f"  max_iters: {config.get('max_iters')}")
    
    # 加载metrics
    metrics_file = log_dir / "metrics.jsonl"
    if metrics_file.exists():
        metrics = load_metrics(metrics_file)
        
        # 最终结果
        final_metrics = [m for m in metrics if 'final/' in str(m.keys())]
        if final_metrics:
            final = final_metrics[-1]
            print(f"\nFinal Results:")
            print(f"  Train Loss: {final.get('final/train_loss', 'N/A'):.4f}")
            print(f"  Val Loss: {final.get('final/val_loss', 'N/A'):.4f}")
            print(f"  Total Time: {final.get('final/total_time_hours', 'N/A'):.2f} hours")
        
        # 最佳验证损失
        val_losses = [(m['step'], m['eval/val_loss']) 
                     for m in metrics if 'eval/val_loss' in m]
        if val_losses:
            best_step, best_loss = min(val_losses, key=lambda x: x[1])
            print(f"\nBest Validation Loss:")
            print(f"  Loss: {best_loss:.4f} at step {best_step}")
        
        # 平均throughput
        tok_rates = [m['train/tokens_per_sec'] 
                    for m in metrics if 'train/tokens_per_sec' in m]
        if tok_rates:
            print(f"\nTraining Throughput:")
            print(f"  Average: {np.mean(tok_rates):.0f} tokens/sec")
            print(f"  Min: {np.min(tok_rates):.0f} tokens/sec")
            print(f"  Max: {np.max(tok_rates):.0f} tokens/sec")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment logs")
    parser.add_argument("log_dir", type=str, help="Path to log directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    
    args = parser.parse_args()
    
    print_summary(args.log_dir)
    
    if args.plot:
        plot_experiment(args.log_dir, args.save)
