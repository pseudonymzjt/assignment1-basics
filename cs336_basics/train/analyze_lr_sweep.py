# analyze_lr_sweep.py
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def find_log_dir():
    """自动寻找项目根目录下的 log 文件夹"""
    candidates = [
        Path("log"),
        Path("logs"),
        Path(__file__).resolve().parent.parent.parent / "log",
        Path(__file__).resolve().parent / "log",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return Path("log")

def collect_all_experiments(log_base=None):
    """收集所有学习率实验的结果（包含正常收敛与发散实验）"""
    log_dir = Path(log_base) if log_base else find_log_dir()
    experiments = {}
    
    if not log_dir.exists():
        print(f"Warning: Log directory '{log_dir}' does not exist.")
        return experiments

    for exp_dir in sorted(log_dir.glob("lr_*")):
        if not exp_dir.is_dir():
            continue
            
        parts = exp_dir.name.split('_lr')
        if len(parts) < 2:
            continue
        lr_str = parts[-1]
        try:
            lr = float(lr_str.replace('e-0', 'e-').replace('e+0', 'e+'))
        except ValueError:
            continue

        metrics_file = exp_dir / "metrics.jsonl"
        diverged_file = exp_dir / "diverged.flag"
        
        train_losses = []
        val_losses = []
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # 兼容不同命名风格
                    t_loss = m.get('train/loss', m.get('train_loss', m.get('loss')))
                    v_loss = m.get('eval/val_loss', m.get('val_loss', m.get('eval_loss')))
                    step = m.get('step', m.get('iter', 0))
                    wall_time = m.get('wall_time', 0)
                    
                    if t_loss is not None:
                        train_losses.append((step, t_loss, wall_time))
                    if v_loss is not None:
                        val_losses.append((step, v_loss, wall_time))

        is_diverged = diverged_file.exists() or (len(val_losses) == 0 and not metrics_file.exists())
        
        if val_losses:
            experiments[lr] = {
                'status': 'COMPLETED',
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': min(val_losses, key=lambda x: x[1])[1],
                'final_val_loss': val_losses[-1][1],
            }
        else:
            experiments[lr] = {
                'status': 'DIVERGED' if is_diverged else 'NO_METRICS',
                'train_losses': train_losses,
                'val_losses': [],
                'best_val_loss': float('inf'),
                'final_val_loss': float('inf'),
            }
    
    return experiments

def print_summary(experiments):
    """打印摘要表"""
    print("\n" + "="*80)
    print("LEARNING RATE SWEEP SUMMARY")
    print("="*80)
    print(f"{'Learning Rate':<15} {'Best Val Loss':<15} {'Final Val Loss':<15} {'Status':<12}")
    print("-"*80)
    
    if not experiments:
        print("No experiments found!")
        print("="*80)
        return

    sorted_exp = sorted(experiments.items(), key=lambda x: x[0])
    
    for lr, data in sorted_exp:
        best = data['best_val_loss']
        final = data['final_val_loss']
        
        if data['status'] == 'DIVERGED':
            status = "⚡ DIVERGED"
            print(f"{lr:<15.2e} {'N/A':<15} {'N/A':<15} {status:<12}")
        elif data['status'] == 'NO_METRICS':
            status = "❓ INCOMPLETE"
            print(f"{lr:<15.2e} {'N/A':<15} {'N/A':<15} {status:<12}")
        else:
            status = "✓ PASS" if best <= 1.45 else "✗ FAIL"
            print(f"{lr:<15.2e} {best:<15.4f} {final:<15.4f} {status:<12}")
    
    print("="*80)
    
    passing = [(lr, data) for lr, data in experiments.items() if data['best_val_loss'] <= 1.45]
    if passing:
        print(f"\n✓ {len(passing)} learning rate(s) achieved target ≤ 1.45:")
        for lr, data in sorted(passing, key=lambda x: x[1]['best_val_loss']):
            print(f"  LR={lr:.2e}: Best Val Loss = {data['best_val_loss']:.4f}")
    else:
        print("\n✗ No learning rate achieved target ≤ 1.45")

def plot_lr_sweep(experiments, save_path="lr_sweep_comparison.png"):
    """绘制学习率对比图"""
    valid_exps = {k: v for k, v in experiments.items() if v['val_losses']}
    if not valid_exps:
        print("No valid evaluation metrics to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 验证损失曲线对比
    ax = axes[0, 0]
    for lr in sorted(valid_exps.keys()):
        val_losses = valid_exps[lr]['val_losses']
        steps, losses, _ = zip(*val_losses)
        ax.plot(steps, losses, label=f'LR={lr:.0e}', marker='o', markersize=3)
    ax.axhline(y=1.45, color='r', linestyle='--', label='Target (1.45)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 最终验证损失 vs 学习率
    ax = axes[0, 1]
    lrs = sorted(valid_exps.keys())
    final_losses = [valid_exps[lr]['final_val_loss'] for lr in lrs]
    ax.plot(lrs, final_losses, 'o-', markersize=8)
    ax.axhline(y=1.45, color='r', linestyle='--', label='Target (1.45)')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Val Loss vs Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 最佳验证损失 vs 学习率
    ax = axes[1, 0]
    best_losses = [valid_exps[lr]['best_val_loss'] for lr in lrs]
    ax.plot(lrs, best_losses, 's-', markersize=8, color='green')
    ax.axhline(y=1.45, color='r', linestyle='--', label='Target (1.45)')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Best Val Loss vs Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 训练损失曲线
    ax = axes[1, 1]
    sorted_by_best = sorted(valid_exps.items(), key=lambda x: x[1]['best_val_loss'])
    
    # 防越界选取具有代表性的 LR
    if len(sorted_by_best) == 1:
        representative = [sorted_by_best[0][0]]
    elif len(sorted_by_best) == 2:
        representative = [sorted_by_best[0][0], sorted_by_best[1][0]]
    else:
        representative = [sorted_by_best[0][0], sorted_by_best[len(sorted_by_best)//2][0], sorted_by_best[-1][0]]
    
    for lr in representative:
        if valid_exps[lr]['train_losses']:
            steps, losses, _ = zip(*valid_exps[lr]['train_losses'])
            window = min(100, max(1, len(losses) // 5))
            if len(losses) > window and window > 1:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
                ax.plot(smoothed_steps, smoothed, label=f'LR={lr:.0e}')
            else:
                ax.plot(steps, losses, label=f'LR={lr:.0e}')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss (Representative LRs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    experiments = collect_all_experiments()
    print_summary(experiments)
    plot_lr_sweep(experiments)