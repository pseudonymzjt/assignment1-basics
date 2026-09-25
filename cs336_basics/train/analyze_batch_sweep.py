# analyze_batch_sweep.py
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def collect_batch_experiments():
    """自动收集所有 batch size 实验结果"""
    experiments = {}
    
    # 查找所有以 bs_ 开头的实验目录（支持 log/bs_* 和 log/batch_sweep/bs_*）
    candidates = list(Path("log").glob("bs_*")) + list(Path("log/batch_sweep").glob("bs_*"))
    
    for exp_dir in sorted(candidates):
        if not exp_dir.is_dir():
            continue
        try:
            bs = int(exp_dir.name.split("bs_")[-1])
        except ValueError:
            continue

        metrics_file = exp_dir / "metrics.jsonl"
        val_losses = []
        train_losses = []
        context_length = 256
        
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
                    
                    step = m.get('step', m.get('iter', 0))
                    tokens = step * bs * context_length
                    wall_time = m.get('wall_time', 0)
                    
                    v_loss = m.get('eval/val_loss', m.get('val_loss', m.get('eval_loss')))
                    t_loss = m.get('train/loss', m.get('train_loss', m.get('loss')))
                    
                    if v_loss is not None:
                        val_losses.append((step, tokens, wall_time, v_loss))
                    if t_loss is not None:
                        train_losses.append((step, tokens, wall_time, t_loss))

        status = "COMPLETED" if val_losses else ("OOM/FAILED" if (exp_dir / "oom.flag").exists() else "INCOMPLETE")
        
        experiments[bs] = {
            'status': status,
            'val_losses': val_losses,
            'train_losses': train_losses,
            'best_val_loss': min(val_losses, key=lambda x: x[3])[3] if val_losses else float('inf'),
            'final_val_loss': val_losses[-1][3] if val_losses else float('inf'),
            'total_time': val_losses[-1][2] if val_losses else 0,
        }
        
    return experiments

def print_summary(experiments):
    print("\n" + "="*85)
    print("BATCH SIZE SWEEP SUMMARY")
    print("="*85)
    print(f"{'Batch Size':<12} {'Final Val Loss':<16} {'Best Val Loss':<16} {'Total Time (s)':<16} {'Status':<12}")
    print("-"*85)
    for bs in sorted(experiments.keys()):
        data = experiments[bs]
        if data['status'] == 'COMPLETED':
            print(f"{bs:<12} {data['final_val_loss']:<16.4f} {data['best_val_loss']:<16.4f} {data['total_time']:<16.1f} ✓ DONE")
        else:
            print(f"{bs:<12} {'N/A':<16} {'N/A':<16} {'N/A':<16} ✗ {data['status']}")
    print("="*85 + "\n")

def plot_batch_sweep(experiments, save_path="batch_size_comparison.png"):
    valid = {bs: exp for bs, exp in experiments.items() if exp['val_losses']}
    if not valid:
        print("No valid runs to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss vs Tokens Processed (数据效率)
    ax = axes[0, 0]
    for bs in sorted(valid.keys()):
        vals = valid[bs]['val_losses']
        tokens = [x[1] / 1e6 for x in vals]
        losses = [x[3] for x in vals]
        ax.plot(tokens, losses, marker='o', markersize=4, label=f'Batch={bs}')
    ax.set_xlabel('Tokens Processed (Millions)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Tokens Processed (Sample Efficiency)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Loss vs Wall-Clock Time (物理耗时效率)
    ax = axes[0, 1]
    for bs in sorted(valid.keys()):
        vals = valid[bs]['val_losses']
        times = [x[2] / 60.0 for x in vals]
        losses = [x[3] for x in vals]
        ax.plot(times, losses, marker='s', markersize=4, label=f'Batch={bs}')
    ax.set_xlabel('Wall-Clock Time (Minutes)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Wall-Clock Time (Time Efficiency)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Loss vs Gradient Steps (梯度步数)
    ax = axes[1, 0]
    for bs in sorted(valid.keys()):
        vals = valid[bs]['val_losses']
        steps = [x[0] for x in vals]
        losses = [x[3] for x in vals]
        ax.plot(steps, losses, marker='^', markersize=4, label=f'Batch={bs}')
    ax.set_xlabel('Gradient Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Gradient Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. 吞吐效率 (Throughput vs Batch Size)
    ax = axes[1, 1]
    bss, throughs = [], []
    for bs in sorted(valid.keys()):
        vals = valid[bs]['val_losses']
        if len(vals) >= 2 and vals[-1][2] > 0:
            total_toks = vals[-1][1]
            total_sec = vals[-1][2]
            bss.append(bs)
            throughs.append(total_toks / total_sec)
    
    if bss:
        ax.plot(bss, throughs, 'o-', color='purple', linewidth=2, markersize=8)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch Size (log scale)')
        ax.set_ylabel('Throughput (Tokens / Second)')
        ax.set_title('Hardware Saturation: Throughput vs Batch Size')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to: {save_path}")

if __name__ == "__main__":
    exps = collect_batch_experiments()
    print_summary(exps)
    plot_batch_sweep(exps)
