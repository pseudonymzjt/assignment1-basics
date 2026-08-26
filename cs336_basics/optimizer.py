import math
from collections.abc import Callable, Iterable
from math import cos, pi
from typing import Optional

import torch


def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int):
    if t < t_w:
        return t / t_w * alpha_max
    elif t <= t_c:
        return alpha_min + 0.5 * (1 + cos((t - t_w) / (t_c - t_w) * pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    total_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in params if p.grad is not None)
    )
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    
    return total_norm

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                
                if param not in self.state:
                    self.state[param] = {
                        'm': torch.zeros_like(param),
                        'v': torch.zeros_like(param),
                        't': 0
                    }
                
                state = self.state[param]
                m, v, t = state['m'], state['v'], state['t']
                t += 1
                
                grad = param.grad
                
                # adjusted alpha (bias correction)
                alpha_t = lr * torch.sqrt(torch.tensor(1 - beta2 ** t)) / (1 - beta1 ** t)
                
                # weight decay
                param.data = param.data - lr * weight_decay * param.data
                
                # update m and v
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                
                # update params
                param.data = param.data - alpha_t * m / (torch.sqrt(v) + eps)
                
                # save state
                state['m'], state['v'], state['t'] = m, v, t


if __name__ == 'main':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    for lr in [1e1, 1e2, 1e3]:
        opt = SGD([weights], lr=lr)
        print(f"====== lr: {lr} ======")
        for t in range(100):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.