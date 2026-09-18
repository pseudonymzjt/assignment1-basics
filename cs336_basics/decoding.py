# cs336_basics/decoding.py
from typing import Optional

import torch
import torch.nn.functional as F


def sample_with_temperature(logits, temperature=1.0):
    """
    使用temperature scaling采样
    
    Args:
        logits: shape (vocab_size,) 的未归一化logits
        temperature: 温度参数，越小越确定性，越大越随机
    
    Returns:
        采样的token ID
    """
    if temperature == 0:
        # temperature为0时，取argmax
        return torch.argmax(logits).item()
    
    # 应用temperature scaling
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # 从分布中采样
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id


def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    """
    Top-p (nucleus) sampling
    
    Args:
        logits: shape (vocab_size,) 的未归一化logits
        top_p: nucleus sampling的概率阈值
        temperature: temperature scaling参数
    
    Returns:
        采样的token ID
    """
    # 应用temperature scaling
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过top_p的位置
    # 保留累积概率 <= top_p 的所有token
    sorted_indices_to_keep = cumulative_probs <= top_p
    
    # 至少保留一个token（概率最高的）
    sorted_indices_to_keep[0] = True
    
    # 过滤掉低概率的token
    filtered_probs = sorted_probs.clone()
    filtered_probs[~sorted_indices_to_keep] = 0.0
    
    # 重新归一化
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    # 从过滤后的分布中采样
    # 注意：我们在排序后的索引上采样
    sample_idx = torch.multinomial(filtered_probs, num_samples=1).item()
    
    # 映射回原始词汇表索引
    token_id = sorted_indices[sample_idx].item()
    
    return token_id


def decode(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> str:
    """
    从语言模型生成文本
    
    Args:
        model: Transformer语言模型
        tokenizer: BPE tokenizer
        prompt: 输入提示文本
        max_tokens: 最大生成token数
        temperature: temperature scaling参数（0表示贪婪解码）
        top_p: 如果不为None，使用top-p sampling
        eos_token_id: end-of-sequence token ID，如果为None则使用tokenizer的eos
        device: 设备
    
    Returns:
        生成的完整文本（包括prompt）
    """
    model.eval()
    
    # 编码prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # 如果没有指定eos_token_id，尝试从tokenizer获取
    if eos_token_id is None:
        # 假设<|endoftext|>是特殊token
        eos_token_id = tokenizer.encode("<|endoftext|>")[0] if hasattr(tokenizer, 'special_tokens') else None
    
    # 转换为tensor
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 获取当前序列长度
            seq_len = input_ids.size(1)
            
            # 如果序列超过context_length，只保留最后context_length个token
            if hasattr(model, 'context_length'):
                context_length = model.context_length
                if seq_len > context_length:
                    input_ids = input_ids[:, -context_length:]
            
            # 前向传播
            logits = model(input_ids)  # shape: (batch_size, seq_len, vocab_size)
            
            # 取最后一个位置的logits
            next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
            
            # 采样下一个token
            if top_p is not None:
                next_token = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature)
            else:
                next_token = sample_with_temperature(next_token_logits, temperature=temperature)
            
            # 检查是否生成了EOS token
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            # 添加到生成的序列
            generated_tokens.append(next_token)
            
            # 更新input_ids
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
    
    # 解码生成的token
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> list[str]:
    """
    批量生成文本
    
    Args:
        model: Transformer语言模型
        tokenizer: BPE tokenizer
        prompts: 输入提示文本列表
        max_tokens: 最大生成token数
        temperature: temperature scaling参数
        top_p: 如果不为None，使用top-p sampling
        eos_token_id: end-of-sequence token ID
        device: 设备
    
    Returns:
        生成的文本列表
    """
    # 简单实现：逐个生成（可以优化为真正的批处理）
    results = []
    for prompt in prompts:
        generated = decode(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            device=device,
        )
        results.append(generated)
    
    return results
