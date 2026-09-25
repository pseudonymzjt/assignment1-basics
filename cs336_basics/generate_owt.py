# cs336_basics/generate_owt.py
import torch
from cs336_basics.tokenizer import tokenizer
from cs336_basics.decoding import decode
from cs336_basics.transformer import Transformer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载 OWT 训练好的 Tokenizer (35,000 词表)
    vocab_path = "output/owt_vocab.json"      # 如果你的文件名不同，请相应微调
    merges_path = "output/owt_merges.json"
    print(f"Loading OWT Tokenizer from: {vocab_path}...")
    tok = tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"]
    )

    # 2. 按照与训练时完全一致的架构初始化模型
    # 注意：vocab_size 必须为 35000！
    print("Initializing Model (48M architecture)...")
    model = Transformer(
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000,
        weights=None,
        vocab_size=35000,      # <-- 关键：改为 OWT 的 35000 词表
        context_length=256,
        num_layers=4,
    ).to(device)

    # 3. 加载刚刚跑完 40,000 步的最终权重
    ckpt_path = "checkpoints/owt_final_40k/ckpt_final.pt"
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 提取 state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # 去掉 torch.compile 可能带来的 '_orig_mod.' 前缀
    clean_state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 4. 设置生成提示词与超参数
    # 强烈建议保留 "Once upon a time" 或使用通用开场白
    # 这样可以与 TinyStories 的生成结果做最直观的“同 Prompt 苹果对苹果对比”！
    prompt = "Once upon a time"
    # prompt = "In recent years, the development of" # 备选互联网通用 Prompt
    
    max_tokens = 300       # 保证生成达到 256 tokens 以上（作业要求）
    temperature = 0.8      # 采样温度
    top_p = 0.95           # 核采样 (Top-p)

    eos_id = tok.encode("<|endoftext|>")[0] if tok.special_tokens else None

    print("\n" + "=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"Parameters: temperature={temperature}, top_p={top_p}")
    print("=" * 60 + "\n")

    # 5. 调用你的 decode 函数生成文本
    with torch.no_grad():
        generated_text = decode(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
            device=device,
        )

    # 统计生成的 Token 数并打印输出
    gen_tokens = tok.encode(generated_text)
    print(generated_text)
    print("\n" + "=" * 60)
    print(f"Generation Complete! Total tokens: {len(gen_tokens)}")
    print("=" * 60)

if __name__ == "__main__":
    main()