# cs336_basics/generate.py
import torch
from cs336_basics.tokenizer import tokenizer
from cs336_basics.decoding import decode
from cs336_basics.transformer import Transformer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载训练好的 Tokenizer
    vocab_path = "output/ts_vocab.json"
    merges_path = "output/ts_merges.json"
    print("Loading Tokenizer...")
    tok = tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"]
    )

    # 2. 按照你的 __init__ 严格传入参数 (weights 传 None，后面通过 load_state_dict 加载)
    print("Initializing Model...")
    model = Transformer(
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000,
        weights=None,
        vocab_size=10000,
        context_length=256,
        num_layers=4,
    ).to(device)

    # 3. 加载训练好的最优权重
    ckpt_path = "checkpoints/best_tinystories/ckpt_final.pt"
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

    # 4. 设置生成提示词与超参数 (题目要求：至少 256 tokens)
    prompt = "Once upon a time"
    max_tokens = 400       # 保证生成达到 256 tokens 以上
    temperature = 0.8     # 0.7~0.8 既有创造力又保证句法连贯
    top_p = 0.999            # 核采样

    eos_id = tok.encode("<|endoftext|>")[0] if tok.special_tokens else None

    print("\n" + "=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"Parameters: temperature={temperature}, top_p={top_p}")
    print("=" * 60 + "\n")

    # 5. 调用你的 decode 函数
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

    # 统计生成的 Token 数
    gen_tokens = tok.encode(generated_text)
    print(generated_text)
    print("\n" + "=" * 60)
    print(f"Generation Complete! Total tokens: {len(gen_tokens)}")
    print("=" * 60)

if __name__ == "__main__":
    main()