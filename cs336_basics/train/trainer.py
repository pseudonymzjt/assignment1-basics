from tests.adapters import run_train_bpe

'''
input_path:
data/owt_train.txt

vocab_size:
32000

special_tokens:
["<|endoftext|>"]
'''
# uv run python -m cProfile -o log/large_trainer.prof cs336_basics/train/trainer.py
# uv run python -c 'import pstats; pstats.Stats("log/trainer.prof").strip_dirs().sort_stats("cumulative").print_stats(50)' > log/trainer-profile.txt
# vocab, merges = run_train_bpe(input_path = 'data/TinyStoriesV2-GPT4-train.txt', vocab_size = 10000, special_tokens = ["<|endoftext|>"])

import base64
import json
import os


def save_bpe_model(vocab: dict, merges: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    # vocab: {id: bytes} -> {id: base64_string}
    vocab_serializable = {
        str(k): base64.b64encode(v).decode('ascii') 
        for k, v in vocab.items()
    }
    
    # merges: [(bytes, bytes)] -> [[base64, base64]]
    merges_serializable = [
        [base64.b64encode(a).decode('ascii'), 
         base64.b64encode(b).decode('ascii')]
        for a, b in merges
    ]
    
    with open(f"{save_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    
    with open(f"{save_dir}/merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"Model saved to {save_dir}/")

def load_bpe_model(save_dir: str):
    with open(f"{save_dir}/vocab.json", "r", encoding="utf-8") as f:
        vocab_serializable = json.load(f)
    
    with open(f"{save_dir}/merges.json", "r", encoding="utf-8") as f:
        merges_serializable = json.load(f)
    
    vocab = {
        int(k): base64.b64decode(v.encode('ascii'))
        for k, v in vocab_serializable.items()
    }
    
    merges = [
        (base64.b64decode(a.encode('ascii')), 
         base64.b64decode(b.encode('ascii')))
        for a, b in merges_serializable
    ]
    
    return vocab, merges

vocab, merges = run_train_bpe(input_path = 'data/TinyStoriesV2-GPT4-train.txt', vocab_size = 10000, special_tokens = ["<|endoftext|>"])
save_bpe_model(vocab, merges, 'output')