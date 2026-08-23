from tests.adapters import run_train_bpe

'''
input_path:
data/TinyStoriesV2-GPT4-train.txt

vocab_size:
10000

special_tokens:
["<|endoftext|>"]
'''
# uv run python -m cProfile -o log/large_trainer.prof cs336_basics/train/trainer.py
# uv run python -c 'import pstats; pstats.Stats("log/trainer.prof").strip_dirs().sort_stats("cumulative").print_stats(50)' > log/trainer-profile.txt
vocab, merges = run_train_bpe(input_path = 'data/TinyStoriesV2-GPT4-train.txt', vocab_size = 10000, special_tokens = ["<|endoftext|>"])