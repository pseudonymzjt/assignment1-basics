from tests.adapters import run_train_bpe

vocab, merges = run_train_bpe(input_path = 'tests/fixtures/corpus.en', vocab_size = 500, special_tokens = ["<|endoftext|>"])