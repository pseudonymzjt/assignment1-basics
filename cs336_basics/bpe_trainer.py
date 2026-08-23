import heapq
import os
from collections import Counter, defaultdict

import regex as re


class HeapItem:
    __slots__ = ("freq", "pair")

    def __init__(self, freq, pair):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other):
        # max(..., key=lambda x: (x[1], x[0]))
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.pair > other.pair

class bpe_trainer:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        # self.special_tokens_dict = {}
        self.pair_freq_dict = {}
        self.word_freq_dict = {}
        self.vocab = {}
        self.merges = []
        # self.vocab_r = {}
        # reuse code from tokenizer.py with small opt
        if self.special_tokens:
            self.sorted_tokens = sorted(self.special_tokens, key = len, reverse = True)
            self.pattern = "(" + "|".join(re.escape(k) for k in self.sorted_tokens) + ")"
        self.DEFAULT_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def read(self):
        # initialize vocab
        if self.vocab_size < 256 + len(self.special_tokens):
            raise ValueError(f"vocab size {self.vocab_size} should be larger than 256")
        self.vocab = {i:bytes([i]) for i in range(256)}
        cnt = 0
        if self.special_tokens:
            for special_token in sorted(self.special_tokens, key = len, reverse = True):
                self.vocab[256 + cnt] = special_token.encode("utf-8")
                # self.vocab_r[special_token.encode("utf-8")] = 256 + cnt
                # self.special_tokens_dict[special_token] = self.vocab[256 + cnt]
                cnt += 1

        # read and process texts
        with open(self.input_path, 'r', encoding = 'utf-8') as file:
            batch = []
            for i, line in enumerate(file, 1):
                batch.append(line)
                if i % 5000 == 0:
                    print(f'pretokenizing {i // 5000} * 5000 batches')
                    self.pre_tokenization(batch)
                    batch = []
            if batch:
                self.pre_tokenization(batch)

    def pre_tokenization_ord(self, text: str, PAT=None):
        pattern = self.DEFAULT_PAT if PAT is None else (re.compile(PAT) if isinstance(PAT, str) else PAT)
        # word freq quick counting
        piece_counts = Counter(m.group() for m in pattern.finditer(text))
        # all words processed at one time
        for piece, count in piece_counts.items():
            piece_bytes = piece.encode("utf-8")
            parts_tuple = tuple(bytes([b]) for b in piece_bytes)

            self.word_freq_dict[parts_tuple] = self.word_freq_dict.get(parts_tuple, 0) + count

            # adjacent pair += count
            n = len(parts_tuple)
            for i in range(n - 1):
                pair = (parts_tuple[i], parts_tuple[i + 1])
                self.pair_freq_dict[pair] = self.pair_freq_dict.get(pair, 0) + count

    def pre_tokenization(self, batch: list[str]):

        line = "".join(batch)
        if self.special_tokens:
            chunks = re.split(self.pattern, line)
            for chunk in chunks:
                if not chunk:
                    # filter empty chunks
                    continue
                if chunk not in self.sorted_tokens:
                    #     # hit special token chunks
                    #     # tokens.append(self.special_tokens_dict[chunk])
                    # else:
                    #     # ordinary test chunks
                    self.pre_tokenization_ord(chunk)
        else:
            self.pre_tokenization_ord(line)


    def _build_initial_indices(self):
        self.pair_to_words = defaultdict(set)
        for word in self.word_freq_dict:
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                self.pair_to_words[p].add(word)

        # build heap
        self.heap = [
            HeapItem(freq, pair)
            for pair, freq in self.pair_freq_dict.items()
            if freq > 0
        ]
        heapq.heapify(self.heap)

    def merge_pair(self):
        # take max pair
        pair = None
        while self.heap:
            top = heapq.heappop(self.heap)
            if top.pair in self.pair_freq_dict and self.pair_freq_dict[top.pair] == top.freq:
                pair = top.pair
                break

        if pair is None:
            self.pair_freq_dict.clear()
            return

        # update Vocab and Merges
        new_token = pair[0] + pair[1]
        self.vocab[len(self.vocab)] = new_token
        self.merges.append(pair)

        # remove merged pair
        affected_words = list(self.pair_to_words.pop(pair, []))
        del self.pair_freq_dict[pair]

        # locally update
        for old_word in affected_words:
            freq = self.word_freq_dict.pop(old_word, 0)
            if freq == 0:
                continue

            # break down
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                if p != pair:
                    new_cnt = self.pair_freq_dict[p] - freq
                    self.pair_to_words[p].discard(old_word)
                    if new_cnt <= 0:
                        del self.pair_freq_dict[p]
                    else:
                        self.pair_freq_dict[p] = new_cnt
                        heapq.heappush(self.heap, HeapItem(new_cnt, p))

            # local greedy merge
            new_word = []
            i = 0
            n = len(old_word)
            while i < n:
                if i < n - 1 and old_word[i] == pair[0] and old_word[i + 1] == pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            new_word = tuple(new_word)

            # write back
            self.word_freq_dict[new_word] = self.word_freq_dict.get(new_word, 0) + freq

            # register new freq
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                new_cnt = self.pair_freq_dict.get(p, 0) + freq
                self.pair_freq_dict[p] = new_cnt
                self.pair_to_words[p].add(new_word)
                heapq.heappush(self.heap, HeapItem(new_cnt, p))

    def merge(self):
        self._build_initial_indices()
        cnt = 0
        while len(self.vocab) < self.vocab_size and self.pair_freq_dict:
            self.merge_pair()
            if cnt % 1000 == 0:
                print(f'=============={cnt} iters==============')
                print(sorted(self.pair_freq_dict.items(), key=lambda x: (x[1], x[0]))[:10])
            cnt += 1