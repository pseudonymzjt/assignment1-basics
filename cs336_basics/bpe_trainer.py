import heapq
import multiprocessing as mp
import os
from collections import Counter, defaultdict

import regex as re

DEFAULT_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTE_LOOKUP = tuple(bytes([i]) for i in range(256))
WHITESPACE_BYTES = b" \t\r\n"


class HeapItem:
    __slots__ = ("freq", "pair")

    def __init__(self, freq, pair):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.pair > other.pair


def _process_chunk_range_worker(args) -> dict[str, int]:
    file_path, start, end, special_pattern = args
    local_counter = Counter()
    BUFFER_SIZE = 16 * 1024 * 1024  # 16MB 流式微块

    with open(file_path, "rb") as f:
        f.seek(start)
        bytes_left = end - start
        buffer = b""

        while bytes_left > 0 or buffer:
            if bytes_left > 0:
                chunk = f.read(min(BUFFER_SIZE, bytes_left))
                bytes_left -= len(chunk)
                buffer += chunk

            if not buffer:
                break

            if bytes_left > 0:
                # seek strict dividing position
                split_idx = -1
                for idx in range(len(buffer) - 2, -1, -1):
                    if buffer[idx] == 10 and buffer[idx + 1] not in WHITESPACE_BYTES:
                        split_idx = idx + 1
                        break

                if split_idx != -1:
                    process_data = buffer[:split_idx]
                    buffer = buffer[split_idx:]
                else:
                    if bytes_left > 0:
                        continue
                    else:
                        process_data = buffer
                        buffer = b""
            else:
                process_data = buffer
                buffer = b""

            # normalize
            process_data = process_data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            text = process_data.decode("utf-8", errors="ignore")

            if special_pattern is not None:
                sub_chunks = special_pattern.split(text)
            else:
                sub_chunks = [text]

            for sc in sub_chunks:
                if sc:
                    local_counter.update(DEFAULT_PAT.findall(sc))

    return dict(local_counter)


class bpe_trainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens = None):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.vocab = {}
        self.merges = []
        self.word_freq_dict = defaultdict(int)
        self.pair_freq_dict = defaultdict(int)
        self.pair_to_words = defaultdict(set)

        self.special_pattern = None
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_pattern = "|".join(re.escape(k) for k in sorted_specials)
            self.special_pattern = re.compile(escaped_pattern)

    def _get_file_chunks(self, num_chunks: int):
        file_size = os.path.getsize(self.input_path)
        if file_size == 0:
            return []
        chunk_size = file_size // num_chunks
        chunks = []

        with open(self.input_path, "rb") as f:
            start = 0
            for i in range(num_chunks):
                if i == num_chunks - 1:
                    end = file_size
                else:
                    end = (i + 1) * chunk_size
                    f.seek(end)

                    while True:
                        line = f.readline()
                        if not line:
                            break
                        pos = f.tell()
                        next_byte = f.read(1)
                        f.seek(pos)
                        if not next_byte or next_byte not in WHITESPACE_BYTES:
                            break
                    end = f.tell()

                if start < end:
                    chunks.append((self.input_path, start, end))
                start = end
                if start >= file_size:
                    break
        return chunks

    def read(self, num_processes  = None):
        if self.vocab_size < 256 + len(self.special_tokens):
            raise ValueError(f"vocab size {self.vocab_size} should be larger than 256 + special tokens")

        self.vocab = {i: bytes([i]) for i in range(256)}
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            for idx, token in enumerate(sorted_specials, start=256):
                self.vocab[idx] = token.encode("utf-8")

        if num_processes is None:
            num_processes = os.cpu_count() or 4

        chunk_ranges = self._get_file_chunks(num_processes)
        tasks = [(self.input_path, start, end, self.special_pattern) for _, start, end in chunk_ranges]

        raw_global_word_counts = Counter()
        with mp.Pool(processes=num_processes) as pool:
            for local_counts in pool.imap_unordered(_process_chunk_range_worker, tasks, chunksize=1):
                raw_global_word_counts.update(local_counts)

        # the only tuplizing position
        for piece, count in raw_global_word_counts.items():
            piece_bytes = piece.encode("utf-8")
            parts_tuple = tuple(BYTE_LOOKUP[b] for b in piece_bytes)
            self.word_freq_dict[parts_tuple] = count

        for word_tuple, freq in self.word_freq_dict.items():
            for i in range(len(word_tuple) - 1):
                self.pair_freq_dict[(word_tuple[i], word_tuple[i + 1])] += freq

    def _build_initial_indices(self):
        self.words = []
        self.word_counts = []
        self.pair_to_words = defaultdict(set)

        for word_tuple, freq in self.word_freq_dict.items():
            word_id = len(self.words)
            self.words.append(list(word_tuple))
            self.word_counts.append(freq)

            for i in range(len(word_tuple) - 1):
                p = (word_tuple[i], word_tuple[i + 1])
                self.pair_to_words[p].add(word_id)

        self.heap = [
            HeapItem(freq, pair)
            for pair, freq in self.pair_freq_dict.items()
            if freq > 0
        ]
        heapq.heapify(self.heap)
        self.word_freq_dict.clear()

    def merge_pair(self):
        # pop max_pair
        pair = None
        while self.heap:
            top = heapq.heappop(self.heap)
            if self.pair_freq_dict.get(top.pair, 0) == top.freq:
                pair = top.pair
                break

        if pair is None:
            self.pair_freq_dict.clear()
            return

        p0, p1 = pair
        new_token = p0 + p1
        self.vocab[len(self.vocab)] = new_token
        self.merges.append(pair)

        affected_word_ids = list(self.pair_to_words.pop(pair, []))
        del self.pair_freq_dict[pair]

        pair_deltas = defaultdict(int)

        for wid in affected_word_ids:
            word = self.words[wid]
            freq = self.word_counts[wid]
            n = len(word)
            if n < 2:
                continue

            # greedily construct new words
            new_word = []
            i = 0
            while i < n:
                if i < n - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # extract old pair and new list
            old_pairs = [(word[k], word[k + 1]) for k in range(n - 1)]
            new_pairs = [(new_word[k], new_word[k + 1]) for k in range(len(new_word) - 1)]

            old_pairs_set = set(old_pairs)
            new_pairs_set = set(new_pairs)

            # discard discreetly
            for p in old_pairs_set:
                if p != pair and p not in new_pairs_set:
                    self.pair_to_words[p].discard(wid)

            # include carefully
            for p in new_pairs_set:
                if p not in old_pairs_set:
                    self.pair_to_words[p].add(wid)

            # calculate overlapped parts
            for p in old_pairs:
                if p != pair:
                    pair_deltas[p] -= freq

            for p in new_pairs:
                pair_deltas[p] += freq

            self.words[wid] = new_word

        # conclude and push
        for p, delta in pair_deltas.items():
            if delta == 0:
                continue
            new_cnt = self.pair_freq_dict.get(p, 0) + delta
            if new_cnt <= 0:
                self.pair_freq_dict.pop(p, None)
            else:
                self.pair_freq_dict[p] = new_cnt
                heapq.heappush(self.heap, HeapItem(new_cnt, p))

    def merge(self):
        if not hasattr(self, "heap"):
            self._build_initial_indices()

        while len(self.vocab) < self.vocab_size and self.pair_freq_dict:
            self.merge_pair()