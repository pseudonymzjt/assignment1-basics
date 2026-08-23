import json
import base64
import regex as re
from collections.abc import Iterable, Iterator

DEFAULT_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_r = {token: token_id for token_id, token in self.vocab.items()}
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}

        # 预构建 special token 相关结构
        self._special_pattern = None
        self._special_to_id = {}
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 这里依赖正确的 bytes 形式查找 token_id
            self._special_to_id = {
                tok: self.vocab_r[tok.encode("utf-8")] for tok in sorted_tokens
            }
            pattern = "(" + "|".join(re.escape(k) for k in sorted_tokens) + ")"
            self._special_pattern = re.compile(pattern)

        self._merge_cache: dict[bytes, tuple[bytes, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        从 Base64 格式的 vocab.json 与 merges.json 恢复 Tokenizer 实例
        """
        # load and decode Vocab in Base64
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            token_id = int(k)
            if isinstance(v, str):
                vocab[token_id] = base64.b64decode(v)
            elif isinstance(v, list):
                vocab[token_id] = bytes(v)
            else:
                vocab[token_id] = bytes(v)

        # load and decode Merges in Base64
        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges: list[tuple[bytes, bytes]] = []
        for p0, p1 in raw_merges:
            b0 = base64.b64decode(p0) if isinstance(p0, str) else bytes(p0)
            b1 = base64.b64decode(p1) if isinstance(p1, str) else bytes(p1)
            merges.append((b0, b1))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def get_special_tokens(self, reversed: bool = False) -> dict | None:
        if not self.special_tokens:
            return None
        if reversed:
            return {v: k for k, v in self._special_to_id.items()}
        return dict(self._special_to_id)

    def bpe_merge(self, piece: bytes) -> tuple[bytes, ...]:
        cached = self._merge_cache.get(piece)
        if cached is not None:
            return cached

        parts = [bytes([b]) for b in piece]
        ranks = self.ranks

        while len(parts) >= 2:
            min_rank = float("inf")
            min_idx = -1
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and r < min_rank:
                    min_rank = r
                    min_idx = i

            if min_idx == -1:
                break

            parts[min_idx] = parts[min_idx] + parts[min_idx + 1]
            del parts[min_idx + 1]

        result = tuple(parts)
        self._merge_cache[piece] = result
        return result

    def encode_ordinary(self, text: str, PAT: re.Pattern | str | None = None) -> list[int]:
        pattern = DEFAULT_PAT if PAT is None else (
            re.compile(PAT) if isinstance(PAT, str) else PAT
        )

        vocab_r = self.vocab_r
        parts_int: list[int] = []

        for match in pattern.finditer(text):
            piece_encoded = match.group().encode("utf-8")
            for part in self.bpe_merge(piece_encoded):
                parts_int.append(vocab_r[part])

        return parts_int

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_ordinary(text)

        tokens: list[int] = []
        chunks = self._special_pattern.split(text)

        for chunk in chunks:
            if not chunk:
                continue
            special_id = self._special_to_id.get(chunk)
            if special_id is not None:
                tokens.append(special_id)
            else:
                tokens.extend(self.encode_ordinary(chunk))
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        vocab = self.vocab
        try:
            byte_parts = [vocab[token_id] for token_id in token_ids]
        except KeyError as e:
            raise ValueError(f"Unknown token ID: {e.args[0]}") from e

        full_bytes = b"".join(byte_parts)
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)