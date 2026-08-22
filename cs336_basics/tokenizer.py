import regex as re


class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_r = {token:token_id for token_id, token in self.vocab.items()}
        self.ranks = {self.merges[i]:i for i in range(len(self.merges))}

    def get_special_tokens(self, reversed = False):
        if self.special_tokens == None:
            return None
        elif reversed:
            return {self.vocab_r[special_token.encode("utf-8")]:special_token for special_token in self.special_tokens}
        else:
            return {special_token:self.vocab_r[special_token.encode("utf-8")] for special_token in self.special_tokens}

    def bpe_merge(self, piece: bytes) -> list[bytes]:
        parts = [bytes([b]) for b in piece]
        ranks = self.ranks

        while(len(parts) >= 2):
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            best_pair = min(pairs, key=lambda p: ranks.get(p, float('inf')))

            if best_pair not in ranks:
                break

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return parts

    def encode_ordinary(self, text: str, PAT = None) -> list[int]:
        if PAT == None:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        parts = []

        for token in re.finditer(PAT, text):
            piece = token.group()
            piece_encoded = piece.encode("utf-8")
            part = self.bpe_merge(piece_encoded)
            part_int = [self.vocab_r[part_ind] for part_ind in part]
            parts.extend(part_int)

        return parts


    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_ordinary(text)
        # transform into a dict for indexing and sorting convenience
        special_tokens = self.get_special_tokens()
        # sort to avoid short pieces matching first
        sorted_tokens = sorted(special_tokens.keys(), key=len, reverse=True)
        # for example (r"(<\|endoftext\|>|<\|padding\|>)")
        pattern = "(" + "|".join(re.escape(k) for k in sorted_tokens) + ")"
        
        # split against special patterns
        chunks = re.split(pattern, text)
        
        tokens = []
        for chunk in chunks:
            if not chunk:
                # filter empty chunks
                continue
            if chunk in special_tokens:
                # hit special token chunks
                tokens.append(special_tokens[chunk])
            else:
                # ordinary test chunks
                tokens.extend(self.encode_ordinary(chunk))
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        byte_parts = []
        # special_tokens_r = self.get_special_tokens(reversed=True)
        
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            # deprecated because of test logic
            # elif token_id in special_tokens_r:
            #     special_str = special_tokens_r[token_id]
            #     byte_parts.append(special_str.encode("utf-8"))
            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        full_bytes = b"".join(byte_parts)

        return full_bytes.decode("utf-8", errors="replace")

    from collections.abc import Iterable, Iterator

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        yield token_id by chunk
        '''
        for chunk in iterable:
            yield from self.encode(chunk)