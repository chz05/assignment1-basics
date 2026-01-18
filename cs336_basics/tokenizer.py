import json
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
        This function should accept a tokenizer class that has a encode and decode method.
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        The merges and special tokens are already in vocab.
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        # special tokens are already in vocab, so we can get the ids from the bytes_to_id dictionary
        self.special_bytes = {s: s.encode("utf-8") for s in self.special_tokens}
        self.special_ids = {s: self.bytes_to_id[self.special_bytes[s]] for s in self.special_tokens}

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
        """ 
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))

        vocab = {token_id: token.encode("utf-8") for token, token_id in vocab_json.items()}
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [(m1.encode("utf-8"), m2.encode("utf-8")) for m1, m2 in merges]
        return cls(vocab, merges, special_tokens)

        
    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Split text, keeping special tokens and marking them."""
        if not self.special_tokens:
            return [(text, False)]
        # Prefer longer tokens first to handle overlaps like "<|endoftext|><|endoftext|>"
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in ordered) + ")"
        parts = re.split(pattern, text)
        result: list[tuple[str, bool]] = []
        for part in parts:
            if part == "":
                continue
            result.append((part, part in self.special_tokens))
        return result

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges using merge rank (lowest rank merges first)."""
        if len(tokens) < 2:
            return tokens

        def get_pairs(seq: list[bytes]) -> set[tuple[bytes, bytes]]:
            return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}

        pairs = get_pairs(tokens)
        while True:
            # find best pair by rank
            best = None
            best_rank = None
            for pair in pairs:
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = pair

            if best is None:
                break

            a, b = best
            merged: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            if len(tokens) < 2:
                break
            pairs = get_pairs(tokens)

        return tokens

    def encode(self, text: str) -> list[int]: 
        """Encode an input text into a sequence of token IDs."""
        token_ids: list[int] = []
        parts = self._split_with_special_tokens(text)
        for part, is_special in parts:
            if is_special:
                token_bytes = self.special_bytes[part]
                token_ids.append(self.bytes_to_id[token_bytes])
                continue

            for match in re.finditer(PAT, part):
                pretoken = match.group()
                byte_tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
                merged_tokens = self._apply_merges(byte_tokens)
                token_ids.extend(self.bytes_to_id[tok] for tok in merged_tokens)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: 
        """ 
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)
        
    
    def decode(self, ids: list[int]) -> str: 
        """ Decode a sequence of token IDs into text """
        raw = b"".join(self.vocab[id] for id in ids)
        
        return raw.decode("utf-8", errors="replace")
        

