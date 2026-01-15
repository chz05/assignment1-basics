import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPE:
    def __init__(self, vocab_size: int, special_tokens: list[str] = ["<|endoftext|>"]):
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {}
        self.vocab_size = vocab_size
        self._init_vocab()

    def _init_vocab(self):
        """Initialize vocabulary with special tokens and 256 byte values."""
        idx = 0
        # Add special tokens first
        for token in self.special_tokens:
            self.vocab[idx] = token.encode("utf-8")
            idx += 1
        # Add 256 byte values
        for b in range(256):
            self.vocab[idx] = bytes([b])
            idx += 1

    def merge_pair(self, word_counts: dict[tuple[bytes, ...], int]) -> tuple[dict[tuple[bytes, ...], int], tuple[bytes, bytes]]:
        """Merge the most frequent pair in word_counts.
        
        Args:
            word_counts: {(b'h', b'e', b'l', b'l', b'o'): 5, ...}
        
        Returns:
            (new_word_counts, best_pair)
        """
        # Step 1: Count frequency of all successive pairs
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        
        # Step 2: Find best pair (highest frequency, lexicographically greater on tie)
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        
        # Step 3: Merge the best pair in all words
        merged_token = best_pair[0] + best_pair[1]
        new_word_counts: dict[tuple[bytes, ...], int] = {}
        
        for word, freq in word_counts.items():
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            new_word_counts[new_word_tuple] = new_word_counts.get(new_word_tuple, 0) + freq
        
        # Step 4: Add merged token to vocab
        self.vocab[len(self.vocab)] = merged_token
        
        return new_word_counts, best_pair
