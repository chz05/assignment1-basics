from __future__ import annotations

import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPE:
    """
    BPE trainer core with an optimized (but node-free) incremental merge:
      - No global rebuild each merge
      - No per-occurrence (word_id, pos) pointers (which break after edits)
      - Uses per-word pair counts and a pair->words index, so each merge only touches affected words.

    Key structures:
      - words[wid] : current token list (bytes) for word wid
      - word_freqs[wid] : frequency of that word in the corpus
      - word_pair_counts[wid] : Counter[(a,b)] -> weighted frequency contribution from this word
      - pair_counts : global Counter[(a,b)] -> total weighted frequency
      - pair_to_words[(a,b)] : set of word ids that currently contain that pair
    """

    def __init__(self, vocab_size: int, special_tokens: List[str] = ["<|endoftext|>"]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # id -> bytes
        self.vocab: Dict[int, bytes] = {}
        self._init_vocab()

        # training state
        self.words: List[List[bytes]] = []
        self.word_freqs: List[int] = []

        # optimized merge tables
        self.word_pair_counts: List[Counter[Tuple[bytes, bytes]]] = []
        self.pair_counts: Counter[Tuple[bytes, bytes]] = Counter()
        self.pair_to_words: Dict[Tuple[bytes, bytes], set[int]] = defaultdict(set)

        self._initialized = False

    # ---------------- vocab init ----------------
    def _init_vocab(self) -> None:
        idx = 0
        for tok in self.special_tokens:
            self.vocab[idx] = tok.encode("utf-8")
            idx += 1
        for b in range(256):
            self.vocab[idx] = bytes([b])
            idx += 1

    # ---------------- merge init ----------------
    def init_merge(self, word_counts: Dict[Tuple[bytes], int]) -> None:
        """
        word_counts maps a "word" represented as a tuple of byte-tokens to an integer frequency.
        Example:
            {
              (b"l", b"o", b"w"): 5,
              (b"l", b"o", b"w", b"e", b"r"): 2,
              ...
            }
        """
        self.words.clear()
        self.word_freqs.clear()
        self.word_pair_counts.clear()
        self.pair_counts.clear()
        self.pair_to_words.clear()

        for wid, (word_tuple, freq) in enumerate(word_counts.items()):
            toks = list(word_tuple)
            self.words.append(toks)
            self.word_freqs.append(int(freq))

            c = Counter()
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                c[p] += freq

            self.word_pair_counts.append(c)

            for p, cnt in c.items():
                self.pair_counts[p] += cnt
                self.pair_to_words[p].add(wid)

        self._initialized = True

    # ---------------- best pair ----------------
    def _best_pair(self) -> Tuple[bytes, bytes]:
        """
        Return the most frequent pair.
        Tie-break: lexicographically greater pair wins (as in your spec/example).
        """
        if not self.pair_counts:
            raise ValueError("No pairs to merge")
        # max on (count, pair) where pair comparison gives lexicographic tie-break
        pair, _ = max(self.pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        return pair

    # ---------------- merge mechanics ----------------
    @staticmethod
    def _merge_word_tokens(toks: List[bytes], best_pair: Tuple[bytes, bytes]) -> List[bytes]:
        """
        Replace all non-overlapping occurrences of best_pair in toks by concatenation.
        Standard BPE does a left-to-right greedy merge within each word.
        """
        a, b = best_pair
        merged = a + b
        out: List[bytes] = []
        i = 0
        n = len(toks)
        while i < n:
            if i < n - 1 and toks[i] == a and toks[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        return out

    def _recount_word_pairs(self, toks: List[bytes], freq: int) -> Counter[Tuple[bytes, bytes]]:
        c = Counter()
        for i in range(len(toks) - 1):
            c[(toks[i], toks[i + 1])] += freq
        return c

    def _do_merge(self, best_pair: Tuple[bytes, bytes]) -> None:
        merged_token = best_pair[0] + best_pair[1]

        # add to vocab if space allows (optional – depends on your training loop)
        if len(self.vocab) < self.vocab_size:
            self.vocab[len(self.vocab)] = merged_token

        affected = list(self.pair_to_words.get(best_pair, set()))
        if not affected:
            # nothing contains the pair (shouldn't happen if indexes are correct)
            self.pair_counts.pop(best_pair, None)
            self.pair_to_words.pop(best_pair, None)
            return

        # Only rebuild per-word counts for affected words, and update globals incrementally.
        for wid in affected:
            freq = self.word_freqs[wid]

            # 1) remove old contribution for this word from globals
            old_c = self.word_pair_counts[wid]
            for p, cnt in old_c.items():
                # decrement global
                newv = self.pair_counts[p] - cnt
                if newv <= 0:
                    self.pair_counts.pop(p, None)
                else:
                    self.pair_counts[p] = newv

                # update membership index
                s = self.pair_to_words.get(p)
                if s is not None:
                    s.discard(wid)
                    if not s:
                        self.pair_to_words.pop(p, None)

            # 2) merge the word itself
            new_toks = self._merge_word_tokens(self.words[wid], best_pair)
            self.words[wid] = new_toks

            # 3) recompute this word's pair counts
            new_c = self._recount_word_pairs(new_toks, freq)
            self.word_pair_counts[wid] = new_c

            # 4) add new contribution to globals
            for p, cnt in new_c.items():
                self.pair_counts[p] += cnt
                self.pair_to_words[p].add(wid)

        # After processing affected words, best_pair might still exist if:
        # - the merge didn't eliminate all instances due to overlaps (rare) or if some words
        #   were not marked affected correctly. With this scheme it should be correct.
        if best_pair in self.pair_counts and self.pair_counts[best_pair] <= 0:
            self.pair_counts.pop(best_pair, None)
            self.pair_to_words.pop(best_pair, None)

    # ---------------- public API ----------------
    def merge_pair(self, word_counts: Dict[Tuple[bytes], int]) -> Tuple[Dict[Tuple[bytes], int], Tuple[bytes, bytes]]:
        """
        Merge the current best pair into a new token, updating internal state.

        Note: This returns the input `word_counts` unchanged because the internal optimized
        representation is kept in `self.words`. If your external interface requires returning
        an updated dict, you can add a method to export `self.words` back into counts.
        """
        if not self._initialized:
            self.init_merge(word_counts)

        best = self._best_pair()
        self._do_merge(best)
        return word_counts, best

