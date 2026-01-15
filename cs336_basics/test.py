from bpe import BPE

bpe = BPE(100, ["<|endoftext|>"])

word_counts = {
    (b'l', b'o', b'w'): 5,
    (b'l', b'o', b'w', b'e', b'r'): 2,
    (b'w', b'i', b'd', b'e', b's', b't'): 3,
    (b'n', b'e', b'w', b'e', b's', b't'): 5,
}

bpe.first_merge_pair(word_counts)
print(bpe.pair_counts)
print(bpe.pair_occur)
print(bpe.words)
print(bpe.word_freqs)
bpe.merge_pair(word_counts)
bpe.merge_pair(word_counts)
bpe.merge_pair(word_counts)
bpe.merge_pair(word_counts)
bpe.merge_pair(word_counts)
print(bpe.vocab)