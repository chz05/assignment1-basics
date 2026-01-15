in bpe.py, focus on merge part.

We need the three data structures in the self-> 
    - pair_counts: dict[tuple[bytes, bytes], int]: Count frequency of all successive pairs
    - pair_occur: dict[tuple[token, token], list[tuple[word_id, pos]]]: know which word and which position the pair occurs at
    - word_counts: dict[tuple[bytes], int]: ex: (l, o , w): 5... Store each “word” as a token sequence with a frequency

We need a function called first_merge_pair, it will create the value for the pair_counts, pair_occur, and word_counts.

Another function called merge_pair. This one will only update the pair_counts, pair_occur and word_counts. It does not need to go through whole word_counts again.

How to update? For example: (l, o) has the highest frequency. We need to first find in pair_occur that what's the original word and what's the position. We will find its neighbohood like (w, lo), we also need to update them. Then update the paircounts and word_counts.