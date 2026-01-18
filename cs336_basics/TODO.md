in tokenizer.py. work on the encode function.


we need to do the following things:

1. split the special tokens in text and pretokensize. We still need to keep it. For example text = "a good cat is a eat<|endoftext|>", we should get a ["a", " good", " cat", " is" " a" " eat" "<|endoftext|>"], also you should tell me which indexes are special token.

2. we already done the pretokenize:
   1. then we should apply merges. For each pretokensize one: such as " good" => we get (' ', 'g', 'o', 'o', 'd'), they will become bytes. Then for each bytes, it will combine with the successive => ' g', it will search inside merges to check whether it has a same pair. If original one will be replaced as ' g'. Then it will also check ' go' inside the merge. If no, it will search 'oo'. like that ... If this one is special token, we just skip it, and it will become a bytes.
   2. after we get the merges one such as [' g', 'oo' ...], it should be become some ints by bytes_to_id variable.