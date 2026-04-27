import regex as re
from collections import Counter


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str]):
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # split on special tokens
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(f"({pattern})", text)

    # initialize
    count_pretokens = Counter()
    count_pairs = Counter()
    vocab = dict()
    # pretoken_set = {}
    cur_size = 0
    merges = []

    # pre-tokenize
    PAT = r"""’(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for part in parts:
        if part in special_tokens and part not in vocab.values():
            vocab[cur_size] = part.encode("utf-8") 
            cur_size += 1
            continue
        for match in re.finditer(PAT, part):
            token = match.group(0)
            # first see this token
            if count_pretokens[token] == 0:
                vocab[cur_size] = token.encode("utf-8") 
                cur_size += 1
            count_pretokens[token] += 1
            
    assert(cur_size <= vocab_size)

    def adjacent_pairs(seq):
        return [(seq[i], seq[i+1]) for i in range(len(seq) - 1)]
    
    for pretoken, count in count_pretokens.items():
        token_bytes = pretoken.encode("utf-8")
        for pair in adjacent_pairs(token_bytes):
            count_pairs[pair] += count
    
    while cur_size < vocab_size:
        best_pair = max(count_pairs, key=count_pairs.get)
        merges.append(best_pair)

        best_count = count_pairs[best_pair]
        new_token = f"{best_pair[0]}{best_pair[1]}"
        vocab[cur_size] = new_token.encode("utf-8") 

        token_parts = set(best_pair)
        for pair in count_pairs.keys():
            if set(pair) & token_parts:
                count_pairs[pair] = 0
        # count_pairs[] # new pair with other char





        





    pass