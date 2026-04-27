import regex as re
from collections import Counter

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    PAT = r"""’(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # split on special tokens
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]

    count_pretokens = Counter()

    # count only normal pretokens
    for part in parts:
        if part in special_tokens:
            continue
        for match in re.finditer(PAT, part):
            token = match.group(0)
            count_pretokens[token] += 1

    # current corpus representation: tuple[bytes, ...] -> count
    corpus = Counter()
    for token, count in count_pretokens.items():
        seq = tuple(bytes([b]) for b in token.encode("utf-8"))
        corpus[seq] += count

    # initial vocab: all single bytes + special tokens
    vocab = {}
    cur_size = 0

    for tok in special_tokens:
        vocab[cur_size] = tok.encode("utf-8")
        cur_size += 1

    for b in range(256):
        vocab[cur_size] = bytes([b])
        cur_size += 1

    merges = []

    def adjacent_pairs(seq):
        return zip(seq, seq[1:])

    def merge_sequence(seq, pair):
        merged = []
        i = 0
        new_sym = pair[0] + pair[1]
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                merged.append(new_sym)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return tuple(merged)

    while cur_size < vocab_size:
        count_pairs = Counter()
        for seq, count in corpus.items():
            for pair in adjacent_pairs(seq):
                count_pairs[pair] += count

        if not count_pairs:
            break

        best_pair = max(count_pairs, key=count_pairs.get)
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[cur_size] = new_token
        cur_size += 1

        new_corpus = Counter()
        for seq, count in corpus.items():
            new_seq = merge_sequence(seq, best_pair)
            new_corpus[new_seq] += count
        corpus = new_corpus

    return vocab, merges