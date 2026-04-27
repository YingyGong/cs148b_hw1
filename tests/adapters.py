from __future__ import annotations

import os
from typing import Any

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re
from collections import Counter

from eecs148b_hw1 import Linear, Embedding, LayerNorm, FFN, SinusoidalPositionalEncoding, Softmax, MultiHeadSelfAttention

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    model = Linear(in_features=d_in, out_features=d_out)
    model.load_state_dict({"W": weights})
    model.eval()
    with torch.no_grad():
        return model(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    model = Embedding(vocab_size, d_model)
    model.load_state_dict({"E":weights})
    model.eval()
    with torch.no_grad():
        return model(token_ids)


def run_ffn(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a 2-layer ReLU FFN, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Inner dimensionality of the feed-forward network.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for the first linear layer.
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for the second linear layer.
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # ffn.load_state_dict({"fc1.weight": w1_weight, "fc2.weight": w2_weight})
    # You can also manually assign the weights
    # ffn.fc1.weight.data = w1_weight
    # ffn.fc2.weight.data = w2_weight
    model = FFN(d_model, d_ff)
    model.load_state_dict({"w1.W": w1_weight, "w2.W": w2_weight})
    model.eval()
    with torch.no_grad():
        return model(in_features)



def run_layernorm(
    d_model: int,
    eps: float,
    weight: Float[Tensor, " d_model"],
    bias: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the affine parameters of LayerNorm, return the output of running
    LayerNorm on the input features.

    Args:
        d_model (int): The dimensionality of the LayerNorm input.
        eps (float): A value added to the denominator for numerical stability.
        weight (Float[Tensor, "d_model"]): LayerNorm scale parameters.
        bias (Float[Tensor, "d_model"]): LayerNorm bias parameters.
        in_features (Float[Tensor, "... d_model"]): Input features to run LayerNorm on.

    Returns:
        Float[Tensor, "... d_model"]: Tensor with the output of running LayerNorm on `in_features`.
    """
    model = LayerNorm(d_model=d_model, eps=eps)
    model.load_state_dict({'g': weight, 'b': bias})
    model.eval()
    with torch.no_grad():
        return model(in_features)


def run_sinusoidal_pe(
    d_model: int,
    max_seq_len: int,
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """Return sinusoidal positional embeddings for the given token positions."""
    model = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
    return model(token_positions)
    

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scores = Q @ K.transpose(-1, -2)
    scale = torch.sqrt(torch.tensor(Q.shape[-1]))
    normalized_scores = scores / scale
    if mask is not None:
        normalized_scores = normalized_scores.masked_fill(~mask, float("-inf"))
    attn_weights = run_softmax(normalized_scores, dim=-1)
    return attn_weights @ V

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # print(q_proj_weight.shape)
    # print("Debug---------------")
    # print(num_heads, d_model)
    # w_q = q_proj_weight.repeat(num_heads, 1)
    # w_k = k_proj_weight.repeat(num_heads, 1)
    # w_v = v_proj_weight.repeat(num_heads, 1)
    # w_o = o_proj_weight.repeat(1, num_heads)
    
    w_q = q_proj_weight
    w_k = k_proj_weight
    w_v = v_proj_weight
    w_o = o_proj_weight    
    
    model = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    model.load_state_dict({"w_q.W": w_q, "w_k.W": w_k, "w_v.W": w_v, "w_o.W": w_o})
    model.eval()
    with torch.no_grad():
        return model(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use LayerNorm and the 2-layer ReLU FFN.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln1.bias`
                Bias of affine transform for the first LayerNorm.
                Shape is (d_model,).
            - `ffn.fc1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.fc2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln2.bias`
                Bias of affine transform for the second LayerNorm.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ln1.bias`
                Bias of affine transform for the first LayerNorm.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.fc1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.fc2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ln2.bias`
                Bias of affine transform for the second LayerNorm.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for LayerNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `ln_final.bias`
                Bias of affine transform for the final LayerNorm.
                Shape is (d_model,).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    model = Softmax(dim=dim)
    model.eval()
    with torch.no_grad():
        return model(in_features)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
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
