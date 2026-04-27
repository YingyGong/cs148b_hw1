import importlib.metadata
from .linear import Linear
from .embedding import Embedding
from .layernorm import LayerNorm
from .ffn import FFN
from .sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from .softmax import Softmax
from .attention import MultiHeadSelfAttention


__version__ = importlib.metadata.version("eecs148b_hw1")
