"""
Core Huffman coding engine.

Provides:
- Huffman tree construction via minimum-heap (priority queue)
- Code table generation (recursive traversal)
- Canonical Huffman code conversion
- Symbol encoding / tree-based decoding
- Information-theoretic metrics (entropy, avg length, efficiency)

This module is the algorithmic foundation for all compression modes.
"""

import math
import heapq
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


class HuffmanNode:
    """Node in a Huffman binary tree.

    Attributes:
        symbol: The symbol value (int representation of a character or pixel).
                None for internal (non-leaf) nodes.
        freq:   Frequency (or probability weight) of the symbol or subtree.
        left:   Left child node (bit 0).
        right:  Right child node (bit 1).
    """

    __slots__ = ('symbol', 'freq', 'left', 'right')

    def __init__(self, symbol: Optional[int] = None, freq: float = 0.0,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: 'HuffmanNode') -> bool:
        """Heap排序依据：频率越小优先级越高；频率相同时符号越小优先级越高。"""
        if self.freq != other.freq:
            return self.freq < other.freq
        # 频率相同时用id保持确定性排序
        return id(self) < id(other)

    def is_leaf(self) -> bool:
        """Check whether this node is a leaf (has no children)."""
        return self.left is None and self.right is None

    def __repr__(self) -> str:
        if self.is_leaf():
            ch = chr(self.symbol) if 32 <= self.symbol < 127 else f'\\x{self.symbol:02x}'
            return f"HuffmanNode({ch!r}, freq={self.freq})"
        return f"HuffmanNode(int, freq={self.freq})"


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------

def build_frequency_table(data: Union[str, List[int], bytes, List[str]]) -> Dict[int, int]:
    """Build a symbol-to-frequency mapping from input data.

    Args:
        data: Input data as str (characters mapped via ord()), bytes (direct),
              or a list of int symbols.

    Returns:
        dict mapping each symbol (int) to its frequency (int).
    """
    if isinstance(data, str):
        counter: Counter = Counter(ord(c) for c in data)
    elif isinstance(data, bytes):
        counter = Counter(data)
    elif isinstance(data, list):
        if data and isinstance(data[0], str):
            counter = Counter(ord(c) for c in data)  # type: ignore
        else:
            counter = Counter(data)  # type: ignore
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    return dict(counter)


# ---------------------------------------------------------------------------
# Huffman tree construction
# ---------------------------------------------------------------------------

def build_huffman_tree(freq_table: Dict[int, int]) -> HuffmanNode:
    """Construct an optimal Huffman tree using a min-heap (priority queue).

    Algorithm:
        1. Create a leaf node for each symbol.
        2. Repeatedly merge the two nodes with smallest frequency.
        3. The final single node is the root of the Huffman tree.

    Args:
        freq_table: Mapping from symbol -> frequency.

    Returns:
        Root node of the constructed Huffman tree.
    """
    if not freq_table:
        return HuffmanNode()

    # Edge case: single symbol -> create a degenerate tree
    if len(freq_table) == 1:
        symbol, freq = next(iter(freq_table.items()))
        return HuffmanNode(symbol=symbol, freq=freq)

    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq_table.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq,
                             left=left, right=right)
        heapq.heappush(heap, parent)

    return heap[0]


# ---------------------------------------------------------------------------
# Code table generation
# ---------------------------------------------------------------------------

def build_code_table(node: HuffmanNode, prefix: str = "",
                     code_table: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """Recursively traverse a Huffman tree to build the symbol-to-code mapping.

    Args:
        node:      Current tree node (start at root).
        prefix:    Accumulated binary code string (start with '').
        code_table: Mutable dict being populated (None auto-creates).

    Returns:
        Dict mapping each symbol (int) to its binary code string (e.g. '0101').
    """
    if code_table is None:
        code_table = {}

    if node.is_leaf():
        if node.symbol is not None:
            code_table[node.symbol] = prefix if prefix else "0"
    else:
        if node.left:
            build_code_table(node.left, prefix + "0", code_table)
        if node.right:
            build_code_table(node.right, prefix + "1", code_table)

    return code_table


def build_inverse_code_table(code_table: Dict[int, str]) -> Dict[str, int]:
    """Invert the code table for decoding: maps binary codes back to symbols."""
    return {code: sym for sym, code in code_table.items()}


def build_canonical_code_table(code_table: Dict[int, str]) -> Dict[int, str]:
    """Convert a plain Huffman code table to canonical Huffman form.

    Canonical Huffman codes have the important property that codes of the
    same length are consecutive binary integers. This allows compact header
    storage by only saving (symbol, code_length) pairs.

    Args:
        code_table: Original code table {symbol: binary_code_string}.

    Returns:
        Canonical code table with the same code lengths but reassigned values.
    """
    # Sort by (code_length, symbol)
    sorted_codes = sorted(code_table.items(), key=lambda x: (len(x[1]), x[0]))

    canonical_table: Dict[int, str] = {}
    current_code = 0
    current_length = 0

    for symbol, code in sorted_codes:
        length = len(code)
        # Shift left for each additional bit
        current_code <<= (length - current_length)
        canonical_table[symbol] = format(current_code, f'0{length}b')
        current_code += 1
        current_length = length

    return canonical_table


def rebuild_tree_from_lengths(symbol_lengths: Dict[int, int]) -> HuffmanNode:
    """Rebuild a Huffman tree from (symbol, code_length) pairs.

    This is used by the decoder to reconstruct the tree from the compact
    header format without needing the full code table.

    Args:
        symbol_lengths: Dict mapping each symbol to its code length in bits.

    Returns:
        Root of the reconstructed Huffman tree.
    """
    # Edge case: single symbol
    if len(symbol_lengths) == 1:
        symbol = next(iter(symbol_lengths.keys()))
        return HuffmanNode(symbol=symbol, freq=1.0)

    # First, build canonical codes from the lengths
    sorted_items = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))

    canonical: Dict[int, str] = {}
    code = 0
    cur_len = 0
    for sym, length in sorted_items:
        code <<= (length - cur_len)
        canonical[sym] = format(code, f'0{length}b')
        code += 1
        cur_len = length

    # Now build the tree: insert each code bit by bit
    root = HuffmanNode()
    for sym, code_str in canonical.items():
        node = root
        for bit in code_str:
            if bit == '0':
                if not node.left:
                    node.left = HuffmanNode()
                node = node.left
            else:
                if not node.right:
                    node.right = HuffmanNode()
                node = node.right
        node.symbol = sym
        node.freq = 0.0  # not needed for decoding

    return root


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------

def encode_data(data: Union[str, List[int], bytes],
                code_table: Dict[int, str]) -> str:
    """Encode an entire data sequence into a binary string using the code table.

    Args:
        data:       Input sequence (str, bytes, or list of ints).
        code_table: Mapping from symbol to binary code string.

    Returns:
        Concatenated binary string of all encoded symbols (e.g. '01001101...').
    """
    if isinstance(data, str):
        symbols = (ord(c) for c in data)
    elif isinstance(data, bytes):
        symbols = data
    else:
        symbols = data

    return ''.join(code_table[s] for s in symbols)


def decode_data(bits: str, root: HuffmanNode) -> List[int]:
    """Decode a binary string back to a sequence of symbols using the Huffman tree.

    Args:
        bits: Binary string of '0' and '1' characters.
        root: Root of the Huffman tree used during encoding.

    Returns:
        List of decoded symbol integers.
    """
    symbols: List[int] = []
    node = root

    # Edge case: single-symbol tree
    if root.is_leaf() and root.symbol is not None:
        return [root.symbol] * len(bits)

    for bit in bits:
        if bit == '0':
            node = node.left if node.left else node
        else:
            node = node.right if node.right else node

        if node.is_leaf():
            symbols.append(node.symbol)
            node = root

    return symbols


# ---------------------------------------------------------------------------
# Information-theoretic metrics
# ---------------------------------------------------------------------------

def compute_entropy(freq_table: Dict[int, int],
                    total: Optional[int] = None) -> float:
    """Compute first-order Shannon entropy H(X) = -Σ pᵢ log₂ pᵢ.

    Args:
        freq_table: Symbol frequency mapping.
        total:      Total count of symbols (computed if None).

    Returns:
        Entropy in bits-per-symbol.
    """
    if total is None:
        total = sum(freq_table.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for freq in freq_table.values():
        p = freq / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_avg_code_length(code_table: Dict[int, str],
                            freq_table: Dict[int, int],
                            total: Optional[int] = None) -> float:
    """Compute the average code length L̄ = Σ pᵢ × len(codeᵢ).

    Args:
        code_table: Symbol-to-code mapping.
        freq_table: Symbol frequency mapping.
        total:      Total count of symbols (computed if None).

    Returns:
        Average code length in bits-per-symbol.
    """
    if total is None:
        total = sum(freq_table.values())
    if total == 0:
        return 0.0

    total_weighted = 0.0
    for symbol, code in code_table.items():
        freq = freq_table.get(symbol, 0)
        total_weighted += (freq / total) * len(code)
    return total_weighted


def coding_efficiency(entropy: float, avg_code_length: float) -> float:
    """Compute coding efficiency η = H / L̄ (0 ≤ η ≤ 1).

    Args:
        entropy:        Shannon entropy of the source (bits/symbol).
        avg_code_length: Actual average code length (bits/symbol).

    Returns:
        Coding efficiency as a fraction; 1.0 = perfect.
    """
    if avg_code_length <= 0:
        return 0.0
    return entropy / avg_code_length


def theoretical_min_code_length(symbol_count: int) -> int:
    """Minimum number of bits needed for fixed-length coding."""
    if symbol_count <= 1:
        return 1
    return math.ceil(math.log2(symbol_count))


def compression_ratio_from_tables(freq_table: Dict[int, int],
                                   code_table: Dict[int, str]) -> float:
    """Compute the compression ratio achieved by Huffman vs. fixed-length coding.

    Ratio = (fixed_length_total) / (huffman_total)

    Returns:
        Compression ratio (>1 means Huffman compresses better).
    """
    total = sum(freq_table.values())
    n_symbols = len(freq_table)
    fixed_len = theoretical_min_code_length(n_symbols)
    fixed_total = total * fixed_len
    huffman_total = sum(freq_table[s] * len(code_table[s]) for s in freq_table)
    if huffman_total == 0:
        return 1.0
    return fixed_total / huffman_total
