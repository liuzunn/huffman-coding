"""
Text encoder using Huffman coding.

Compresses plain-text files (.txt) using character-level Huffman coding,
writes the result in the custom .huf binary format (see src/utils.py for
format spec), and reports detailed compression metrics.
"""

import os
from typing import Dict, Optional

from src.huffman import (
    build_frequency_table,
    build_huffman_tree,
    build_code_table,
    build_canonical_code_table,
    encode_data,
    compute_entropy,
    compute_avg_code_length,
    coding_efficiency,
    compression_ratio_from_tables,
)
from src.utils import (
    BitWriter,
    write_compressed_file,
    format_size,
)


class TextEncoder:
    """Encodes plain text with Huffman coding and writes a .huf file.

    Usage:
        encoder = TextEncoder()
        report = encoder.encode("input.txt", "output.huf")
        print(report["summary"])
    """

    def __init__(self):
        self.freq_table: Dict[int, int] = {}
        self.code_table: Dict[int, str] = {}
        self.entropy: float = 0.0
        self.avg_length: float = 0.0
        self.efficiency: float = 0.0
        self.ratio_vs_fixed: float = 0.0

    def encode(self, text: str, output_path: str,
               use_canonical: bool = True) -> Dict:
        """Encode a text string and write the compressed file.

        Args:
            text:            The plain-text content to encode.
            output_path:     Destination .huf file path.
            use_canonical:   Whether to use canonical Huffman codes (recommended).

        Returns:
            dict with keys: input_chars, unique_symbols, entropy, avg_length,
                            efficiency, compression_ratio, original_size,
                            compressed_size, summary
        """
        # 1. Frequency analysis
        self.freq_table = build_frequency_table(text)
        total_chars = len(text)

        # 2. Build Huffman tree
        tree = build_huffman_tree(self.freq_table)

        # 3. Build code table
        code_table = build_code_table(tree)

        # 4. Canonical conversion (optional)
        if use_canonical:
            self.code_table = build_canonical_code_table(code_table)
        else:
            self.code_table = code_table

        # 5. Encode to bit string
        bits_str = encode_data(text, self.code_table)

        # 6. Pack bits into bytes
        bw = BitWriter()
        bw.write_bits(bits_str)
        payload = bw.to_bytes()
        padding = bw.padding_bits

        # 7. Build symbol_lengths dict for the header
        symbol_lengths: Dict[int, int] = {
            sym: len(code) for sym, code in self.code_table.items()
        }

        # 8. Write compressed file
        compressed_size = write_compressed_file(
            output_path, symbol_lengths, payload, padding, total_chars
        )

        # 9. Compute metrics
        self.entropy = compute_entropy(self.freq_table, total_chars)
        self.avg_length = compute_avg_code_length(
            self.code_table, self.freq_table, total_chars
        )
        self.efficiency = coding_efficiency(self.entropy, self.avg_length)
        self.ratio_vs_fixed = compression_ratio_from_tables(
            self.freq_table, self.code_table
        )

        # Size in bits (approximate)
        original_bits = total_chars * 8  # assuming 8-bit characters
        compressed_bits = compressed_size * 8

        info = {
            "input_chars": total_chars,
            "unique_symbols": len(self.freq_table),
            "entropy": round(self.entropy, 4),
            "avg_length": round(self.avg_length, 4),
            "efficiency": round(self.efficiency * 100, 2),
            "compression_ratio": round(compression_ratio(original_bits, compressed_bits), 4),
            "ratio_vs_fixed": round(self.ratio_vs_fixed, 4),
            "original_size": original_bits // 8,
            "compressed_size": compressed_size,
            "original_size_str": format_size(original_bits // 8),
            "compressed_size_str": format_size(compressed_size),
        }

        # Build human-readable summary
        lines = [
            "=" * 60,
            "  Text Compression Report",
            "=" * 60,
            f"  Input characters : {total_chars}",
            f"  Unique symbols   : {len(self.freq_table)}",
            f"  Entropy H(X)     : {info['entropy']:.4f} bits/symbol",
            f"  Avg code length  : {info['avg_length']:.4f} bits/symbol",
            f"  Efficiency       : {info['efficiency']:.2f}%",
            f"  Original size    : {info['original_size_str']}",
            f"  Compressed size  : {info['compressed_size_str']}",
            f"  Compression ratio: {info['compression_ratio']:.4f}x",
            f"  vs fixed-length  : {info['ratio_vs_fixed']:.4f}x",
            "-" * 60,
        ]
        info["summary"] = "\n".join(lines)

        return info


# Import at module level for use in info dict
from src.utils import compression_ratio
