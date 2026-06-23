"""
Text decoder using Huffman coding.

Reads a .huf compressed file, reconstructs the Huffman tree from the
stored symbol-length table, decodes the bitstream, and returns the
original plain-text content.
"""

from typing import Dict, List

from src.huffman import (
    rebuild_tree_from_lengths,
    decode_data,
)
from src.utils import (
    read_compressed_file,
)


class TextDecoder:
    """Decodes a .huf compressed file back to plain text.

    Usage:
        decoder = TextDecoder()
        text = decoder.decode("input.huf")
        print(f"Recovered {len(text)} characters")
    """

    def __init__(self):
        self.symbol_lengths: Dict[int, int] = {}
        self.original_symbol_count: int = 0

    def decode(self, input_path: str) -> str:
        """Read a .huf file and decode it back to text.

        Args:
            input_path: Path to the .huf compressed file.

        Returns:
            Decoded plain-text string.
        """
        # 1. Read compressed file
        self.symbol_lengths, payload, padding, self.original_symbol_count = \
            read_compressed_file(input_path)

        # 2. Rebuild Huffman tree from symbol-length table
        tree = rebuild_tree_from_lengths(self.symbol_lengths)

        # 3. Convert packed bytes to bit string (trim padding)
        total_bits = len(payload) * 8 - padding
        bits = ''
        for i in range(total_bits):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)   # MSB first
            bits += '1' if (payload[byte_idx] >> bit_idx) & 1 else '0'

        # 4. Decode bits back to symbols
        symbols = decode_data(bits, tree)

        # 5. Convert int symbols back to characters
        text = ''.join(chr(s) for s in symbols)

        # 6. Truncate to expected length (safety check)
        if len(text) > self.original_symbol_count:
            text = text[:self.original_symbol_count]

        return text

    def decode_to_file(self, input_path: str, output_path: str,
                       encoding: str = 'utf-8') -> int:
        """Decode a .huf file and write the result to disk.

        Args:
            input_path:  Path to .huf file.
            output_path: Output .txt file path.
            encoding:    Encoding for the output text file.

        Returns:
            Number of characters written.
        """
        text = self.decode(input_path)
        with open(output_path, 'w', encoding=encoding) as f:
            f.write(text)
        return len(text)
