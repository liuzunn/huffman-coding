"""
Unit tests for the core Huffman engine.
"""

import unittest
import math

# Add project root to path for imports
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.huffman import (
    HuffmanNode, build_frequency_table, build_huffman_tree,
    build_code_table, build_canonical_code_table, encode_data,
    decode_data, rebuild_tree_from_lengths,
    compute_entropy, compute_avg_code_length, coding_efficiency,
    theoretical_min_code_length, compression_ratio_from_tables,
)


class TestHuffmanNode(unittest.TestCase):
    """Test HuffmanNode class."""

    def test_create_leaf(self):
        node = HuffmanNode(symbol=65, freq=10.0)
        self.assertTrue(node.is_leaf())
        self.assertEqual(node.symbol, 65)
        self.assertEqual(node.freq, 10.0)

    def test_create_internal(self):
        left = HuffmanNode(symbol=65, freq=5.0)
        right = HuffmanNode(symbol=66, freq=3.0)
        parent = HuffmanNode(freq=8.0, left=left, right=right)
        self.assertFalse(parent.is_leaf())
        self.assertIsNone(parent.symbol)

    def test_compare(self):
        a = HuffmanNode(symbol=1, freq=5.0)
        b = HuffmanNode(symbol=2, freq=3.0)
        self.assertTrue(b < a)  # lower freq should sort first


class TestFrequencyTable(unittest.TestCase):
    """Test frequency table construction."""

    def test_string_input(self):
        freq = build_frequency_table("aabbbc")
        self.assertEqual(freq[ord('a')], 2)
        self.assertEqual(freq[ord('b')], 3)
        self.assertEqual(freq[ord('c')], 1)

    def test_empty_string(self):
        freq = build_frequency_table("")
        self.assertEqual(len(freq), 0)

    def test_bytes_input(self):
        freq = build_frequency_table(b"abc")
        self.assertEqual(freq[97], 1)

    def test_list_input(self):
        freq = build_frequency_table([1, 1, 2, 3])
        self.assertEqual(freq[1], 2)


class TestHuffmanTree(unittest.TestCase):
    """Test Huffman tree construction."""

    def test_single_symbol(self):
        freq = {65: 10}
        tree = build_huffman_tree(freq)
        self.assertTrue(tree.is_leaf())
        self.assertEqual(tree.symbol, 65)

    def test_two_symbols(self):
        freq = {65: 5, 66: 3}
        tree = build_huffman_tree(freq)
        # Tree should have 1 internal + 2 leaves = 3 nodes
        self.assertFalse(tree.is_leaf())
        self.assertIsNotNone(tree.left)
        self.assertIsNotNone(tree.right)

    def test_three_symbols(self):
        freq = {65: 5, 66: 3, 67: 2}
        tree = build_huffman_tree(freq)
        # Should have exactly 3 leaf symbols reachable
        code_table = build_code_table(tree)
        self.assertEqual(set(code_table.keys()), {65, 66, 67})


class TestCodeTable(unittest.TestCase):
    """Test code table generation."""

    def test_prefix_property(self):
        """Verify that no code is a prefix of another (prefix-free property)."""
        freq = build_frequency_table("hello world this is a test")
        tree = build_huffman_tree(freq)
        codes = build_code_table(tree)

        code_list = list(codes.values())
        for i, c1 in enumerate(code_list):
            for j, c2 in enumerate(code_list):
                if i != j:
                    self.assertFalse(c2.startswith(c1),
                                     f"Code {c1} is prefix of {c2}")

    def test_all_symbols_present(self):
        text = "the quick brown fox jumps over the lazy dog"
        freq = build_frequency_table(text)
        tree = build_huffman_tree(freq)
        codes = build_code_table(tree)
        self.assertEqual(set(codes.keys()), set(freq.keys()))

    def test_canonical_order(self):
        text = "aabbbcdddde"
        freq = build_frequency_table(text)
        tree = build_huffman_tree(freq)
        codes = build_code_table(tree)
        canonical = build_canonical_code_table(codes)

        # Codes with same length should be consecutive binary numbers
        by_length = {}
        for sym, code in canonical.items():
            l = len(code)
            by_length.setdefault(l, []).append(int(code, 2))

        for length, values in by_length.items():
            values.sort()
            expected = list(range(values[0], values[0] + len(values)))
            self.assertEqual(values, expected,
                             f"Canonical codes of length {length} not consecutive")


class TestEncodeDecode(unittest.TestCase):
    """Test end-to-end encode/decode round-trip."""

    def _roundtrip(self, text: str):
        freq = build_frequency_table(text)
        tree = build_huffman_tree(freq)
        codes = build_code_table(tree)
        encoded = encode_data(text, codes)
        decoded = decode_data(encoded, tree)
        reconstructed = ''.join(chr(s) for s in decoded)
        return reconstructed

    def test_roundtrip_ascii(self):
        original = "Hello, Huffman Coding!"
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)

    def test_roundtrip_unicode(self):
        original = "中文测试 Unicode 支持 ✅"
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)

    def test_roundtrip_long_text(self):
        original = "The quick brown fox jumps over the lazy dog. " * 50
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)

    def test_single_character(self):
        original = "AAAAA"
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)


class TestMetrics(unittest.TestCase):
    """Test information-theoretic metric calculations."""

    def test_entropy_uniform(self):
        """Uniform distribution should give maximal entropy."""
        freq = {0: 1, 1: 1, 2: 1, 3: 1}
        h = compute_entropy(freq)
        self.assertAlmostEqual(h, 2.0, places=4)

    def test_entropy_deterministic(self):
        """Single symbol should give zero entropy."""
        freq = {0: 100}
        h = compute_entropy(freq)
        self.assertAlmostEqual(h, 0.0, places=4)

    def test_coding_efficiency_perfect(self):
        self.assertAlmostEqual(coding_efficiency(1.0, 1.0), 1.0)

    def test_coding_efficiency_zero_avg_len(self):
        self.assertEqual(coding_efficiency(1.0, 0.0), 0.0)

    def test_fixed_length_small(self):
        self.assertEqual(theoretical_min_code_length(4), 2)


class TestRebuildFromLengths(unittest.TestCase):
    """Test tree reconstruction from code lengths."""

    def test_roundtrip_rebuild(self):
        text = "test data for tree reconstruction verification"
        freq = build_frequency_table(text)
        tree = build_huffman_tree(freq)
        codes = build_code_table(tree)

        symbol_lengths = {s: len(c) for s, c in codes.items()}
        rebuilt = rebuild_tree_from_lengths(symbol_lengths)
        rebuilt_codes = build_code_table(rebuilt)

        # Codes should have the same lengths as originals
        for s in codes:
            self.assertEqual(len(codes[s]), len(rebuilt_codes[s]),
                             f"Code length mismatch for symbol {s}")

        # Encode with rebuilt codes, decode with rebuilt tree
        encoded = encode_data(text, rebuilt_codes)
        decoded = decode_data(encoded, rebuilt)
        reconstructed = ''.join(chr(s) for s in decoded)
        self.assertEqual(text, reconstructed)


if __name__ == '__main__':
    unittest.main()
