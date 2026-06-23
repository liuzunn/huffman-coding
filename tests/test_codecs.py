"""
Integration tests for text & image encoders/decoders.
"""

import unittest
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoder import TextEncoder
from src.decoder import TextDecoder


class TestTextCodecRoundTrip(unittest.TestCase):
    """End-to-end encode -> decode round-trip for text."""

    def _roundtrip(self, text: str):
        with tempfile.NamedTemporaryFile(suffix='.huf', delete=False) as f:
            out_path = f.name
        try:
            encoder = TextEncoder()
            encoder.encode(text, out_path)

            decoder = TextDecoder()
            reconstructed = decoder.decode(out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

        return reconstructed

    def test_english(self):
        original = "Hello, Huffman Coding! This is a test of lossless compression."
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)

    def test_chinese(self):
        original = "哈夫曼编码是一种最优前缀码，广泛用于无损数据压缩。"
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)

    def test_mixed(self):
        original = "English 中文 Mix 混合 ✅ 12345"
        reconstructed = self._roundtrip(original)
        self.assertEqual(original, reconstructed)


if __name__ == '__main__':
    unittest.main()
