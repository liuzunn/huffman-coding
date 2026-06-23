"""
Unit tests for bit-level I/O (BitWriter / BitReader).
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import BitWriter, BitReader


class TestBitWriter(unittest.TestCase):
    """Test the BitWriter class."""

    def test_write_single_bit(self):
        bw = BitWriter()
        bw.write_bit(1)
        bw.write_bit(0)
        bw.write_bit(1)
        data = bw.to_bytes()
        # Bits 101 -> byte 10100000 = 0xA0
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], 0b10100000)

    def test_write_exact_byte(self):
        bw = BitWriter()
        bits = '11001100'
        bw.write_bits(bits)
        data = bw.to_bytes()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], 0b11001100)

    def test_write_multi_byte(self):
        bw = BitWriter()
        bw.write_bits('1111000011001100')
        data = bw.to_bytes()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 0b11110000)
        self.assertEqual(data[1], 0b11001100)

    def test_write_uint8(self):
        bw = BitWriter()
        bw.write_uint8(0xAB)
        data = bw.to_bytes()
        self.assertEqual(data[0], 0xAB)

    def test_write_uint16(self):
        bw = BitWriter()
        bw.write_uint16(0x1234)
        data = bw.to_bytes()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 0x34)
        self.assertEqual(data[1], 0x12)

    def test_write_uint32(self):
        bw = BitWriter()
        bw.write_uint32(0xDEADBEEF)
        data = bw.to_bytes()
        self.assertEqual(len(data), 4)
        self.assertEqual(int.from_bytes(data, 'little'), 0xDEADBEEF)


class TestBitReader(unittest.TestCase):
    """Test the BitReader class."""

    def test_single_byte_read(self):
        data = bytes([0b10101010])
        br = BitReader(data)
        bits = br.read_bits(8)
        self.assertEqual(bits, '10101010')

    def test_read_partial_byte(self):
        data = bytes([0b11110000])
        br = BitReader(data)
        bits = br.read_bits(4)
        self.assertEqual(bits, '1111')

    def test_read_multi_byte(self):
        data = bytes([0b11110000, 0b10101010])
        br = BitReader(data)
        bits = br.read_bits(16)
        self.assertEqual(bits, '1111000010101010')

    def test_roundtrip(self):
        """Write bits, then read them back."""
        bw = BitWriter()
        test_pattern = '011010011001010010101101100001'
        bw.write_bits(test_pattern)
        packed = bw.to_bytes()

        br = BitReader(packed)
        total_bits = len(test_pattern)
        reconstructed = br.read_bits(total_bits)
        self.assertEqual(reconstructed, test_pattern)

    def test_read_uint8(self):
        data = bytes([0x42])
        br = BitReader(data)
        self.assertEqual(br.read_uint8(), 0x42)

    def test_read_uint16(self):
        data = b'\x34\x12'
        br = BitReader(data)
        self.assertEqual(br.read_uint16(), 0x1234)

    def test_read_uint32(self):
        data = b'\xEF\xBE\xAD\xDE'
        br = BitReader(data)
        self.assertEqual(br.read_uint32(), 0xDEADBEEF)

    def test_align(self):
        """Reading bits then aligning should skip to next byte boundary."""
        data = bytes([0b10100000, 0b11110000])
        br = BitReader(data)
        br.read_bits(3)  # read 101
        val = br.read_uint8()  # should skip to next byte
        self.assertEqual(val, 0b11110000)


class TestBitStreamRoundTrip(unittest.TestCase):
    """End-to-end BitWriter -> BitReader round-trip tests."""

    def test_variable_patterns(self):
        patterns = [
            '0',
            '1' * 20,
            '01' * 10,
            '00001111' * 3,
            ''.join(str(i % 2) for i in range(100)),
        ]
        for pat in patterns:
            bw = BitWriter()
            bw.write_bits(pat)
            data = bw.to_bytes()
            br = BitReader(data)
            reconstructed = br.read_bits(len(pat))
            self.assertEqual(reconstructed, pat,
                             f"Failed for pattern of length {len(pat)}")

    def test_mixed_types(self):
        """Write bits, uint8, uint16, bits; then read in order."""
        bw = BitWriter()
        bw.write_bits('110')
        bw.write_uint8(0xAB)
        bw.write_uint16(0x1234)
        bw.write_bits('01')
        data = bw.to_bytes()

        br = BitReader(data)
        bits1 = br.read_bits(3)
        self.assertEqual(bits1, '110')
        u8 = br.read_uint8()
        self.assertEqual(u8, 0xAB)
        u16 = br.read_uint16()
        self.assertEqual(u16, 0x1234)
        bits2 = br.read_bits(2)
        self.assertEqual(bits2, '01')


if __name__ == '__main__':
    unittest.main()
