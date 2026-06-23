#!/usr/bin/env python3
"""
Complete Huffman Coding System
==============================
A self-contained implementation of Huffman coding for text compression.

Features:
  - Huffman tree construction via heapq (min-heap)
  - Prefix-free code table generation
  - Text encoding and bitstream decoding
  - Full text file input/output with binary compressed format
  - Average code length and coding efficiency reporting
  - Character frequency distribution visualization
"""

import heapq
import math
import os
import struct
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


class BitWriter:
    """Pack bits MSB-first into a byte buffer."""

    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_in_buffer = 0

    def write_bit(self, bit: int):
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.bits_in_buffer += 1
        if self.bits_in_buffer == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_in_buffer = 0

    def write_bits(self, bits: str):
        for ch in bits:
            self.write_bit(1 if ch == '1' else 0)

    def write_uint8(self, value: int):
        self._flush_byte()
        self.buffer.append(value & 0xFF)

    def write_uint16(self, value: int):
        self._flush_byte()
        self.buffer.extend(struct.pack('<H', value & 0xFFFF))

    def write_uint32(self, value: int):
        self._flush_byte()
        self.buffer.extend(struct.pack('<I', value & 0xFFFFFFFF))

    def write_uint64(self, value: int):
        self._flush_byte()
        self.buffer.extend(struct.pack('<Q', value & 0xFFFFFFFFFFFFFFFF))

    def write_bytes(self, data: bytes):
        self._flush_byte()
        self.buffer.extend(data)

    def _flush_byte(self):
        if self.bits_in_buffer > 0:
            self.current_byte <<= (8 - self.bits_in_buffer)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_in_buffer = 0

    def flush(self) -> bytearray:
        if self.bits_in_buffer > 0:
            self.current_byte <<= (8 - self.bits_in_buffer)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_in_buffer = 0
        return self.buffer

    def to_bytes(self) -> bytes:
        self.flush()
        return bytes(self.buffer)

    @property
    def padding(self) -> int:
        return 8 - self.bits_in_buffer if self.bits_in_buffer > 0 else 0


class BitReader:
    """Read bits MSB-first from a bytes object."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            raise EOFError("End of stream")
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def read_bits(self, n: int) -> str:
        return ''.join(str(self.read_bit()) for _ in range(n))

    def read_uint8(self) -> int:
        self._align()
        v = self.data[self.byte_pos]
        self.byte_pos += 1
        return v

    def read_uint16(self) -> int:
        self._align()
        v = struct.unpack('<H', self.data[self.byte_pos:self.byte_pos + 2])[0]
        self.byte_pos += 2
        return v

    def read_uint32(self) -> int:
        self._align()
        v = struct.unpack('<I', self.data[self.byte_pos:self.byte_pos + 4])[0]
        self.byte_pos += 4
        return v

    def read_uint64(self) -> int:
        self._align()
        v = struct.unpack('<Q', self.data[self.byte_pos:self.byte_pos + 8])[0]
        self.byte_pos += 8
        return v
    def read_bytes(self, n: int) -> bytes:
        self._align()
        result = self.data[self.byte_pos:self.byte_pos + n]
        self.byte_pos += n
        return result

    def _align(self):
        if self.bit_pos != 0:
            self.bit_pos = 0
            self.byte_pos += 1


class HuffmanNode:
    """Node in a Huffman binary tree."""

    __slots__ = ('symbol', 'freq', 'left', 'right')

    def __init__(self, symbol=None, freq=0.0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq < other.freq
        return id(self) < id(other)

    def is_leaf(self):
        return self.left is None and self.right is None


class HuffmanCoding:
    """Complete Huffman coding system with encode/decode and metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._freq_table: Dict[int, int] = {}
        self._code_table: Dict[int, str] = {}
        self._tree: Optional[HuffmanNode] = None
        self._total_chars: int = 0

    @property
    def freq_table(self):
        return {chr(k) if 32 <= k < 127 else f'U+{k:04X}': v
                for k, v in sorted(self._freq_table.items(), key=lambda x: -x[1])}

    @property
    def entropy(self) -> float:
        if self._total_chars == 0:
            return 0.0
        h = 0.0
        for f in self._freq_table.values():
            p = f / self._total_chars
            h -= p * math.log2(p)
        return h

    @property
    def avg_length(self) -> float:
        if self._total_chars == 0:
            return 0.0
        total = 0.0
        for sym, code in self._code_table.items():
            p = self._freq_table[sym] / self._total_chars
            total += p * len(code)
        return total

    @property
    def efficiency(self) -> float:
        L = self.avg_length
        return self.entropy / L if L > 0 else 0.0

    @property
    def fixed_length_bits(self) -> int:
        n = len(self._freq_table)
        return 1 if n <= 1 else math.ceil(math.log2(n))

    def build_tree(self, data):
        """Build Huffman tree from input.

        1. Count frequencies
        2. Create leaf nodes (min-heap)
        3. Merge two lowest-frequency nodes repeatedly
        4. Final node = tree root
        5. Generate prefix-free code table
        """
        self.reset()
        if isinstance(data, str):
            self._freq_table = dict(Counter(ord(c) for c in data))
        else:
            self._freq_table = dict(Counter(data))
        self._total_chars = sum(self._freq_table.values())
        if not self._freq_table:
            return {}

        heap = [HuffmanNode(symbol=s, freq=f)
                for s, f in self._freq_table.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(freq=left.freq + right.freq,
                                 left=left, right=right)
            heapq.heappush(heap, parent)

        self._tree = heap[0]
        self._build_code_table(self._tree)
        return dict(self._freq_table)
    def _build_code_table(self, node, prefix=""):
        if node.is_leaf() and node.symbol is not None:
            self._code_table[node.symbol] = prefix or "0"
            return
        if node.left:
            self._build_code_table(node.left, prefix + "0")
        if node.right:
            self._build_code_table(node.right, prefix + "1")

    @staticmethod
    def _canonical(code_table):
        """Convert code table to canonical form (same lengths, reassigned values)."""
        items = sorted(code_table.items(), key=lambda x: (len(x[1]), x[0]))
        result, code, cur_len = {}, 0, 0
        for sym, orig in items:
            length = len(orig)
            code <<= (length - cur_len)
            result[sym] = format(code, f'0{length}b')
            code += 1
            cur_len = length
        return result

    def encode(self, text: str) -> Tuple[str, Dict]:
        if not text:
            return "", {}
        self.build_tree(text)
        # Convert to canonical (consistent with file header reconstruction)
        self._code_table = self._canonical(self._code_table)
        # Rebuild tree so decode() matches the canonical code table
        sym_lens = {s: len(c) for s, c in self._code_table.items()}
        self._tree = self._rebuild_tree_from_lengths(sym_lens)
        bits = ''.join(self._code_table[ord(c)] for c in text)
        return bits, self._metrics()

    def decode(self, bitstream: str) -> str:
        """Decode bitstream string -> text.

        Tree must have been built by prior encode()/build_tree() call.
        For file decoding, use decode_file().
        """
        if not bitstream or not self._tree:
            return ""
        if self._tree.is_leaf() and self._tree.symbol is not None:
            return chr(self._tree.symbol) * len(bitstream)

        chars, node = [], self._tree
        for bit in bitstream:
            node = node.left if bit == '0' else node.right
            if node is not None and node.is_leaf():
                chars.append(chr(node.symbol))
                node = self._tree
        return ''.join(chars)

    def encode_file(self, input_path: str, output_path: str) -> Dict:
        """Compress a text file. Returns metrics dict."""
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        bits, metrics = self.encode(text)
        self._write_binary(output_path, bits)
        print(f"  Original: {metrics['total_chars']} chars "
              f"({self._fmt_size(len(text))})")
        print(f"  Compressed: {self._fmt_size(os.path.getsize(output_path))}")
        print(f"  Entropy H(X): {metrics['entropy']:.4f} bits/symbol")
        print(f"  Average code length: {metrics['avg_length']:.4f} bits/symbol")
        print(f"  Coding efficiency: {metrics['efficiency']:.2f}%")
        return metrics

    def decode_file(self, input_path: str, output_path: str) -> str:
        """Decompress a .huf file back to text."""
        code_lengths, payload, padding, orig_size = self._read_binary(input_path)
        self._tree = self._rebuild_tree_from_lengths(code_lengths)
        self._freq_table = code_lengths
        self._total_chars = orig_size

        total_bits = len(payload) * 8 - padding
        bits = ''.join(
            '1' if (payload[i // 8] >> (7 - (i % 8))) & 1 else '0'
            for i in range(total_bits)
        )
        text = self.decode(bits)
        if len(text) > orig_size:
            text = text[:orig_size]
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  Decoded {len(text)} characters -> {output_path}")
        return text

    def show_freq_table(self, top_n: int = 15):
        """Print character frequency distribution with codes."""
        if not self._freq_table:
            print("(No frequency data)")
            return
        total = sum(self._freq_table.values())
        items = sorted(self._freq_table.items(), key=lambda x: -x[1])
        print(f"\n{'Char':<8} {'Freq':<8} {'Pct':<8} {'Code':<12} {'Len'}")
        print("-" * 50)
        for sym, count in items[:top_n]:
            ch = chr(sym) if 32 <= sym < 127 else f'[{sym}]'
            code = self._code_table.get(sym, '')
            print(f"{ch:<8} {count:<8} {count/total*100:<8.2f} {code:<12} {len(code)}")
        if len(items) > top_n:
            print(f"  ... and {len(items) - top_n} more symbols")
        print()

    @staticmethod
    def _rebuild_tree_from_lengths(sym_lens):
        if len(sym_lens) == 1:
            return HuffmanNode(symbol=next(iter(sym_lens.keys())), freq=1.0)
        items = sorted(sym_lens.items(), key=lambda x: (x[1], x[0]))
        codes = {}
        code, cur_len = 0, 0
        for sym, length in items:
            code <<= (length - cur_len)
            codes[sym] = format(code, f'0{length}b')
            code += 1
            cur_len = length
        root = HuffmanNode()
        for sym, cs in codes.items():
            node = root
            for b in cs:
                if b == '0':
                    if not node.left:
                        node.left = HuffmanNode()
                    node = node.left
                else:
                    if not node.right:
                        node.right = HuffmanNode()
                    node = node.right
            node.symbol = sym
        return root

    def _write_binary(self, path: str, bits: str):
        """Write .huf file: Magic(2B)+Version(1B)+OrigSize(8B)
        +NumSyms(2B)+SymTable(Nx5B)+Padding(1B)+Payload"""
        bw = BitWriter()
        bw.write_bytes(b'HF')
        bw.write_uint8(1)
        bw.write_uint64(self._total_chars)
        bw.write_uint16(len(self._freq_table))
        for sym in sorted(self._freq_table.keys()):
            bw.write_uint32(sym)
            bw.write_uint8(len(self._code_table[sym]))
        pbw = BitWriter()
        pbw.write_bits(bits)
        bw.write_uint8(pbw.padding)
        bw.write_bytes(pbw.to_bytes())
        with open(path, 'wb') as f:
            f.write(bw.to_bytes())
    @staticmethod
    def _read_binary(path):
        with open(path, 'rb') as f:
            data = f.read()
        br = BitReader(data)
        if br.read_bytes(2) != b'HF':
            raise ValueError("Invalid file format")
        br.read_uint8()
        orig_size = br.read_uint64()
        num_syms = br.read_uint16()
        sym_lens = {}
        for _ in range(num_syms):
            sym_lens[br.read_uint32()] = br.read_uint8()
        padding = br.read_uint8()
        payload = br.read_bytes(len(data) - br.byte_pos)
        return sym_lens, payload, padding, orig_size

    def _metrics(self):
        n = self._total_chars
        fixed_bits = n * self.fixed_length_bits
        code_bits = sum(self._freq_table[s] * len(self._code_table[s])
                        for s in self._freq_table) if self._freq_table else 0
        return {
            "total_chars": n,
            "unique_symbols": len(self._freq_table),
            "entropy": round(self.entropy, 4),
            "avg_length": round(self.avg_length, 4),
            "efficiency": round(self.efficiency * 100, 2),
            "compression_ratio": round(fixed_bits / code_bits, 4) if code_bits else 0.0,
            "freq_table": self.freq_table,
        }

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ['B', 'KB', 'MB']:
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} GB"


if __name__ == '__main__':
    hc = HuffmanCoding()
    print("=" * 55)
    print("  Huffman Coding System")
    print("=" * 55)

    # Test 1: string encode/decode
    print("\n[1] String encode/decode roundtrip")
    t1 = "Hello Huffman Coding! Optimal prefix codes."
    b1, m1 = hc.encode(t1)
    d1 = hc.decode(b1)
    print(f"  Input:  {t1}")
    print(f"  Output: {d1}")
    print(f"  Bits:   {len(b1)}")
    print(f"  Match:  {'YES (lossless)' if t1 == d1 else 'MISMATCH'}")

    # Test 2: frequency distribution
    print("\n[2] Character frequency distribution")
    hc.show_freq_table()

    # Test 3: Chinese / Unicode support
    print("[3] Chinese text")
    t2 = "哈夫曼编码是一种最优前缀码"
    b2, m2 = hc.encode(t2)
    d2 = hc.decode(b2)
    print(f"  Original: {t2}")
    print(f"  Decoded:  {d2}")
    print(f"  Match:    {'YES' if t2 == d2 else 'NO'}")

    # Test 4: file compression
    print("\n[4] File compression (.txt -> .huf)")
    sf = os.path.join(os.path.dirname(__file__), 'data', 'sample.txt')
    if os.path.exists(sf):
        *_, m4 = hc.encode_file(sf, 'demo.huf')
        d4 = hc.decode_file('demo.huf', 'demo.txt')
        with open(sf) as f: o4 = f.read()
        with open('demo.txt') as f: r4 = f.read()
        print(f"  Roundtrip: {'LOSSELSS' if o4 == r4 else 'MISMATCH'}")
        os.remove('demo.huf'); os.remove('demo.txt')

    # Test 5: metrics
    print("\n[5] Information-theoretic metrics")
    t5 = "The quick brown fox jumps over the lazy dog."
    b5, m5 = hc.encode(t5)
    print(f'  Text: "{t5}"')
    print(f"  Unique symbols:  {m5['unique_symbols']}")
    print(f"  Entropy H(X):    {m5['entropy']:.4f} bits/symbol")
    print(f"  Avg code length: {m5['avg_length']:.4f} bits/symbol")
    print(f"  Efficiency:      {m5['efficiency']:.2f}%")
    print(f"  Compression vs fixed-length: {m5['compression_ratio']:.4f}x")
    print("\n" + "=" * 55)
    print("  Demo complete.")
    print("=" * 55)
