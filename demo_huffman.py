#!/usr/bin/env python3
"""Demo script for the Huffman Coding System."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from huffman_coding import HuffmanCoding

hc = HuffmanCoding()
print("=" * 55)
print("  Huffman Coding System - Demo")
print("=" * 55)

# Test 1: String roundtrip
print("\n[1] String encode/decode roundtrip")
t1 = "Hello Huffman Coding! Optimal prefix codes."
b1, m1 = hc.encode(t1)
d1 = hc.decode(b1)
print(f"  Input:  {t1}")
print(f"  Output: {d1}")
print(f"  Bits:   {len(b1)}")
print(f"  Match:  {'YES (lossless)' if t1 == d1 else 'MISMATCH'}")

# Test 2: Frequency distribution
print("\n[2] Character frequency distribution")
hc.show_freq_table()

# Test 3: Chinese text
print("[3] Chinese text")
t2 = "哈夫曼编码是一种最优前缀码"
b2, m2 = hc.encode(t2)
d2 = hc.decode(b2)
print(f"  Original: {t2}")
print(f"  Decoded:  {d2}")
print(f"  Match:    {'YES' if t2 == d2 else 'NO'}")

# Test 4: File compression
print("\n[4] File compression (.txt -> .huf)")
sf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample.txt')
if os.path.exists(sf):
    m4 = hc.encode_file(sf, 'demo.huf')
    d4 = hc.decode_file('demo.huf', 'demo.txt')
    with open(sf) as f: o4 = f.read()
    with open('demo.txt') as f: r4 = f.read()
    print(f"  Roundtrip: {'LOSSELSS' if o4 == r4 else 'MISMATCH'}")
    os.remove('demo.huf'); os.remove('demo.txt')
else:
    print("  (sample.txt not found)")

# Test 5: Metrics
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
