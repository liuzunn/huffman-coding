"""
Utility classes and functions for the Huffman coding system.

Provides:
- BitWriter / BitReader: bit-level I/O with MSB-first packing
- File format constants and header serialization (.huf format)
- Compression metrics (PSNR, ratio, BPS)
- File I/O helpers for text, binary, and image data
"""

import math
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple


# =========================================================================
#  Bit-level I/O
# =========================================================================

class BitWriter:
    """Writes bits to a bytearray. Bits are accumulated MSB-first and flushed
    as complete bytes. Any leftover bits are zero-padded at flush time.

    Usage:
        bw = BitWriter()
        bw.write_bit(1)
        bw.write_bits("0101")
        data = bw.to_bytes()   # auto-flush
    """

    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_in_buffer = 0
        self._padding = 0

    def write_bit(self, bit: int):
        """Write a single bit (0 or 1)."""
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.bits_in_buffer += 1
        if self.bits_in_buffer == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_in_buffer = 0

    def write_bits(self, bits: str):
        """Write a string of '0' and '1' characters."""
        for ch in bits:
            self.write_bit(1 if ch == '1' else 0)

    def write_uint8(self, value: int):
        """Write an 8-bit unsigned integer (byte-aligned)."""
        self._flush_byte()
        self.buffer.append(value & 0xFF)

    def write_uint16(self, value: int):
        """Write a 16-bit unsigned integer in little-endian order."""
        self._flush_byte()
        self.buffer.extend(struct.pack('<H', value & 0xFFFF))

    def write_uint32(self, value: int):
        """Write a 32-bit unsigned integer in little-endian order."""
        self._flush_byte()
        self.buffer.extend(struct.pack('<I', value & 0xFFFFFFFF))

    def write_uint64(self, value: int):
        """Write a 64-bit unsigned integer in little-endian order."""
        self._flush_byte()
        self.buffer.extend(struct.pack('<Q', value & 0xFFFFFFFFFFFFFFFF))

    def write_bytes(self, data: bytes):
        """Write raw bytes (must be byte-aligned)."""
        self._flush_byte()
        self.buffer.extend(data)

    def _flush_byte(self):
        """Internal: flush any buffered bits (zero-padded) before byte writes."""
        if self.bits_in_buffer > 0:
            padding = 8 - self.bits_in_buffer
            self.current_byte <<= padding
            self.buffer.append(self.current_byte)
            self._padding += padding
            self.current_byte = 0
            self.bits_in_buffer = 0

    @property
    def padding_bits(self) -> int:
        """Number of zero-padding bits added during the last flush."""
        # Recompute from current state if not flushed yet
        if self.bits_in_buffer > 0:
            return 8 - self.bits_in_buffer
        return self._padding

    def flush(self) -> bytearray:
        """Explicitly flush remaining bits (zero-padded). Returns full buffer."""
        if self.bits_in_buffer > 0:
            padding = 8 - self.bits_in_buffer
            self.current_byte <<= padding
            self.buffer.append(self.current_byte)
            self._padding = padding
            self.current_byte = 0
            self.bits_in_buffer = 0
        return self.buffer

    def to_bytes(self) -> bytes:
        """Flush and return all bytes written so far."""
        self.flush()
        return bytes(self.buffer)

    def reset(self):
        """Clear the buffer and reset state."""
        self.buffer.clear()
        self.current_byte = 0
        self.bits_in_buffer = 0
        self._padding = 0


class BitReader:
    """Reads bits from a bytes object. Bits are read MSB-first.

    Usage:
        br = BitReader(data)
        bit = br.read_bit()
        bits = br.read_bits(5)
        val = br.read_uint16()
    """

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0   # 0-7, position within current byte (MSB = 0)

    def read_bit(self) -> int:
        """Read a single bit. Returns 0 or 1. Raises EOFError if exhausted."""
        if self.byte_pos >= len(self.data):
            raise EOFError("No more bits to read — end of compressed stream")
        # MSB-first: bit at (7 - bit_pos)
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def read_bits(self, n: int) -> str:
        """Read n bits and return as a binary string of '0'/'1'."""
        return ''.join(str(self.read_bit()) for _ in range(n))

    def read_uint8(self) -> int:
        """Read an 8-bit unsigned integer (byte-aligned)."""
        self._align()
        if self.byte_pos + 1 > len(self.data):
            raise EOFError("Not enough data for uint8")
        value = self.data[self.byte_pos]
        self.byte_pos += 1
        return value

    def read_uint16(self) -> int:
        """Read a 16-bit unsigned integer, little-endian (byte-aligned)."""
        self._align()
        if self.byte_pos + 2 > len(self.data):
            raise EOFError("Not enough data for uint16")
        value = struct.unpack('<H', self.data[self.byte_pos:self.byte_pos + 2])[0]
        self.byte_pos += 2
        return value

    def read_uint32(self) -> int:
        """Read a 32-bit unsigned integer, little-endian (byte-aligned)."""
        self._align()
        if self.byte_pos + 4 > len(self.data):
            raise EOFError("Not enough data for uint32")
        value = struct.unpack('<I', self.data[self.byte_pos:self.byte_pos + 4])[0]
        self.byte_pos += 4
        return value

    def read_uint64(self) -> int:
        """Read a 64-bit unsigned integer, little-endian (byte-aligned)."""
        self._align()
        if self.byte_pos + 8 > len(self.data):
            raise EOFError("Not enough data for uint64")
        value = struct.unpack('<Q', self.data[self.byte_pos:self.byte_pos + 8])[0]
        self.byte_pos += 8
        return value

    def read_bytes(self, n: int) -> bytes:
        """Read n raw bytes (byte-aligned)."""
        self._align()
        if self.byte_pos + n > len(self.data):
            raise EOFError(f"Not enough data for {n} bytes")
        result = self.data[self.byte_pos:self.byte_pos + n]
        self.byte_pos += n
        return result

    def _align(self):
        """Discard any remaining bits in the current byte and move to next byte."""
        if self.bit_pos != 0:
            self.bit_pos = 0
            self.byte_pos += 1

    @property
    def is_aligned(self) -> bool:
        """Check if the reader is currently at a byte boundary."""
        return self.bit_pos == 0

    @property
    def bytes_consumed(self) -> int:
        """Total bytes consumed so far (including partial bytes)."""
        return self.byte_pos + (1 if self.bit_pos > 0 else 0)


# =========================================================================
#  Compressed File Format (.huf)
# =========================================================================

# Constants
HUF_MAGIC = b'HF'          # 2-byte magic identifier
HUF_VERSION = 1             # Format version

# Header structure:
#   [0-1]   Magic:           2 bytes  (b'HF')
#   [2]     Version:         1 byte   (1)
#   [3-10]  Original size:   8 bytes  (uint64 LE — number of symbols)
#   [11-12] Symbol count:    2 bytes  (uint16 LE)
#   [13+]   Symbol table:    N entries × (symbol: 4 bytes + length: 1 byte)
#           + 1 byte : padding bits count
#           + payload: variable-length bitstream

HEADER_MAGIC_SIZE = 2
HEADER_VERSION_SIZE = 1
HEADER_ORIG_SIZE = 8
HEADER_SYM_COUNT_SIZE = 2
SYMBOL_ENTRY_SIZE = 5   # 4 bytes symbol + 1 byte length


def write_compressed_file(output_path: str, symbol_lengths: Dict[int, int],
                          bitstream: bytes, padding: int,
                          original_symbol_count: int) -> int:
    """Write a complete .huf compressed file.

    Args:
        output_path:          File path to write.
        symbol_lengths:       {symbol: code_length} mapping.
        bitstream:            Packed bytes of the encoded payload.
        padding:              Number of zero-padding bits in the last byte.
        original_symbol_count: Total number of original symbols (for verification).

    Returns:
        Total bytes written.
    """
    bw = BitWriter()

    # ---- Header ----
    bw.write_bytes(HUF_MAGIC)                           # magic
    bw.write_uint8(HUF_VERSION)                         # version
    bw.write_uint64(original_symbol_count)              # original size

    num_symbols = len(symbol_lengths)
    bw.write_uint16(num_symbols)                        # symbol count

    # Symbol table: sorted by symbol for deterministic format
    sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: x[0])
    for sym, length in sorted_symbols:
        bw.write_uint32(sym)                            # symbol value
        bw.write_uint8(length)                          # code length (bits)

    # ---- Payload ----
    bw.write_uint8(padding)                             # padding indicator
    bw.write_bytes(bitstream)                           # compressed payload
    # Note: bitstream is already byte-packed, so we write raw bytes

    data = bw.to_bytes()
    with open(output_path, 'wb') as f:
        f.write(data)
    return len(data)


def read_compressed_file(input_path: str) -> Tuple[Dict[int, int], bytes, int, int]:
    """Read a .huf compressed file and extract header + payload.

    Args:
        input_path: Path to the .huf file.

    Returns:
        (symbol_lengths, bitstream_bytes, padding, original_symbol_count)
    """
    with open(input_path, 'rb') as f:
        data = f.read()

    br = BitReader(data)

    # ---- Parse header ----
    magic = br.read_bytes(HEADER_MAGIC_SIZE)
    if magic != HUF_MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r} (expected {HUF_MAGIC!r})")

    version = br.read_uint8()
    if version != HUF_VERSION:
        raise ValueError(f"Unsupported version: {version} (expected {HUF_VERSION})")

    original_symbol_count = br.read_uint64()
    num_symbols = br.read_uint16()

    # Read symbol table
    symbol_lengths: Dict[int, int] = {}
    for _ in range(num_symbols):
        sym = br.read_uint32()
        length = br.read_uint8()
        symbol_lengths[sym] = length

    # Read padding indicator
    padding = br.read_uint8()

    # Read payload (remaining bytes)
    payload = br.read_bytes(len(data) - br.byte_pos)

    return symbol_lengths, payload, padding, original_symbol_count


# =========================================================================
#  Compression Metrics
# =========================================================================

def compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """Compression ratio = original_size / compressed_size.

    >1.0 means compression is beneficial; <1.0 means expansion.
    """
    if compressed_bits <= 0:
        return 0.0
    return original_bits / compressed_bits


def bit_per_symbol(total_bits: int, num_symbols: int) -> float:
    """Average number of bits used per symbol."""
    if num_symbols <= 0:
        return 0.0
    return total_bits / num_symbols


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two images.

    Returns infinity for identical images (lossless compression).
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    mse = np.mean(
        (original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    )
    if mse < 1e-12:   # effectively identical
        return float('inf')
    max_val = 255.0
    return 10.0 * math.log10(max_val ** 2 / mse)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form (B/KB/MB)."""
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


# =========================================================================
#  File I/O Helpers
# =========================================================================

def read_text_file(path: str, encoding: str = 'utf-8') -> str:
    """Read a text file and return its full content as a string."""
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def write_text_file(path: str, content: str, encoding: str = 'utf-8'):
    """Write a string to a text file."""
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


def read_binary_file(path: str) -> bytes:
    """Read a binary file and return its content as bytes."""
    with open(path, 'rb') as f:
        return f.read()


def write_binary_file(path: str, data: bytes):
    """Write bytes to a binary file."""
    with open(path, 'wb') as f:
        f.write(data)

def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image using OpenCV with Unicode-safe path handling.

    OpenCV imread/imwrite do not support Unicode paths on Windows,
    so we use np.frombuffer + cv2.imdecode instead.
    """
    import cv2
    import numpy as np
    with open(path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(file_bytes, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img

def write_image(path: str, image: np.ndarray):
    """Write a numpy array as an image file (Unicode-safe).
    
    Uses OpenCV imencode to avoid Unicode path issues with imwrite.
    """
    import cv2
    success, buf = cv2.imencode('.png', image)
    if not success:
        raise RuntimeError("Failed to encode image as PNG")
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
