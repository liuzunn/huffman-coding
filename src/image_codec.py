"""
Image encoder/decoder using Huffman coding with optional DPCM prediction.

Supports:
- Grayscale images: direct pixel-value Huffman coding
- Color images: per-channel independent Huffman coding
- DPCM + Huffman: differential pulse-code modulation before entropy coding
  (significantly improves compression for smooth image regions)
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.huffman import (
    build_frequency_table,
    build_huffman_tree,
    build_code_table,
    build_canonical_code_table,
    encode_data,
    decode_data,
    rebuild_tree_from_lengths,
    compute_entropy,
    compute_avg_code_length,
    coding_efficiency,
)
from src.utils import (
    BitWriter,
    BitReader,
    write_compressed_file,
    read_compressed_file,
    read_image,
    write_image,
    compute_psnr,
    compression_ratio,
    format_size,
)


# =========================================================================
#  DPCM Predictors
# =========================================================================

def _med_predictor(top: int, left: int, top_left: int) -> int:
    """Median-Edge-Detector (MED) predictor used in JPEG-LS.

    Prediction is median of three candidates:
        - left + top - top_left
        - left
        - top

    This adaptively handles horizontal, vertical, and diagonal edges.
    """
    a = left
    b = top
    c = top_left

    if c >= max(a, b):
        return min(a, b)
    elif c <= min(a, b):
        return max(a, b)
    else:
        return a + b - c


def dpcm_encode(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """Apply DPCM with MED predictor on a 2D image.

    Args:
        image: 2D array (H, W), dtype uint8.

    Returns:
        (residual, base_first_pixel) where residual values are in range [-255, 255].
        Residuals are mapped to [0, 511] for Huffman coding by adding 255.
        base_first_pixel is the original value of image[0,0] for reconstruction.
    """
    h, w = image.shape
    residual = np.zeros((h, w), dtype=np.int32)
    base_first_pixel = int(image[0, 0])
    residual[0, 0] = 0

    # First row: only left neighbor
    for j in range(1, w):
        residual[0, j] = int(image[0, j]) - int(image[0, j - 1])

    # First column: only top neighbor
    for i in range(1, h):
        residual[i, 0] = int(image[i, 0]) - int(image[i - 1, 0])

    # Rest: MED predictor
    for i in range(1, h):
        for j in range(1, w):
            pred = _med_predictor(
                int(image[i - 1, j]),     # top
                int(image[i, j - 1]),     # left
                int(image[i - 1, j - 1]), # top-left
            )
            residual[i, j] = int(image[i, j]) - pred

    # Map residual to non-negative: range [-255, 255] -> [0, 510]
    mapped = residual + 255
    return mapped, base_first_pixel


def dpcm_decode(mapped_residual: np.ndarray, base_first_pixel: int) -> np.ndarray:
    """Reconstruct image from DPCM residuals.

    Args:
        mapped_residual:  DPCM residual mapped to [0, 510] range (H, W).
        base_first_pixel: The original value of image[0,0].

    Returns:
        Reconstructed 2D uint8 image.
    """
    h, w = mapped_residual.shape
    residual = mapped_residual.astype(np.int32) - 255
    recon = np.zeros((h, w), dtype=np.int32)

    recon[0, 0] = base_first_pixel

    # First row
    for j in range(1, w):
        recon[0, j] = recon[0, j - 1] + residual[0, j]

    # First column
    for i in range(1, h):
        recon[i, 0] = recon[i - 1, 0] + residual[i, 0]

    # Rest
    for i in range(1, h):
        for j in range(1, w):
            pred = _med_predictor(
                int(recon[i - 1, j]),
                int(recon[i, j - 1]),
                int(recon[i - 1, j - 1]),
            )
            recon[i, j] = pred + residual[i, j]

    # Clamp and convert
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    return recon


# =========================================================================
#  Image Encoder
# =========================================================================

class ImageEncoder:
    """Encodes images using Huffman coding with optional DPCM.

    Usage:
        encoder = ImageEncoder()
        report = encoder.encode("input.png", "output.huf", mode="dpcm")
    """

    def __init__(self):
        self.mode = "direct"
        self.is_grayscale = False

    def encode(self, image_path: str, output_path: str,
               grayscale: bool = False, mode: str = "direct") -> Dict:
        """Encode an image using Huffman coding.

        Args:
            image_path:  Input image file path.
            output_path: Output .huf file path.
            grayscale:   Force grayscale mode (convert if input is color).
            mode:        'direct'    encode pixel values directly.
                         'dpcm'      apply DPCM predictor then encode residuals.

        Returns:
            dict with compression metrics and summary text.
        """
        self.mode = mode
        self.is_grayscale = grayscale

        # 1. Read image
        img = read_image(image_path, grayscale=grayscale)

        if img.ndim == 2:
            # Grayscale    single channel
            h, w = img.shape
            result = self._encode_channel(img, output_path, f"{os.path.basename(image_path)} (gray)")
            result.update({
                "channels": 1,
                "width": w,
                "height": h,
            })
        else:
            # Color    encode each channel independently
            h, w, c = img.shape
            summaries = []
            channel_names = ['Blue (B)', 'Green (G)', 'Red (R)']

            # In OpenCV, image is BGR
            total_original = 0
            total_compressed = 0
            total_entropy = 0.0
            total_avg_len = 0.0

            per_channel = []
            for ch in range(c):
                ch_data = img[:, :, ch]
                ch_result = self._encode_channel_data(
                    ch_data, output_path, channel_names[ch], ch_index=ch
                )
                per_channel.append(ch_result)
                total_original += ch_result["original_size"]
                total_compressed += ch_result["compressed_size"]
                total_entropy += ch_result["entropy"]
                total_avg_len += ch_result["avg_length"]

            avg_entropy = total_entropy / c
            avg_len = total_avg_len / c
            ratio = compression_ratio(total_original * 8, total_compressed * 8) \
                if total_compressed > 0 else 0.0
            eff = coding_efficiency(avg_entropy, avg_len)

            result = {
                "mode": mode,
                "channels": c,
                "width": w,
                "height": h,
                "entropy": round(avg_entropy, 4),
                "avg_length": round(avg_len, 4),
                "efficiency": round(eff * 100, 2),
                "compression_ratio": round(ratio, 4),
                "original_size": total_original,
                "compressed_size": total_compressed,
                "original_size_str": format_size(total_original),
                "compressed_size_str": format_size(total_compressed),
                "psnr": float('inf'),
                "per_channel": per_channel,
            }

            # Build summary
            lines = [
                "=" * 60,
                f"  Image Compression Report    {os.path.basename(image_path)}",
                "=" * 60,
                f"  Mode             : {mode.upper()}",
                f"  Dimensions       : {w}x{h}x{c}",
                f"  Avg entropy H(X) : {avg_entropy:.4f} bits/pixel",
                f"  Avg code length  : {avg_len:.4f} bits/pixel",
                f"  Efficiency       : {eff*100:.2f}%",
                f"  Original size    : {format_size(total_original)}",
                f"  Compressed size  : {format_size(total_compressed)}",
                f"  Compression ratio: {ratio:.4f}x",
                f"  PSNR             : inf dB (lossless)",
                "-" * 60,
            ]
            result["summary"] = "\n".join(lines)

        return result

    def _encode_channel(self, channel: np.ndarray, base_output_path: str,
                        label: str) -> Dict:
        """Encode a single image channel."""
        return self._encode_channel_data(channel, base_output_path, label)

    def _encode_channel_data(self, data: np.ndarray, base_output_path: str,
                             label: str, ch_index: int = 0) -> Dict:
        """Encode a single 2D data array (channel) with Huffman coding."""
        flat = data.ravel()

        if self.mode == "dpcm":
            # Apply DPCM before Huffman
            mapped_residual, base_val = dpcm_encode(data)
            symbols = mapped_residual.ravel().tolist()
            # Prepend base pixel value so decoder can extract original (0,0)
            symbols = [base_val] + mapped_residual.ravel().tolist()
        else:
            symbols = flat.tolist()

        # Huffman encoding
        freq_table = build_frequency_table(symbols)
        tree = build_huffman_tree(freq_table)
        code_table = build_code_table(tree)
        canonical = build_canonical_code_table(code_table)

        bits_str = encode_data(symbols, canonical)

        bw = BitWriter()
        bw.write_bits(bits_str)
        payload = bw.to_bytes()
        padding = bw.padding_bits

        symbol_lengths = {sym: len(c) for sym, c in canonical.items()}
        total_symbols = len(symbols)

        # Write to temp file or final output
        if self.mode == "dpcm":
            # Store dpcm_base in extended header (via a comment mechanism)
            # We'll write a single file per channel for multi-channel
            ch_output = base_output_path.replace('.huf', f'_ch{ch_index}.huf')
        else:
            ch_output = base_output_path.replace('.huf', f'_ch{ch_index}.huf')

        compressed_size = write_compressed_file(
            ch_output, symbol_lengths, payload, padding, total_symbols
        )

        # Compute metrics
        entropy = compute_entropy(freq_table, total_symbols)
        avg_len = compute_avg_code_length(canonical, freq_table, total_symbols)
        eff = coding_efficiency(entropy, avg_len)
        orig_size = len(flat)  # bytes (1 byte per pixel for 8-bit)
        ratio = compression_ratio(orig_size * 8, compressed_size * 8)

        return {
            "channel": label,
            "entropy": round(entropy, 4),
            "avg_length": round(avg_len, 4),
            "efficiency": round(eff * 100, 2),
            "compression_ratio": round(ratio, 4),
            "original_size": orig_size,
            "compressed_size": compressed_size,
        }


# =========================================================================
#  Image Decoder
# =========================================================================

class ImageDecoder:
    """Decodes Huffman-coded images back to image files.

    Usage:
        decoder = ImageDecoder()
        img = decoder.decode("output.huf", mode="direct", shape=(512, 512, 3))
    """

    def decode(self, input_path: str, shape: Tuple[int, ...],
               grayscale: bool = False, mode: str = "direct") -> np.ndarray:
        """Decode a .huf image file.

        Args:
            input_path:  Path to the compressed .huf file (or channel files).
            shape:       Original image shape (H, W) or (H, W, C).
            grayscale:   Whether the image is grayscale.
            mode:        'direct' or 'dpcm'.

        Returns:
            Decoded image as uint8 numpy array.
        """
        if grayscale or len(shape) == 2:
            h, w = shape[:2]
            return self._decode_single(input_path, (h, w), mode)
        else:
            h, w, c = shape
            channels = []
            base_path = input_path.replace('.huf', '')
            for ch in range(c):
                ch_path = f"{base_path}_ch{ch}.huf"
                ch_data = self._decode_single(ch_path, (h, w), mode)
                channels.append(ch_data)
            return np.stack(channels, axis=2)

    def _decode_single(self, input_path: str, shape: Tuple[int, int],
                       mode: str) -> np.ndarray:
        """Decode a single-channel .huf file."""
        h, w = shape
        n_pixels = h * w

        symbol_lengths, payload, padding, original_count = \
            read_compressed_file(input_path)

        tree = rebuild_tree_from_lengths(symbol_lengths)

        total_bits = len(payload) * 8 - padding
        bits = ''
        for i in range(total_bits):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            bits += '1' if (payload[byte_idx] >> bit_idx) & 1 else '0'

        symbols = decode_data(bits, tree)
        # Truncate to expected pixel count (+1 for DPCM base pixel)
        expected = n_pixels + (1 if mode == "dpcm" else 0)
        if len(symbols) > expected:
            symbols = symbols[:expected]

        if mode == "dpcm":
            # First symbol is the base pixel value (stored by encoder)
            base_val = symbols[0]
            residual_symbols = symbols[1:]
            mapped_residual = np.array(residual_symbols, dtype=np.int32).reshape(h, w)
            reconstructed = dpcm_decode(mapped_residual, base_val)
        else:
            reconstructed = np.array(symbols, dtype=np.uint8).reshape(h, w)

        return reconstructed

    def decode_single_channel(self, input_path: str, shape: Tuple[int, int],
                              mode: str = "direct") -> np.ndarray:
        """Convenience method for single-channel decode."""
        return self._decode_single(input_path, shape, mode)


# =========================================================================
#  Convenience function
# =========================================================================

def compress_image(input_path: str, output_path: str,
                   grayscale: bool = False, mode: str = "direct") -> Dict:
    """One-shot image compression with Huffman coding.

    Args:
        input_path:  Source image file.
        output_path: Target .huf path (will become the base name; each channel
                     gets its own _chN.huf file).
        grayscale:   Force grayscale processing.
        mode:        'direct' or 'dpcm'.

    Returns:
        Metrics dictionary.
    """
    encoder = ImageEncoder()
    return encoder.encode(input_path, output_path, grayscale=grayscale, mode=mode)


def decompress_image(input_path: str, output_path: str,
                     shape: Tuple[int, ...],
                     grayscale: bool = False, mode: str = "direct"):
    """One-shot image decompression.

    Args:
        input_path:  Base .huf file path.
        output_path: Output image file path.
        shape:       Original image shape (H, W) or (H, W, C).
        grayscale:   Whether grayscale.
        mode:        'direct' or 'dpcm'.
    """
    decoder = ImageDecoder()
    img = decoder.decode(input_path, shape, grayscale=grayscale, mode=mode)
    write_image(output_path, img)
