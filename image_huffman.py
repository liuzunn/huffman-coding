#!/usr/bin/env python3
"""Image Huffman Coding: grayscale image compression with lossless reconstruction."""
import heapq, math, os, struct, sys
from collections import Counter
import cv2, numpy as np
from matplotlib import pyplot as plt

class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.byte = 0
        self.bits = 0
    def write(self, b):
        self.byte = (self.byte << 1) | (b & 1)
        self.bits += 1
        if self.bits == 8:
            self.buf.append(self.byte)
            self.byte = 0
            self.bits = 0
    def write_str(self, s):
        for c in s:
            self.write(1 if c == "1" else 0)
    def flush(self):
        if self.bits > 0:
            self.byte <<= (8 - self.bits)
            self.buf.append(self.byte)
            self.byte = 0
            self.bits = 0
        return bytes(self.buf)
    @property
    def padding(self):
        return 8 - self.bits if self.bits > 0 else 0

class HuffmanNode:
    __slots__ = ("sym", "freq", "left", "right")
    def __init__(self, sym=None, freq=0, left=None, right=None):
        self.sym = sym; self.freq = freq; self.left = left; self.right = right
    def __lt__(self, o):
        return (self.freq, id(self)) < (o.freq, id(o))
    def is_leaf(self):
        return self.left is None and self.right is None

class ImageHuffman:
    def __init__(self):
        self.freq = {}
        self.codes = {}
        self.tree = None
        self.h = self.w = 0

    def load_image(self, path):
        """Step 1-2: Read image, convert to grayscale."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            from PIL import Image
            img = np.array(Image.open(path).convert("L"))
        self.h, self.w = img.shape
        self.img = img
        print(f"[OK] Loaded: {path} -> {self.w}x{self.h} grayscale")

    def build_histogram(self):
        """Step 3: Count pixel value frequencies (0-255)."""
        pixels = self.img.ravel().tolist()
        self.freq = dict(Counter(pixels))
        n = len(pixels)
        h = 0.0
        for c in self.freq.values():
            p = c / n
            h -= p * math.log2(p)
        self.entropy = h
        print(f"[OK] Histogram: {len(self.freq)} unique / {n} total")
        print(f"[OK] Entropy: {h:.4f} bits/pixel")
        return self.freq

    def build_tree(self):
        """Step 4: Build Huffman tree and code table."""
        heap = [HuffmanNode(sym=s, freq=f) for s, f in self.freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, HuffmanNode(freq=a.freq + b.freq, left=a, right=b))
        self.tree = heap[0] if heap else None
        self.codes = {}
        self._gen_codes(self.tree)
        # Convert to canonical codes for decoder compatibility
        items = sorted(self.codes.items(), key=lambda x: (len(x[1]), x[0]))
        out, code, cur = {}, 0, 0
        for sym, orig in items:
            length = len(orig)
            code <<= (length - cur)
            out[sym] = format(code, f"0{length}b")
            code += 1; cur = length
        self.codes = out
        n = sum(self.freq.values())
        self.avg_len = sum(self.freq[s] * len(c) for s, c in self.codes.items()) / n
        print(f"[OK] Tree: {len(self.codes)} codes, avg len: {self.avg_len:.4f} bpp, "
              f"eff: {self.entropy / self.avg_len * 100:.2f}%")

    def _gen_codes(self, node, prefix=""):
        if node.is_leaf() and node.sym is not None:
            self.codes[node.sym] = prefix or "0"
            return
        if node.left:
            self._gen_codes(node.left, prefix + "0")
        if node.right:
            self._gen_codes(node.right, prefix + "1")

    def encode(self):
        """Encode pixel data to bitstream."""
        bw = BitWriter()
        for p in self.img.ravel():
            bw.write_str(self.codes[p])
        return bw

    def save(self, path):
        """Step 5: Save compressed .huf file."""
        bw = self.encode()
        out = bytearray()
        out.extend(b"HF"); out.append(1)                  # magic + version
        out.extend(struct.pack("<HH", self.h, self.w))    # dimensions
        out.extend(struct.pack("<H", len(self.freq)))     # num symbols
        for sym in sorted(self.freq.keys()):
            out.extend(struct.pack("<I", sym))            # pixel value
            out.extend(struct.pack("<H", len(self.codes[sym])))  # code length
        out.append(bw.padding)
        out.extend(bw.flush())
        with open(path, "wb") as f:
            f.write(bytes(out))
        orig = self.h * self.w
        comp = os.path.getsize(path)
        print(f"[OK] Compressed: {orig} B -> {comp} B (ratio: {orig/comp:.4f}x)")

    def load(self, path):
        """Step 6: Load .huf file and reconstruct image."""
        with open(path, "rb") as f:
            data = f.read()
        if data[0:2] != b"HF":
            raise ValueError("Invalid file format")
        pos = 3  # skip magic + version
        h = struct.unpack("<H", data[pos:pos+2])[0]; pos += 2
        w = struct.unpack("<H", data[pos:pos+2])[0]; pos += 2
        n_syms = struct.unpack("<H", data[pos:pos+2])[0]; pos += 2
        sym_lens = {}
        for _ in range(n_syms):
            s = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
            ln = struct.unpack("<H", data[pos:pos+2])[0]; pos += 2
            sym_lens[s] = ln
        pad = data[pos]; pos += 1
        payload = data[pos:]
        # Rebuild canonical tree
        items = sorted(sym_lens.items(), key=lambda x: (x[1], x[0]))
        cd, code, cur = {}, 0, 0
        for sym, ln in items:
            code <<= (ln - cur)
            cd[sym] = format(code, f"0{ln}b")
            code += 1; cur = ln
        root = HuffmanNode()
        for sym, cs in cd.items():
            node = root
            for b in cs:
                if b == "0":
                    if not node.left: node.left = HuffmanNode()
                    node = node.left
                else:
                    if not node.right: node.right = HuffmanNode()
                    node = node.right
            node.sym = sym
        self.tree = root
        # Decode bitstream
        bits = len(payload) * 8 - pad
        s = "".join("1" if (payload[i//8] >> (7-(i%8))) & 1 else "0" for i in range(bits))
        pixels, node = [], root
        for b in s:
            node = node.left if b == "0" else node.right
            if node and node.is_leaf():
                pixels.append(node.sym); node = root
        if len(pixels) > h * w:
            pixels = pixels[:h*w]
        self.h, self.w, self.img = h, w, np.array(pixels, dtype=np.uint8).reshape(h, w)
        print(f"[OK] Decoded: {w}x{h} grayscale ({len(pixels)} pixels)")

    def psnr(self, other):
        """Compute PSNR (inf = identical, lossless)."""
        mse = np.mean((self.img.astype(np.float64) - other.astype(np.float64))**2)
        return float("inf") if mse < 1e-12 else 10 * math.log10(255**2 / mse)

    def show_histogram(self, save=None):
        """Step 8a: Display/save pixel distribution histogram."""
        plt.figure(figsize=(10, 5))
        plt.hist(self.img.ravel(), bins=256, range=(0, 255),
                 color="#2E86AB", alpha=0.85, edgecolor="white", linewidth=0.3)
        plt.title("Grayscale Pixel Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Pixel Value"); plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        if save: plt.savefig(save, dpi=150); print(f"[OK] Histogram: {save}")
        plt.close()

    def show_comparison(self, recon, title="Original vs Reconstructed", save=None):
        """Step 8b: Side-by-side original and reconstructed."""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
        a1.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        a1.set_title(f"Original ({self.w}x{self.h})", fontsize=12)
        a1.axis("off")
        a2.imshow(recon, cmap="gray", vmin=0, vmax=255)
        p = self.psnr(recon)
        ps = "inf dB" if p == float("inf") else f"{p:.2f} dB"
        a2.set_title(f"Reconstructed (PSNR: {ps})", fontsize=12)
        a2.axis("off")
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save: plt.savefig(save, dpi=150); print(f"[OK] Comparison: {save}")
        plt.close()

def demo():
    print("=" * 60)
    print("  Image Huffman Coding System")
    print("=" * 60)
    base = os.path.dirname(os.path.abspath(__file__))
    img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "data", "image.png")
    if not os.path.exists(img):
        print(f"[!] Not found: {img}"); return
    stem = img.rsplit(".", 1)[0]
    h1 = ImageHuffman(); h2 = ImageHuffman()
    print("\n--- Step 1-2: Load & Grayscale ---")
    h1.load_image(img)
    print("\n--- Step 3-4: Histogram + Huffman Tree ---")
    h1.build_histogram(); h1.build_tree()
    print("\n--- Step 5: Compress ---")
    h1.save(stem + ".huf")
    print("\n--- Step 6: Decode & Reconstruct ---")
    h2.load(stem + ".huf")
    p = h1.psnr(h2.img)
    status = "LOSSELSS" if p == float("inf") else f"LOSSY (PSNR: {p:.2f} dB)"
    print(f"\n  >>> {status}")
    print("\n--- Step 7-8: Visualize ---")
    h1.show_histogram(save=stem + "_histogram.png")
    h1.show_comparison(h2.img, save=stem + "_comparison.png")
    c = os.path.getsize(stem + ".huf")
    print(f"\n  Summary: {h1.w}x{h1.h} | entropy {h1.entropy:.4f} | "
          f"avg len {h1.avg_len:.4f} | ratio {h1.h*h1.w/c:.4f}x")
    print(f"  Reconstruction: {status}")

if __name__ == "__main__":
    demo()
