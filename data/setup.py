# -*- coding: utf-8 -*-
"""
Generate sample data for Huffman coding experiments.
Creates text samples and synthetic test images.
"""

import os
import numpy as np
from PIL import Image

TEXT_LINES = [
    "The discipline of information theory was founded by Claude Shannon",
    "in 1948 with his seminal paper A Mathematical Theory of Communication.",
    "",
    "Source coding removes redundancy while preserving information content.",
    "",
    "Huffman coding (David A. Huffman, 1952) produces optimal prefix codes.",
    "It merges lowest-frequency nodes into a binary tree structure.",
    "Canonical Huffman allows compact header storage using code lengths only.",
]
TEXT = chr(10).join(TEXT_LINES)


def create_sample_text(output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(TEXT)
    print(f"[OK] Text: {output_path} ({len(TEXT)} chars)")


def create_sample_images(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    size = 256
    gray = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size * 0.7)
            v = int(255 * max(0, min(1, 1 - d)))
            gray[y, x] = v
    gray[30:70, 30:70] = np.random.randint(180, 220, (40, 40)).astype(np.uint8)
    gray[80:120, 100:140] = np.random.randint(120, 160, (40, 40)).astype(np.uint8)
    gray[120:130, :] = 0
    gray[180:185, :] = 255
    Image.fromarray(gray).save(os.path.join(output_dir, "image.png"))

    color = np.zeros((size, size, 3), dtype=np.uint8)
    color[:, :, 0] = gray
    color[:, :, 1] = (gray * 0.7 + np.roll(gray, 20, axis=1) * 0.3).astype(np.uint8)
    color[:, :, 2] = (gray * 0.5 + np.roll(gray, -20, axis=0) * 0.5).astype(np.uint8)
    Image.fromarray(color).save(os.path.join(output_dir, "image_color.png"))

    checker = np.zeros((64, 64), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            checker[y, x] = 0 if ((x // 8) + (y // 8)) % 2 == 0 else 255
    Image.fromarray(checker).save(os.path.join(output_dir, "test_checker.png"))
    print(f"[OK] Images created in {output_dir}")


if __name__ == "__main__":
    sd = os.path.dirname(os.path.abspath(__file__))
    create_sample_text(os.path.join(sd, "sample.txt"))
    create_sample_images(sd)
