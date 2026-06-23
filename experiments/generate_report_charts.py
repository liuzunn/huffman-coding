#!/usr/bin/env python3
"""Professional publication-quality chart generation for Huffman coding analysis.

Charts:
  1. fig1_compression_ratio.png — Grouped bar: Huffman vs Fixed-Length vs 8-bit
  2. fig2_entropy_vs_length.png  — Entropy H(X) vs Average Code Length L
  3. fig3_pixel_histogram.png    — Grayscale pixel value distribution

Output: results/figures/
"""

import csv, math, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
#  Publication-grade style configuration
# ============================================================
rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.figsize": (7, 4.5),
    "figure.subplot.left": 0.12,
    "figure.subplot.bottom": 0.12,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "tables", "experiments_all.csv")
IMG = os.path.join(ROOT, "data", "image.png")
OUT = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True)

# Color palette: colorblind-friendly
C_BLUE = "#2E86AB"
C_RED = "#C73E1D"
C_GREEN = "#16A085"
C_PURPLE = "#A23B72"
C_ORANGE = "#F18F01"
C_GRAY = "#95A5A6"
HATCH = ["//", "\\", "xx", "..", "++", "oo"]


# ============================================================
#  Load experimental data
# ============================================================
def load_data():
    """Read experiment results from CSV."""
    rows = []
    if not os.path.exists(CSV):
        return rows
    with open(CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Entropy"] = float(row["Entropy"])
            row["AvgLen"] = float(row["AvgLen"])
            row["Ratio"] = float(row["Ratio"])
            row["Eff%"] = float(row["Eff%"])
            row["vsFixed"] = float(row.get("vsFixed", 0))
            rows.append(row)
    return rows


# ============================================================
#  Figure 1: Compression Ratio Comparison
# ============================================================
def plot_compression_ratio(data):
    """Grouped bar chart: Huffman vs 8-bit vs Fixed-length coding."""
    fig, ax = plt.subplots()

    # Use shortened names for x-axis labels
    labels = [r["Source"][:18] + ".." if len(r["Source"]) > 18 else r["Source"]
              for r in data]
    x = np.arange(len(data))
    width = 0.28

    # Bars
    bars_huf = ax.bar(x - width, [r["Ratio"] for r in data], width,
                      label="Huffman vs 8-bit",
                      color=C_BLUE, edgecolor="white", linewidth=0.5)
    bars_fix = ax.bar(x, [r["vsFixed"] for r in data], width,
                      label="Huffman vs Fixed-Length",
                      color=C_ORANGE, edgecolor="white", linewidth=0.5)

    # Break-even line
    ax.axhline(y=1.0, color=C_GRAY, linestyle="--", linewidth=0.8,
               label="Break-even (ratio = 1)")

    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression Ratio: Huffman Coding Performance", pad=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max([r["Ratio"] for r in data] + [1.2]) * 1.25)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Value labels on Huffman bars
    for bar in bars_huf:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.3f}x", ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    path = os.path.join(OUT, "fig1_compression_ratio.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ============================================================
#  Figure 2: Entropy vs Average Code Length
# ============================================================
def plot_entropy_vs_length(data):
    """Dual bar chart: theoretical entropy vs actual code length."""
    fig, ax = plt.subplots()

    labels = [r["Source"][:18] + ".." if len(r["Source"]) > 18 else r["Source"]
              for r in data]
    x = np.arange(len(data))
    width = 0.35

    # Entropy bars (theoretical minimum)
    bars_ent = ax.bar(x - width / 2, [r["Entropy"] for r in data], width,
                      label="Entropy H(X)",
                      color=C_GREEN, edgecolor="white", linewidth=0.5)

    # Average code length bars (actual performance)
    bars_len = ax.bar(x + width / 2, [r["AvgLen"] for r in data], width,
                      label="Avg Code Length L",
                      color=C_PURPLE, edgecolor="white", linewidth=0.5)

    # Efficiency annotations
    for i, r in enumerate(data):
        gap = r["AvgLen"] - r["Entropy"]
        ax.annotate(f"gap={gap:.3f}",
                    xy=(i + width / 2, r["AvgLen"]),
                    xytext=(i + width / 2, r["AvgLen"] + 0.12),
                    ha="center", va="bottom", fontsize=6, color=C_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Bits / Symbol")
    ax.set_title("Shannon Entropy vs Average Huffman Code Length", pad=10)
    ax.legend(fontsize=8)
    max_h = max([max(r["Entropy"], r["AvgLen"]) for r in data])
    ax.set_ylim(0, max_h * 1.25)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUT, "fig2_entropy_vs_length.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ============================================================
#  Figure 3: Pixel Histogram
# ============================================================
def plot_pixel_histogram():
    """Histogram of grayscale pixel values from sample image."""
    try:
        from PIL import Image
        img = np.array(Image.open(IMG).convert("L"))
    except Exception:
        import cv2
        img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
        if img is None:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(IMG).convert("L"))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Histogram with fine bins
    counts, bins, patches = ax.hist(img.ravel(), bins=256, range=(0, 255),
                                    color=C_BLUE, alpha=0.85,
                                    edgecolor="white", linewidth=0.3)

    # Statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    entropy = 0.0
    pdf = counts / counts.sum()
    for p in pdf[pdf > 0]:
        entropy -= p * math.log2(p)

    # Annotate statistics
    stats_text = (f"Mean = {mean_val:.1f}\n"
                  f"Std Dev = {std_val:.1f}\n"
                  f"Entropy H = {entropy:.3f} bits/pixel")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=C_GRAY, alpha=0.9))

    # Labels
    ax.set_xlabel("Pixel Value (0-255)")
    ax.set_ylabel("Frequency")
    ax.set_title("Grayscale Pixel Value Distribution", pad=10)
    ax.set_xlim(0, 255)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUT, "fig3_pixel_histogram.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ============================================================
#  Main
# ============================================================
def generate_all_charts(project_root=None):
    """Entry point called by main.py for one-shot chart generation."""
    global ROOT, CSV, OUT
    if project_root:
        ROOT = project_root
        CSV = os.path.join(ROOT, "results", "tables", "experiments_all.csv")
        OUT = os.path.join(ROOT, "results", "figures")
        os.makedirs(OUT, exist_ok=True)
    
    print("Generating publication-quality charts...\n")
    if os.path.exists(CSV):
        data = load_data()
        print(f"Loaded {len(data)} experiment data points from {CSV}")
    else:
        print(f"[WARN] CSV not found: {CSV}")
        data = []
    
    if data:
        plot_compression_ratio(data)
        plot_entropy_vs_length(data)
    else:
        print("[WARN] Skipping figures 1-2: no data")
    
    plot_pixel_histogram()
    print("\n[OK] All charts generated in", OUT)


if __name__ == "__main__":
    print("Generating publication-quality charts...\n")

    # Load experimental data
    if os.path.exists(CSV):
        data = load_data()
        print(f"Loaded {len(data)} experiment data points from {CSV}")
    else:
        print(f"[WARN] CSV not found: {CSV}")
        data = []

    # Generate figures
    if data:
        plot_compression_ratio(data)
        plot_entropy_vs_length(data)
    else:
        print("[WARN] Skipping figures 1-2: no data")

    plot_pixel_histogram()

    print("\n[OK] All charts generated in", OUT)