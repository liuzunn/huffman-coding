#!/usr/bin/env python3
"""Comprehensive Huffman Coding Experiments.
Text: short/medium/long, English/Chinese. Image: gray/color, 64/128/256.
Output: CSV tables + 3 charts in results/."""

import csv, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from src.encoder import TextEncoder
from src.huffman import compute_entropy, build_frequency_table
from src.image_codec import ImageEncoder
from src.utils import read_text_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TABLES = os.path.join(ROOT, "results", "tables")
FIGURES = os.path.join(ROOT, "results", "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_image_cv2(path, gray):
    """Unicode-safe image read (cv2.imread fails on Chinese paths)."""
    import cv2, numpy as np
    with open(path, "rb") as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return cv2.imdecode(file_bytes, flag)


plt.rcParams.update({"figure.dpi":150,"font.size":10,"axes.titlesize":12})
C = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B","#16A085"]

def gen_text():
    texts = {}
    en = "The quick brown fox jumps over the lazy dog. "
    texts["English Short"] = en * 3
    texts["English Medium"] = ("Information theory by Claude Shannon (1948) provides "
        "the framework for data compression and communication. ") * 5
    texts["English Long"] = ("Huffman coding (1952) produces optimal prefix codes "
        "by building a binary tree bottom-up, merging lowest-frequency nodes. "
        "It achieves average code length within one bit of Shannon entropy. "
        "Used in DEFLATE (gzip, PNG), JPEG, and many more standards. ") * 4
    cn = chr(21704)+chr(22827)+chr(23572)+chr(32593)+chr(32534)+chr(30721)+chr(26159)+chr(19968)+chr(31181)+chr(20248)+chr(21069)
    texts["Chinese Short"] = cn
    texts["Chinese Medium"] = (cn + chr(12290)) * 3
    texts["Chinese Long"] = (cn + chr(12290)) * 5
    for n, t in texts.items():
        p = os.path.join(DATA, n.replace(" ","_")+".txt")
        with open(p,"w",encoding="utf-8") as f:
            f.write(t)
    print(f"[OK] {len(texts)} text samples")

def gen_images():
    for sz, name in [(64,"gray_64"),(128,"gray_128"),(256,"gray_256")]:
        img = np.zeros((sz,sz), dtype=np.uint8)
        cy, cx = sz//2, sz//2
        for y in range(sz):
            for x in range(sz):
                d = np.sqrt((x-cx)**2+(y-cy)**2)/(sz*0.7)
                img[y,x] = max(0, min(255, int(255*(1-d))))
        img[sz//4:sz//2, sz//4:sz//2] = 200
        cv2.imwrite(os.path.join(DATA, name+".png"), img)
    print(f"[OK] image samples")

def encode_text(name, path):
    text = read_text_file(path)
    enc = TextEncoder()
    out = os.path.join(TABLES, name.replace(" ","_")+".huf")
    r = enc.encode(text, out)
    return {"name":name,"type":"Text","chars":r["input_chars"],
        "unique":r["unique_symbols"],"entropy":r["entropy"],
        "avg_len":r["avg_length"],"efficiency":r["efficiency"],
        "orig_bits":r["input_chars"]*8,"comp_bits":r["compressed_size"]*8,
        "ratio":r["compression_ratio"]}

def encode_image(name, path, gray):
    enc = ImageEncoder()
    out = os.path.join(TABLES, name.replace(" ","_")+".huf")
    r = enc.encode(path, out, grayscale=gray, mode="direct")
    img = read_image_cv2(path, gray)
    if img is None or img.size==0:
        return None
    npx = img.shape[0]*img.shape[1]
    return {"name":name,"type":"Image (Gray)" if gray else "Image (Color)",
        "chars":npx,"unique":256,"entropy":r["entropy"],
        "avg_len":r["avg_length"],"efficiency":r["efficiency"],
        "orig_bits":npx*8,"comp_bits":r["compressed_size"]*8,
        "ratio":r["compression_ratio"]}

def run_all():
    print("="*65)
    print("  HUFFMAN CODING - COMPREHENSIVE EXPERIMENTS")
    print("="*65)
    gen_text()
    gen_images()
    results = []
    print("\n--- Text Compression ---")
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".txt"):
            continue
        n = f.replace(".txt","").replace("_"," ")
        if n.lower() in ["sample","setup"]:
            continue
        r = encode_text(n, os.path.join(DATA,f))
        results.append(r)
        print(f'  {n:25s} {r["chars"]:5d} ch H={r["entropy"]:.2f} L={r["avg_len"]:.2f} eta={r["efficiency"]:.1f}% ratio={r["ratio"]:.3f}x')
    print("\n--- Image Compression ---")
    for label,fname,gray in [("Gray 64x64","gray_64.png",True),
        ("Gray 128x128","gray_128.png",True),("Gray 256x256","image.png",True),
        ("Color 256x256","image_color.png",False)]:
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            continue
        r = encode_image(label, p, gray)
        if r:
            results.append(r)
            print(f'  {label:25s} {r["chars"]:6d} px H={r["entropy"]:.2f} L={r["avg_len"]:.2f} eta={r["efficiency"]:.1f}% ratio={r["ratio"]:.3f}x')
    print("\n--- Theoretical Analysis ---")
    for r in results:
        fb = max(1, math.ceil(math.log2(max(2, r["unique"]))))
        r["fixed_bits"] = fb
        r["ratio_vs_fixed"] = (r["chars"]*fb)/r["comp_bits"] if r["comp_bits"]>0 else 0
        print(f'  {r["name"]:25s} H={r["entropy"]:.4f} L={r["avg_len"]:.4f} eta={r["efficiency"]:.2f}% ratio={r["ratio"]:.4f}x vsFixed={r["ratio_vs_fixed"]:.4f}x')
    print("\n--- Generating Charts ---")
    names = [r["name"] for r in results]
    ns = [(n[:15]+"..") if len(n)>15 else n for n in names]
    x = range(len(results))
    fig, ax = plt.subplots(figsize=(10,5))
    w = 0.35
    b1 = ax.bar([i-w/2 for i in x], [r["ratio"] for r in results], w, label="vs 8-bit", color=C[0])
    b2 = ax.bar([i+w/2 for i in x], [r["ratio_vs_fixed"] for r in results], w, label="vs Fixed-Length", color=C[1])
    ax.axhline(1, color="gray", ls="--", alpha=0.5, label="Break-even")
    ax.set_xticks(list(x)); ax.set_xticklabels(ns, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Compression Ratio"); ax.set_title("Compression Ratio: Huffman Coding")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.2f}x", ha="center", va="bottom", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES,"fig_compression_ratio.png"), bbox_inches="tight"); plt.close()
    print("  [OK] fig_compression_ratio.png")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar([i-w/2 for i in x], [r["entropy"] for r in results], w, label="Entropy H(X)", color=C[2])
    ax.bar([i+w/2 for i in x], [r["avg_len"] for r in results], w, label="Avg Code Length L", color=C[3])
    ax.set_xticks(list(x)); ax.set_xticklabels(ns, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Bits / Symbol"); ax.set_title("Shannon Entropy vs Average Huffman Code Length")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES,"fig_entropy_vs_length.png"), bbox_inches="tight"); plt.close()
    print("  [OK] fig_entropy_vs_length.png")
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(x, [r["efficiency"] for r in results], color=[C[i%len(C)] for i in x])
    for bar, eff in zip(bars, [r["efficiency"] for r in results]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{eff:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(ns, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Efficiency (%)"); ax.set_title("Huffman Coding Efficiency")
    ax.set_ylim(0,105); ax.axhline(100, color="gray", ls="--", alpha=0.5, label="Optimal")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES,"fig_efficiency.png"), bbox_inches="tight"); plt.close()
    print("  [OK] fig_efficiency.png")
    csvpath = os.path.join(TABLES, "experiments_summary.csv")
    with open(csvpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source","Type","Symbols","Unique","Entropy","AvgLen","Eff%","OrigBits","CompBits","Ratio","vsFixed"])
        for r in results:
            w.writerow([r["name"],r["type"],r["chars"],r["unique"],
                f'{r["entropy"]:.4f}',f'{r["avg_len"]:.4f}',f'{r["efficiency"]:.2f}',
                r["orig_bits"],r["comp_bits"],f'{r["ratio"]:.4f}',f'{r["ratio_vs_fixed"]:.4f}'])
    print(f"  [OK] {csvpath}")

    # Also write experiments_all.csv (subset used by chart generation)
    all_csv = os.path.join(TABLES, "experiments_all.csv")
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source","Type","Symbols","Unique","Entropy","AvgLen","Eff%","Ratio","vsFixed"])
        for r in results:
            w.writerow([r["name"],r["type"],r["chars"],r["unique"],
                f'{r["entropy"]:.4f}',f'{r["avg_len"]:.4f}',f'{r["efficiency"]:.2f}',
                f'{r["ratio"]:.4f}',f'{r["ratio_vs_fixed"]:.4f}'])
    print(f"  [OK] {all_csv}")
    print("\n" + "="*65 + "\n  EXPERIMENTS COMPLETE\n" + "="*65)

if __name__ == "__main__":
    run_all()
