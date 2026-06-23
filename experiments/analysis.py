"""
Statistical analysis functions for compression experiments.
Provides helper functions for computing comparisons and formatting
results for reports.
"""

from typing import Dict, List, Tuple


def build_comparison_table(text_results: List[Dict],
                            image_results: List[Dict]) -> str:
    """Build a markdown comparison table for the report."""
    lines = [
        "## Compression Performance Summary",
        "",
        "### Text Compression",
        "",
        "| Source | Symbols | Unique | Entropy | Avg Len | Eff(%) | Ratio |",
        "|--------|---------|--------|---------|---------|--------|-------|",
    ]

    for r in text_results:
        lines.append(
            f"| {r['name']} | {r['chars']} | {r['unique']} "
            f"| {r['entropy']:.4f} | {r['avg_length']:.4f} "
            f"| {r['efficiency']:.2f} | {r['ratio']:.4f}x |"
        )

    lines += [
        "",
        "### Image Compression",
        "",
        "| Source | Mode | Size | Entropy | Avg Len | Eff(%) | Ratio | PSNR |",
        "|--------|------|------|---------|---------|--------|-------|------|",
    ]

    for r in image_results:
        sz = f"{r['width']}x{r['height']}x{r['channels']}"
        psnr = 'inf' if r['psnr'] == float('inf') else f"{r['psnr']:.2f}dB"
        lines.append(
            f"| {r['label']} | {r['mode'].upper()} | {sz} "
            f"| {r['entropy']:.4f} | {r['avg_length']:.4f} "
            f"| {r['efficiency']:.2f} | {r['ratio']:.4f}x | {psnr} |"
        )

    lines.append("")
    return "\n".join(lines)


def summarize_efficiency(text_results: List[Dict],
                          image_results: List[Dict]) -> str:
    """Generate a narrative summary of compression efficiency."""
    if text_results:
        avg_eff_txt = sum(r['efficiency'] for r in text_results) / len(text_results)
    else:
        avg_eff_txt = 0.0

    if image_results:
        avg_eff_img = sum(r['efficiency'] for r in image_results) / len(image_results)
    else:
        avg_eff_img = 0.0

    lines = [
        "## Efficiency Analysis",
        "",
        f"Average coding efficiency for text: {avg_eff_txt:.2f}%",
        f"Average coding efficiency for images: {avg_eff_img:.2f}%",
        "",
        "Huffman coding achieves near-optimal compression for text due to the",
        "highly skewed character frequency distribution. For images, DPCM mode",
        "reduces entropy by decorrelating neighboring pixels, resulting in",
        "shorter average code lengths and higher compression ratios.",
    ]
    return "\n".join(lines)
