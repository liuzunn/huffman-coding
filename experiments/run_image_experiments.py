"""
Image compression experiments.
Encodes images (grayscale/color) with Huffman coding in direct and DPCM modes,
records metrics, and saves summary tables.
"""

import os
import csv
from typing import Dict, List

from src.utils import read_image, format_size
from src.image_codec import ImageEncoder


def run_image_experiments(root_dir: str) -> List[Dict]:
    """Run image compression experiments on sample images.

    Args:
        root_dir: Project root directory.

    Returns:
        List of result dicts with compression metrics per experiment.
    """
    data_dir = os.path.join(root_dir, 'data')
    results_dir = os.path.join(root_dir, 'results', 'tables')
    os.makedirs(results_dir, exist_ok=True)

    # Image candidates: (label, path, grayscale, mode)
    candidates = [
        ('Gray Direct', os.path.join(data_dir, 'image.png'), True, 'direct'),
        ('Gray DPCM',   os.path.join(data_dir, 'image.png'), True, 'dpcm'),
    ]

    # Add color image if available
    color_path = os.path.join(data_dir, 'image_color.png')
    if os.path.exists(color_path):
        candidates.append(('Color Direct', color_path, False, 'direct'))
        candidates.append(('Color DPCM', color_path, False, 'dpcm'))

    results = []

    for label, path, gray, mode in candidates:
        if not os.path.exists(path):
            continue
        try:
            encoder = ImageEncoder()
            out_base = os.path.join(
                results_dir,
                f"{label.replace(' ', '_').lower()}.huf"
            )
            report = encoder.encode(path, out_base, grayscale=gray, mode=mode)

            row = {
                'label': label,
                'mode': mode,
                'gray': gray,
                'width': report.get('width', '?'),
                'height': report.get('height', '?'),
                'channels': report.get('channels', 1),
                'entropy': report.get('entropy', 0),
                'avg_length': report.get('avg_length', 0),
                'efficiency': report.get('efficiency', 0),
                'ratio': report.get('compression_ratio', 0),
                'orig_size': report.get('original_size', 0),
                'comp_size': report.get('compressed_size', 0),
                'psnr': report.get('psnr', float('inf')),
            }
            results.append(row)
            print(report.get('summary', str(report)))

        except Exception as e:
            print(f"  ! Error compressing {label}: {e}")

    # Save CSV table
    if results:
        csv_path = os.path.join(results_dir, 'image_experiments.csv')
        headers = ['Label', 'Mode', 'Size', 'Entropy', 'AvgLen',
                   'Eff%', 'Ratio', 'PSNR']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in results:
                sz = f"{r['width']}x{r['height']}x{r['channels']}"
                psnr_str = 'inf' if r['psnr'] == float('inf') else f"{r['psnr']:.2f}"
                writer.writerow([
                    r['label'],
                    r['mode'].upper(),
                    sz,
                    f"{r['entropy']:.4f}",
                    f"{r['avg_length']:.4f}",
                    f"{r['efficiency']:.2f}",
                    f"{r['ratio']:.4f}",
                    psnr_str,
                ])
        print(f"  Table saved: {csv_path}")

    return results
