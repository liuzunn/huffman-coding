"""
Text compression experiments.
Encodes text samples with Huffman coding, records metrics, and saves
summary tables for report generation.
"""

import os
import csv
from typing import Dict, List, Tuple

from src.utils import read_text_file, format_size
from src.encoder import TextEncoder


def run_text_experiments(root_dir: str) -> List[Dict]:
    """Run text compression experiments on all sample texts.

    Args:
        root_dir: Project root directory.

    Returns:
        List of result dicts, one per text file, with keys:
        name, chars, unique, entropy, avg_len, efficiency, ratio, orig, comp
    """
    data_dir = os.path.join(root_dir, 'data')
    results_dir = os.path.join(root_dir, 'results', 'tables')
    os.makedirs(results_dir, exist_ok=True)

    # Discover all text samples
    text_files = {}
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.txt'):
            label = f.replace('_', ' ').replace('.txt', '').title()
            text_files[label] = os.path.join(data_dir, f)

    if not text_files:
        print("  No text files found in data/. Run 'python main.py setup' first.")
        return []

    results = []
    headers = ['Name', 'Chars', 'Unique', 'Entropy', 'AvgLen',
               'Efficiency%', 'Ratio', 'Orig(B)', 'Comp(B)']

    for name, path in text_files.items():
        try:
            text = read_text_file(path)
            encoder = TextEncoder()

            out_path = os.path.join(results_dir, f"{name.replace(' ', '_')}.huf")
            report = encoder.encode(text, out_path)

            row = {
                'name': name,
                'chars': report['input_chars'],
                'unique': report['unique_symbols'],
                'entropy': report['entropy'],
                'avg_len': report['avg_length'],
                'efficiency': report['efficiency'],
                'ratio': report['compression_ratio'],
                'orig_size': report['original_size'],
                'comp_size': report['compressed_size'],
            }
            results.append(row)
            print(report['summary'])
        except Exception as e:
            print(f"  ! Error encoding {name}: {e}")

    # Save CSV table
    if results:
        csv_path = os.path.join(results_dir, 'text_experiments.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in results:
                writer.writerow([
                    r['name'],
                    r['chars'],
                    r['unique'],
                    f"{r['entropy']:.4f}",
                    f"{r['avg_len']:.4f}",
                    f"{r['efficiency']:.2f}",
                    f"{r['ratio']:.4f}",
                    r['orig_size'],
                    r['comp_size'],
                ])
        print(f"  Table saved: {csv_path}")

    return results


def compute_text_statistics(report: Dict) -> Dict:
    """Extract statistics from a text compression report for charting."""
    return {
        'entropy': report['entropy'],
        'avg_length': report['avg_length'],
        'efficiency': report['efficiency'],
        'compression_ratio': report['compression_ratio'],
        'ratio_vs_fixed': report.get('ratio_vs_fixed', 0),
    }
