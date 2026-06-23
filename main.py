#!/usr/bin/env python3
"""
CLI entry point for the Huffman Coding System.

Supports:
  encode-text   鈥?Compress a text file using Huffman coding
  decode-text   鈥?Decompress a .huf file back to plain text
  encode-image  鈥?Compress an image using Huffman coding (+ optional DPCM)
  decode-image  鈥?Decompress a .huf image file back to a viewable image
  analyze       鈥?Display information-theoretic analysis of a file
  run-experiments 鈥?Execute all experiments and generate figures
  generate-report 鈥?Produce the course-design report (Word + LaTeX)
  setup         鈥?Generate sample text and image data
"""

import argparse
import os
import sys


def _resolve_project_root() -> str:
    """Find the project root directory (where main.py lives)."""
    return os.path.dirname(os.path.abspath(__file__))


def _cmd_setup(args):
    """Generate sample data for experiments."""
    sys.path.insert(0, _resolve_project_root())
    from data import setup
    setup.create_sample_text(os.path.join(_resolve_project_root(), 'data', 'sample.txt'))
    setup.create_sample_images(os.path.join(_resolve_project_root(), 'data'))
    print("\nSample data ready in data/")


def _cmd_encode_text(args):
    """Compress a text file with Huffman coding."""
    sys.path.insert(0, _resolve_project_root())
    from src.encoder import TextEncoder
    from src.utils import read_text_file

    text = read_text_file(args.input, encoding=args.encoding)
    print(f"Read {len(text)} characters from {args.input}")

    encoder = TextEncoder()
    report = encoder.encode(text, args.output, use_canonical=True)
    print(report["summary"])


def _cmd_decode_text(args):
    """Decompress a .huf file to plain text."""
    sys.path.insert(0, _resolve_project_root())
    from src.decoder import TextDecoder

    decoder = TextDecoder()
    n_chars = decoder.decode_to_file(args.input, args.output)
    print(f"Decoded {n_chars} characters 鈫?{args.output}")


def _cmd_encode_image(args):
    """Compress an image with Huffman coding."""
    sys.path.insert(0, _resolve_project_root())
    from src.image_codec import ImageEncoder

    encoder = ImageEncoder()
    report = encoder.encode(args.input, args.output,
                            grayscale=args.grayscale, mode=args.mode)
    print(report.get("summary", str(report)))


def _cmd_decode_image(args):
    """Decompress a .huf image back to a viewable image."""
    sys.path.insert(0, _resolve_project_root())
    from src.image_codec import ImageDecoder
    from src.utils import read_image, write_image

    # Get shape from --shape arg, or fall back to reading the original PNG
    if args.shape:
        parts = [int(x) for x in args.shape.split('x')]
        shape = tuple(parts)
    else:
        original = read_image(args.input.replace('.huf', '.png'), grayscale=args.grayscale)
        shape = original.shape

    decoder = ImageDecoder()
    img = decoder.decode(args.input, shape,
                         grayscale=args.grayscale, mode=args.mode)
    write_image(args.output, img)
    print(f"Decoded image 鈫?{args.output} (shape: {img.shape})")


def _cmd_analyze(args):
    """Analyze the information-theoretic properties of a file."""
    sys.path.insert(0, _resolve_project_root())
    from src.huffman import (
        build_frequency_table, compute_entropy,
        theoretical_min_code_length
    )
    from src.utils import format_size

    file_size = os.path.getsize(args.input)

    if args.type == 'text':
        from src.utils import read_text_file
        data = read_text_file(args.input)
        symbols = [ord(c) for c in data]
    else:  # image
        from src.utils import read_image
        img = read_image(args.input, grayscale=args.grayscale)
        symbols = img.ravel().tolist()

    freq = build_frequency_table(symbols)
    total = len(symbols)
    entropy = compute_entropy(freq, total)
    n_unique = len(freq)
    fixed_len = theoretical_min_code_length(n_unique)

    print("=" * 60)
    print(f"  Information-Theoretic Analysis")
    print("=" * 60)
    print(f"  File             : {os.path.basename(args.input)}")
    print(f"  File size        : {format_size(file_size)}")
    print(f"  Total symbols    : {total}")
    print(f"  Unique symbols   : {n_unique}")
    print(f"  Shannon entropy  : {entropy:.4f} bits/symbol")
    print(f"  Fixed-length min : {fixed_len} bits/symbol")
    print(f"  Max entropy      : {fixed_len:.4f} bits/symbol (uniform)")
    print(f"  Redundancy       : {fixed_len - entropy:.4f} bits/symbol")
    print("-" * 60)


def _cmd_run_experiments(args):
    """Run all compression experiments and generate figures."""
    # First ensure sample data exists
    data_dir = os.path.join(_resolve_project_root(), 'data')
    if not os.path.exists(os.path.join(data_dir, 'sample.txt')):
        print("Sample data not found. Running setup first...")
        _cmd_setup(None)

    sys.path.insert(0, _resolve_project_root())
    from experiments.comprehensive_experiments import run_all
    from experiments.generate_report_charts import generate_all_charts

    print("\n" + "=" * 60)
    print("  COMPREHENSIVE HUFFMAN CODING EXPERIMENTS")
    print("=" * 60)
    run_all()

    print("\n" + "=" * 60)
    print("  GENERATING REPORT CHARTS")
    print("=" * 60)
    generate_all_charts(_resolve_project_root())

    results_dir = os.path.join(_resolve_project_root(), 'results')
    print(f"\nAll results saved to: {results_dir}")


def _cmd_generate_report(args):
    """Generate the course design report (Word + LaTeX)."""
    sys.path.insert(0, _resolve_project_root())

    # Generate Word report
    print("Generating Word report...")
    from report.generate_word_report import create_word_report
    output_dir = os.path.join(_resolve_project_root(), 'report', 'generated')
    os.makedirs(output_dir, exist_ok=True)
    docx_path = os.path.join(output_dir, 'report.docx')
    create_word_report(docx_path, _resolve_project_root())
    print(f"  鉁?Word report: {docx_path}")

    print("\nLaTeX report template available at: report/report_template.tex")
    print("Compile with: xelatex report_template.tex && bibtex report_template && xelatex report_template.tex")


def main():
    parser = argparse.ArgumentParser(
        description="Huffman Coding System 鈥?Text & Image Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # setup
    p_setup = subparsers.add_parser('setup', help='Generate sample text/image data')
    p_setup.set_defaults(func=_cmd_setup)

    # encode-text
    p_et = subparsers.add_parser('encode-text', help='Compress a text file')
    p_et.add_argument('--input', required=True, help='Input .txt file')
    p_et.add_argument('--output', required=True, help='Output .huf file')
    p_et.add_argument('--encoding', default='utf-8', help='Text encoding (default: utf-8)')
    p_et.set_defaults(func=_cmd_encode_text)

    # decode-text
    p_dt = subparsers.add_parser('decode-text', help='Decompress a text file')
    p_dt.add_argument('--input', required=True, help='Input .huf file')
    p_dt.add_argument('--output', required=True, help='Output .txt file')
    p_dt.set_defaults(func=_cmd_decode_text)

    # encode-image
    p_ei = subparsers.add_parser('encode-image', help='Compress an image')
    p_ei.add_argument('--input', required=True, help='Input image file')
    p_ei.add_argument('--output', required=True, help='Output .huf file (base name)')
    p_ei.add_argument('--mode', choices=['direct', 'dpcm'], default='direct',
                      help='Encoding mode (default: direct)')
    p_ei.add_argument('--grayscale', action='store_true',
                      help='Convert to grayscale before encoding')
    p_ei.set_defaults(func=_cmd_encode_image)

    # decode-image
    p_di = subparsers.add_parser('decode-image', help='Decompress an image')
    p_di.add_argument('--input', required=True, help='Input .huf file (base)')
    p_di.add_argument('--output', required=True, help='Output image file')
    p_di.add_argument('--mode', choices=['direct', 'dpcm'], default='direct',
                      help='Decoding mode (default: direct)')
    p_di.add_argument('--shape', default='',
                      help='Image shape e.g. 256x256 or 256x256x3')
    p_di.add_argument('--grayscale', action='store_true',
                      help='Image is grayscale')
    p_di.set_defaults(func=_cmd_decode_image)

    # analyze
    p_an = subparsers.add_parser('analyze', help='Analyze file entropy')
    p_an.add_argument('--input', required=True, help='Input file')
    p_an.add_argument('--type', choices=['text', 'image'], required=True,
                      help='File type')
    p_an.add_argument('--grayscale', action='store_true',
                      help='For images: treat as grayscale')
    p_an.set_defaults(func=_cmd_analyze)

    # run-experiments
    p_re = subparsers.add_parser('run-experiments',
                                 help='Run all experiments and generate figures')
    p_re.set_defaults(func=_cmd_run_experiments)

    # generate-report
    p_gr = subparsers.add_parser('generate-report',
                                 help='Generate course design report')
    p_gr.set_defaults(func=_cmd_generate_report)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch
    args.func(args)


if __name__ == '__main__':
    main()

