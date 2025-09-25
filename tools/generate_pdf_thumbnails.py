"""Generate poster thumbnails from PDF files using PyMuPDF.

Usage examples:
    python tools/generate_pdf_thumbnails.py \
        --input-dir papers \
        --output-dir poster_thumbnails \
        --width 512 --height 512

Requires PyMuPDF (install with `pip install pymupdf`).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

try:
    import fitz  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - clear guidance for users
    raise SystemExit(
        "PyMuPDF (fitz) is required. Install it with `pip install pymupdf`."
    ) from exc


@dataclass(frozen=True)
class ThumbnailConfig:
    input_dir: Path
    output_dir: Path
    size: Tuple[int, int]
    page: int
    suffix: str = ""


def parse_args(argv: Iterable[str]) -> ThumbnailConfig:
    parser = argparse.ArgumentParser(description="Generate PNG thumbnails from PDF posters.")
    parser.add_argument(
        "--input-dir",
        default="papers",
        type=Path,
        help="Directory containing source PDF files (default: papers)",
    )
    parser.add_argument(
        "--output-dir",
        default="poster_thumbnails",
        type=Path,
        help="Directory where thumbnails will be written (default: poster_thumbnails)",
    )
    parser.add_argument(
        "--width",
        default=512,
        type=int,
        help="Target thumbnail width in pixels (default: 512)",
    )
    parser.add_argument(
        "--height",
        default=512,
        type=int,
        help="Target thumbnail height in pixels (default: 512)",
    )
    parser.add_argument(
        "--page",
        default=0,
        type=int,
        help="Zero-based page index to render (default: 0)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="Optional suffix appended to the generated filename (default: '')",
    )

    args = parser.parse_args(list(argv))
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive integers")
    if args.page < 0:
        parser.error("page index cannot be negative")

    return ThumbnailConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=(args.width, args.height),
        page=args.page,
        suffix=args.suffix,
    )


def slugify(value: str) -> str:
    """Convert filename stems into URL and filesystem friendly slugs."""
    normalized = re.sub(r"[\s_]+", "-", value.strip().lower())
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    return normalized or "poster"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def render_thumbnail(pdf_path: Path, cfg: ThumbnailConfig) -> Path:
    thumb_stem = slugify(pdf_path.stem)
    if cfg.suffix:
        thumb_stem = f"{thumb_stem}-{slugify(cfg.suffix)}"

    output_path = cfg.output_dir / f"{thumb_stem}.png"

    with fitz.open(pdf_path) as document:
        if cfg.page >= len(document):
            raise ValueError(
                f"PDF '{pdf_path.name}' has only {len(document)} pages; page {cfg.page} not available"
            )

        page = document.load_page(cfg.page)
        rect = page.rect
        scale_x = cfg.size[0] / rect.width
        scale_y = cfg.size[1] / rect.height
        zoom_matrix = fitz.Matrix(scale_x, scale_y)
        pix = page.get_pixmap(matrix=zoom_matrix, annots=False)
        pix.save(output_path)

    return output_path


def main(argv: Iterable[str] | None = None) -> int:
    cfg = parse_args(argv or sys.argv[1:])

    if not cfg.input_dir.exists():
        raise SystemExit(f"Input directory '{cfg.input_dir}' does not exist")

    ensure_output_dir(cfg.output_dir)

    pdf_files = sorted(p for p in cfg.input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {cfg.input_dir.resolve()}")
        return 0

    print(f"Generating thumbnails from {len(pdf_files)} PDF(s) in '{cfg.input_dir}' → '{cfg.output_dir}'")

    success = 0
    for pdf_file in pdf_files:
        try:
            output_path = render_thumbnail(pdf_file, cfg)
            print(f"  ✓ {pdf_file.name} → {output_path.relative_to(cfg.output_dir.parent)}")
            success += 1
        except Exception as error:  # pylint: disable=broad-except
            print(f"  ✗ {pdf_file.name}: {error}")

    print(f"Done. {success}/{len(pdf_files)} thumbnail(s) generated.")
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
