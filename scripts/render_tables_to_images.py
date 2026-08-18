"""Render the MMQA table corpus into PNG images (used as table modality inputs)."""

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Missing dependency Pillow. Run: pip install pillow") from exc


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def measure(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> int:
    if not text:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return max(0, int(right) - int(left))


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    text = safe_text(text)
    if not text:
        return [""]

    if measure(draw, text, font) <= max_width:
        return [text]

    words = text.split(" ")
    if len(words) <= 1:
        # Split spaceless text by character to avoid overflow.
        lines: list[str] = []
        cur = ""
        for ch in text:
            candidate = f"{cur}{ch}"
            if cur and measure(draw, candidate, font) > max_width:
                lines.append(cur)
                cur = ch
            else:
                cur = candidate
        if cur:
            lines.append(cur)
        return lines or [""]

    lines = []
    current = words[0]
    for w in words[1:]:
        candidate = f"{current} {w}"
        if measure(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def choose_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    # Try common fonts, fall back to the Pillow default font.
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "simhei.ttf",
        "msyh.ttc",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=18)
        except OSError:
            continue
    return ImageFont.load_default()


def to_table_matrix(record: dict) -> list[list[str]]:
    table = record.get("table", {})
    headers: list[str] = []
    for h in table.get("header", []) or []:
        if isinstance(h, dict):
            headers.append(safe_text(h.get("column_name", "")))
        else:
            headers.append(safe_text(h))

    rows: list[list[str]] = []
    for row in table.get("table_rows", []) or []:
        row_text = []
        for cell in row:
            if isinstance(cell, dict):
                row_text.append(safe_text(cell.get("text", "")))
            else:
                row_text.append(safe_text(cell))
        rows.append(row_text)

    col_count = max(len(headers), *(len(r) for r in rows), 0)
    if col_count == 0:
        return []

    if not headers:
        headers = [f"Column {i + 1}" for i in range(col_count)]
    headers = headers + [""] * (col_count - len(headers))

    padded_rows = [r + [""] * (col_count - len(r)) for r in rows]
    return [headers] + padded_rows


def render_table_image(
    matrix: Sequence[Sequence[str]],
    out_path: Path,
    title: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    min_col_width: int = 120,
    max_col_width: int = 520,
) -> None:
    scratch = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(scratch)

    padding_x = 10
    padding_y = 8
    line_spacing = 6
    border = 2
    row_line_width = 1
    title_gap = 12
    title_font = font
    _, _, _, title_h = draw.textbbox((0, 0), "Hg", font=title_font)
    _, _, _, line_h = draw.textbbox((0, 0), "Hg", font=font)
    title_h, line_h = int(title_h), int(line_h)

    col_count = len(matrix[0])
    col_widths: list[int] = [min_col_width] * col_count
    for col in range(col_count):
        max_w = measure(draw, matrix[0][col], font)
        for row in matrix[1:]:
            max_w = max(max_w, measure(draw, row[col], font))
        col_widths[col] = max(min_col_width, min(max_col_width, max_w + 2 * padding_x))

    wrapped_cells: list[list[list[str]]] = []
    row_heights: list[int] = []
    for row in matrix:
        wrapped_row: list[list[str]] = []
        max_lines = 1
        for col_idx, cell_text in enumerate(row):
            lines = wrap_text(
                draw, cell_text, font, col_widths[col_idx] - 2 * padding_x
            )
            wrapped_row.append(lines)
            max_lines = max(max_lines, len(lines))
        wrapped_cells.append(wrapped_row)
        row_h = max_lines * line_h + (max_lines - 1) * line_spacing + 2 * padding_y
        row_heights.append(row_h)

    table_width = sum(col_widths) + border * 2
    table_height = sum(row_heights) + border * 2 + row_line_width * (len(matrix) - 1)
    title_block_h = title_h + title_gap + 2 * border

    image = Image.new("RGB", (table_width, table_height + title_block_h), "white")
    draw = ImageDraw.Draw(image)

    draw.text((border, border), safe_text(title), fill="black", font=title_font)
    table_top = title_block_h

    draw.rectangle(
        (0, table_top, table_width - 1, table_top + table_height - 1),
        outline="black",
        width=border,
        fill="white",
    )

    draw.rectangle(
        (
            border,
            table_top + border,
            table_width - border - 1,
            table_top + border + row_heights[0],
        ),
        fill="#f2f2f2",
    )

    y = table_top + border
    for row_idx, (row_cells, row_h) in enumerate(zip(wrapped_cells, row_heights)):
        x = border
        for col_idx, cell_lines in enumerate(row_cells):
            col_w = col_widths[col_idx]
            if col_idx > 0:
                draw.line(
                    (x, y, x, y + row_h),
                    fill="black",
                    width=row_line_width,
                )
            text_y = y + padding_y
            for line in cell_lines:
                draw.text((x + padding_x, text_y), line, fill="black", font=font)
                text_y += line_h + line_spacing
            x += col_w

        if row_idx < len(wrapped_cells) - 1:
            draw.line(
                (border, y + row_h, table_width - border, y + row_h),
                fill="black",
                width=row_line_width,
            )
        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def sanitize_filename(name: str) -> str:
    name = safe_text(name)
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    return name[:200] if name else "table"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the MMQA table corpus into PNG images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_corpus_table.json"),
        help="Input table corpus JSON (dict keyed by table id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Output image directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render at most this many tables (0 means all).",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input file must be a dict keyed by table id.")

    items = list(data.items())
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    font = choose_font()
    total = len(items)
    if total == 0:
        print("No tables to render.")
        return

    done = 0
    skipped = 0
    for table_id, record in items:
        matrix = to_table_matrix(record if isinstance(record, dict) else {})
        if not matrix:
            skipped += 1
            continue

        table = record.get("table", {}) if isinstance(record, dict) else {}
        table_name = safe_text(table.get("table_name", ""))
        page_title = safe_text(record.get("title", ""))
        title = table_name or page_title or str(table_id)

        filename = f"{sanitize_filename(str(table_id))}.png"
        out_path = args.output_dir / filename
        render_table_image(matrix, out_path, title=title, font=font)
        done += 1

        if done % 50 == 0 or done == total:
            print(f"[{done}/{total}] written: {out_path}")

    print(f"Done. rendered={done} skipped={skipped} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
