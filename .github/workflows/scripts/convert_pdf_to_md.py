"""Convert PDF files to Markdown.

Usage:
    python convert_pdf_to_md.py INPUT.pdf [OUTPUT.md]
    python convert_pdf_to_md.py INPUT_DIR [OUTPUT_DIR]

With no arguments, PDFs in ./pdfs are written to ./markdowns when that
folder exists. Otherwise usage is printed.
"""

from __future__ import annotations

import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf

BULLET_RE = re.compile(r"^(?:[•●▪‣◦]|\d+[.)]|[A-Za-z][.)])\s+\S")
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")


@dataclass
class TextLine:
    x: float
    y: float
    x1: float
    y1: float
    size: float
    text: str
    bold: bool


@dataclass
class Block:
    x: float
    y: float
    x1: float
    y1: float
    kind: str
    payload: object


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def line_from_spans(line: dict) -> TextLine | None:
    parts: list[str] = []
    previous_x1 = None
    size = 0.0
    bold = True
    has_span = False
    for span in line["spans"]:
        piece = span["text"]
        if not piece:
            continue
        has_span = True
        if previous_x1 is not None and span["bbox"][0] - previous_x1 > 1.5:
            parts.append(" ")
        parts.append(piece)
        previous_x1 = span["bbox"][2]
        size = max(size, span["size"])
        bold = bold and bool(span.get("flags", 0) & 16)
    text = clean_text("".join(parts))
    if not text or not has_span:
        return None
    x0, y0, x1, y1 = line["bbox"]
    return TextLine(x=x0, y=y0, x1=x1, y1=y1, size=size, text=text, bold=bold)


def page_lines(page) -> list[TextLine]:
    lines: list[TextLine] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            parsed = line_from_spans(raw_line)
            if parsed:
                lines.append(parsed)
    return lines


def repeated_margin_text(doc) -> set[str]:
    """Lines that repeat in the top or bottom margin across many pages."""
    counts: dict[str, int] = {}
    for page in doc:
        height = page.rect.height
        seen: set[str] = set()
        for line in page_lines(page):
            in_margin = line.y < height * 0.08 or line.y1 > height * 0.92
            if not in_margin or PAGE_NUMBER_RE.fullmatch(line.text):
                continue
            seen.add(line.text.casefold())
        for text in seen:
            counts[text] = counts.get(text, 0) + 1
    threshold = max(2, int(doc.page_count * 0.4))
    return {text for text, count in counts.items() if count >= threshold}


def is_margin_chrome(line: TextLine, page, chrome: set[str]) -> bool:
    height = page.rect.height
    in_margin = line.y < height * 0.08 or line.y1 > height * 0.92
    if not in_margin:
        return False
    if PAGE_NUMBER_RE.fullmatch(line.text):
        return True
    return line.text.casefold() in chrome


def overlaps(rect: pymupdf.Rect, other: pymupdf.Rect) -> bool:
    intersection = rect & other
    if intersection.is_empty:
        return False
    area = rect.width * rect.height
    if area <= 0:
        return False
    return (intersection.width * intersection.height) / area > 0.5


def extract_tables(page) -> list[tuple[pymupdf.Rect, list[list[str]]]]:
    try:
        finder = page.find_tables()
    except Exception:
        return []
    page_area = page.rect.width * page.rect.height
    tables = []
    for table in finder.tables:
        bbox = pymupdf.Rect(table.bbox)
        if page_area and (bbox.width * bbox.height) > page_area * 0.85:
            continue
        rows = []
        for raw_row in table.extract() or []:
            row = [clean_text((cell or "").replace("\n", " ")) for cell in raw_row]
            if any(row):
                rows.append(row)
        if len(rows) < 2 or max(len(row) for row in rows) < 2:
            continue
        tables.append((bbox, rows))
    tables.sort(key=lambda item: (item[0].y0, item[0].x0))
    kept = []
    for bbox, rows in tables:
        if any(overlaps(bbox, earlier) or overlaps(earlier, bbox) for earlier, _ in kept):
            continue
        kept.append((bbox, rows))
    return kept


def table_markdown(rows: list[list[str]]) -> str:
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]

    def cell(value: str) -> str:
        return value.replace("|", "\\|").strip() or " "

    header = "| " + " | ".join(cell(value) for value in normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in normalized[0]) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in normalized[1:]]
    return "\n".join([header, separator, *body])


def column_read(blocks: list[Block]) -> list[Block]:
    """Read side-by-side blocks down each column, left column first."""
    if len(blocks) < 2:
        return blocks
    columns: list[dict] = []
    for block in sorted(blocks, key=lambda item: item.x):
        placed = False
        for column in columns:
            overlaps_x = block.x < column["x1"] - 8 and block.x1 > column["x"] + 8
            if overlaps_x:
                column["blocks"].append(block)
                column["x"] = min(column["x"], block.x)
                column["x1"] = max(column["x1"], block.x1)
                placed = True
                break
        if not placed:
            columns.append({"x": block.x, "x1": block.x1, "blocks": [block]})
    columns.sort(key=lambda column: column["x"])
    side_by_side = len(columns) > 1 and all(
        columns[index]["x1"] <= columns[index + 1]["x"] + 15 for index in range(len(columns) - 1)
    )
    if not side_by_side:
        return sorted(blocks, key=lambda item: (item.y, item.x))
    ordered: list[Block] = []
    for column in columns:
        ordered.extend(sorted(column["blocks"], key=lambda item: item.y))
    return ordered


def reading_order(blocks: list[Block]) -> list[Block]:
    """Keep rows that sit alone as separators, and read the rest by column."""
    if len(blocks) < 2:
        return blocks
    separators: list[Block] = []
    flowing: list[Block] = []
    for block in blocks:
        shares_row = any(
            other is not block and block.y < other.y1 - 2 and other.y < block.y1 - 2
            for other in blocks
        )
        if shares_row:
            flowing.append(block)
        else:
            separators.append(block)
    ordered: list[Block] = []
    cursor = -1.0
    for separator in sorted(separators, key=lambda item: item.y):
        band = [block for block in flowing if cursor < block.y < separator.y]
        ordered.extend(column_read(band))
        ordered.append(separator)
        cursor = separator.y1
    ordered.extend(column_read([block for block in flowing if block.y > cursor]))
    return ordered


def heading_level(size: float, body_size: float) -> int | None:
    if body_size <= 0 or size < body_size * 1.25:
        return None
    if size >= body_size * 1.8:
        return 2
    if size >= body_size * 1.45:
        return 3
    return 4


def render_lines(lines: list[TextLine], body_size: float) -> str:
    chunks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            chunks.append(clean_text(" ".join(paragraph)))
            paragraph.clear()

    for index, line in enumerate(lines):
        text = line.text
        level = heading_level(line.size, body_size)
        is_short = len(text) <= 140
        if level and is_short:
            flush_paragraph()
            prefix = "#" * level + " "
            previous_line = lines[index - 1] if index else None
            close_heading = (
                previous_line is not None
                and heading_level(previous_line.size, body_size) == level
                and line.y - previous_line.y1 < max(8, previous_line.size * 0.9)
                and chunks
                and chunks[-1].startswith(prefix)
                and not chunks[-1].startswith("#" * (level + 1))
            )
            if close_heading:
                chunks[-1] = f"{chunks[-1]} {text}"
            else:
                chunks.append(f"{prefix}{text}")
            continue
        if BULLET_RE.match(text):
            flush_paragraph()
            bullet = re.sub(r"^[•●▪‣◦]\s*", "", text)
            bullet = re.sub(r"^\d+[.)]\s*", "", bullet)
            bullet = re.sub(r"^[A-Za-z][.)]\s*", "", bullet)
            chunks.append(f"- {bullet}")
            continue
        previous_line = lines[index - 1] if index else None
        list_wrap = (
            previous_line is not None
            and chunks
            and chunks[-1].startswith("- ")
            and not BULLET_RE.match(text)
            and heading_level(line.size, body_size) is None
            and line.y - previous_line.y1 < max(6, previous_line.size * 0.85)
        )
        if list_wrap:
            if chunks[-1].endswith("-") and text[:1].islower():
                chunks[-1] = chunks[-1][:-1] + text
            else:
                joiner = "" if chunks[-1].endswith("-") else " "
                chunks[-1] = chunks[-1] + joiner + text
            continue
        content = f"**{text}**" if line.bold and len(text) < 200 else text
        previous = lines[index - 1] if index else None
        continues = (
            previous is not None
            and heading_level(previous.size, body_size) is None
            and not BULLET_RE.match(previous.text)
            and abs(line.size - previous.size) < 1.5
            and line.y - previous.y1 < max(4, previous.size * 0.8)
        )
        if continues and paragraph:
            if paragraph[-1].endswith("-") and content[:1].islower():
                paragraph[-1] = paragraph[-1][:-1] + content
            elif paragraph[-1].endswith("-"):
                paragraph[-1] += content
            else:
                paragraph.append(content)
        else:
            flush_paragraph()
            paragraph.append(content)
    flush_paragraph()
    return "\n\n".join(chunk for chunk in chunks if chunk)


def body_font_size(doc, chrome: set[str]) -> float:
    sizes: list[float] = []
    for page in doc:
        for line in page_lines(page):
            if is_margin_chrome(line, page, chrome):
                continue
            sizes.append(line.size)
    if not sizes:
        return 12.0
    return statistics.median(sizes)


def page_markdown(page, chrome: set[str], body_size: float) -> str:
    tables = extract_tables(page)
    table_rects = [bbox for bbox, _ in tables]
    text_blocks: list[list[TextLine]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        rect = pymupdf.Rect(block["bbox"])
        if any(overlaps(rect, table_rect) for table_rect in table_rects):
            continue
        lines = []
        for raw_line in block.get("lines", []):
            parsed = line_from_spans(raw_line)
            if parsed and not is_margin_chrome(parsed, page, chrome):
                lines.append(parsed)
        if lines:
            text_blocks.append(lines)

    blocks: list[Block] = []
    for lines in text_blocks:
        x0 = min(line.x for line in lines)
        y0 = min(line.y for line in lines)
        x1 = max(line.x1 for line in lines)
        y1 = max(line.y1 for line in lines)
        rendered = render_lines(lines, body_size)
        if rendered:
            blocks.append(Block(x=x0, y=y0, x1=x1, y1=y1, kind="text", payload=rendered))
    for bbox, rows in tables:
        blocks.append(Block(x=bbox.x0, y=bbox.y0, x1=bbox.x1, y1=bbox.y1, kind="table", payload=table_markdown(rows)))
    ordered = merge_wrapped_text(reading_order(blocks))
    if not ordered:
        return ""
    return "\n\n".join(str(block.payload) for block in ordered)


def merge_wrapped_text(blocks: list[Block]) -> list[Block]:
    """Join a short line that continues the previous text block."""
    merged: list[Block] = []
    for block in blocks:
        if merged and merged[-1].kind == "text" and block.kind == "text":
            previous = str(merged[-1].payload).rstrip()
            nxt = str(block.payload).strip()
            close = block.y - merged[-1].y1 < 18 and abs(block.x - merged[-1].x) < 50
            continues = (
                close
                and nxt
                and not previous.endswith((".", "!", "?", ":", "|"))
                and not nxt.startswith(("#", "-", "|"))
                and (nxt[:1].islower() or nxt.startswith("&") or len(nxt.split()) <= 2)
            )
            if continues:
                last = merged[-1]
                merged[-1] = Block(
                    x=min(last.x, block.x),
                    y=last.y,
                    x1=max(last.x1, block.x1),
                    y1=block.y1,
                    kind="text",
                    payload=f"{previous} {nxt}",
                )
                continue
        merged.append(block)
    return merged


def document_title(doc, pdf_path: Path) -> str:
    title = clean_text((doc.metadata or {}).get("title") or "")
    if title and title.casefold() not in {"untitled", "microsoft word"}:
        return title
    return pdf_path.stem.replace("_", " ")


def convert(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    if doc.is_encrypted and not doc.authenticate(""):
        raise ValueError(f"Cannot read encrypted PDF: {pdf_path}")
    chrome = repeated_margin_text(doc)
    body_size = body_font_size(doc, chrome)
    sections = [f"# {document_title(doc, pdf_path)}", ""]
    for index, page in enumerate(doc):
        body = page_markdown(page, chrome, body_size).strip()
        if not body:
            body = "*(No extractable text on this page.)*"
        sections.append(f"<!-- page {index + 1} -->")
        sections.append("")
        sections.append(body)
        sections.append("")
    text = "\n".join(sections)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def write_markdown(pdf_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(convert(pdf_path), encoding="utf-8", newline="\n")
    print(f"wrote {output_path}")


def convert_directory(source: Path, output_dir: Path) -> int:
    pdfs = sorted(path for path in source.rglob("*.pdf") if path.is_file())
    if not pdfs:
        print(f"No PDF files in {source}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdfs:
        relative = pdf_path.relative_to(source).with_suffix(".md")
        write_markdown(pdf_path, output_dir / relative)
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        pdf_dir = Path("pdfs")
        if pdf_dir.is_dir():
            return convert_directory(pdf_dir, Path("markdowns"))
        print(__doc__.strip(), file=sys.stderr)
        return 2

    source = Path(argv[0])
    if not source.exists():
        print(f"Not found: {source}", file=sys.stderr)
        return 1
    if source.is_dir():
        output_dir = Path(argv[1]) if len(argv) > 1 else source
        return convert_directory(source, output_dir)

    output = Path(argv[1]) if len(argv) > 1 else source.with_suffix(".md")
    write_markdown(source, output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
