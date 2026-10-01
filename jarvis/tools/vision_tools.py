"""
Vision-based desktop automation tools.
Uses the multimodal LLM (qwen3.5:4b) for visual grounding and document understanding.

Everything that blocks (screen capture, image encoding, PDF and spreadsheet parsing)
runs in a worker thread: audio capture and barge-in share the event loop, so it must
never stall. Every read is bounded by a size limit and reads only what it returns.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
from itertools import islice
from pathlib import Path
from typing import Any

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

MB = 1024 * 1024

# Screenshots are downsized before they reach the vision encoder.
SCREENSHOT_SIZE = (672, 448)

# Images: refuse anything that is not an image or is too big to base64 sensibly.
MAX_IMAGE_BYTES = 20 * MB
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})

# PDFs: only the first pages are parsed, and the returned text is capped.
MAX_PDF_BYTES = 50 * MB
MAX_PDF_PAGES = 20
MAX_PDF_CHARS = 8000

# Spreadsheets: a preview needs a handful of rows, never the whole file.
PREVIEW_ROWS = 10
MAX_EXCEL_BYTES = 25 * MB
MAX_CSV_COUNT_BYTES = 256 * MB
MAX_COUNTED_ROWS = 500_000
CSV_CHUNK_ROWS = 50_000
MAX_SUMMARY_CHARS = 8000

_NUMBER = r"(\d+(?:\.\d+)?|\.\d+)"
_COORD_RE = re.compile(
    rf"\bx\s*[=:]\s*{_NUMBER}\s*[,;\s]+\s*y\s*[=:]\s*{_NUMBER}",
    re.IGNORECASE,
)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")


def _mb(size: int) -> str:
    return f"{size / MB:.1f} MB"


def _size_error(path: Path, limit: int, what: str) -> str | None:
    """A message the model can relay when ``path`` is over ``limit`` bytes."""
    size = path.stat().st_size
    if size <= limit:
        return None
    return f"{what} is too large to read: {_mb(size)} (limit {_mb(limit)}): {path}"


def parse_element_coords(
    content: str, query: str, screen_w: int, screen_h: int
) -> list[dict[str, Any]]:
    """Turn the model's 'name: x=0.XX y=0.YY' lines into pixel coordinates.

    A line that does not parse, or whose coordinates are not normalised (0..1), is
    skipped: clicking a guessed spot is worse than reporting nothing.
    """
    results: list[dict[str, Any]] = []
    skipped = 0
    max_x, max_y = max(screen_w - 1, 0), max(screen_h - 1, 0)
    for line in content.splitlines():
        match = _COORD_RE.search(line)
        if not match:
            continue
        try:
            nx, ny = float(match.group(1)), float(match.group(2))
        except ValueError:
            skipped += 1
            continue
        if nx > 1.0 or ny > 1.0:
            skipped += 1
            logger.debug("find elements skipped non-normalised line=%r", line)
            continue
        name = _LIST_MARKER_RE.sub("", line[:match.start()]).strip(" \t:-,;(*`")
        results.append({
            "name": name or query,
            "x": min(int(nx * screen_w), max_x),
            "y": min(int(ny * screen_h), max_y),
        })
    if skipped:
        logger.warning("find elements skipped lines=%d query=%r", skipped, query)
    return results


def capture_screen_b64() -> str:
    """Capture the screen, downsize for the vision model, return base64 PNG (blocking)."""
    import pyautogui  # heavy; loaded only when the screen is captured

    img = pyautogui.screenshot()
    # Downsize to reduce vision encoder latency
    img = img.resize(SCREENSHOT_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _screen_size() -> tuple[int, int]:
    import pyautogui

    width, height = pyautogui.size()
    return int(width), int(height)


def _read_image_b64(image_path: str) -> tuple[str | None, str]:
    """Return (base64, "") for a readable image, else (None, message) (blocking)."""
    path = Path(image_path).expanduser().resolve()
    try:
        if not path.is_file():
            return None, f"Image not found: {path}"
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            supported = ", ".join(sorted(IMAGE_SUFFIXES))
            return None, f"Unsupported image type '{path.suffix}' (supported: {supported}): {path}"
        error = _size_error(path, MAX_IMAGE_BYTES, "Image")
        if error:
            return None, error
        return base64.b64encode(path.read_bytes()).decode(), ""
    except OSError as e:
        logger.error("image read failed error=%s", e)
        return None, f"Error reading image: {e}"


def _open_pdf(pdfplumber: Any, path: Path) -> Any:
    # Ask for one page past the limit so we can tell the reader pages were left out.
    try:
        return pdfplumber.open(str(path), pages=list(range(1, MAX_PDF_PAGES + 2)))
    except TypeError:  # older pdfplumber without the pages argument
        return pdfplumber.open(str(path))


def _read_pdf_text(file_path: str) -> str:
    """Extract the text of the first pages of a PDF (blocking)."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        return f"PDF not found: {path}"

    try:
        error = _size_error(path, MAX_PDF_BYTES, "PDF")
        if error:
            return error

        import pdfplumber  # heavy; loaded only when a PDF is read

        parts: list[str] = []
        total = 0
        pages_left_out = False
        with _open_pdf(pdfplumber, path) as pdf:
            for i, page in enumerate(islice(pdf.pages, MAX_PDF_PAGES + 1)):
                if i >= MAX_PDF_PAGES:
                    pages_left_out = True
                    break
                text = page.extract_text()
                if text:
                    part = f"--- Page {i + 1} ---\n{text}"
                    parts.append(part)
                    total += len(part)
                    if total > MAX_PDF_CHARS:
                        break  # no point parsing pages we will not return
        full_text = "\n\n".join(parts)
        if len(full_text) > MAX_PDF_CHARS:
            full_text = full_text[:MAX_PDF_CHARS] + "\n\n[...truncated...]"
        elif pages_left_out:
            full_text += f"\n\n[...only the first {MAX_PDF_PAGES} pages were read...]"
        return full_text if full_text else "No text found in PDF."
    except Exception as e:
        logger.error("pdf read failed error=%s", e)
        return f"Error reading PDF: {e}"


def _count_csv_rows(pd: Any, path: Path) -> str:
    """Describe the row count of a CSV known to have more than PREVIEW_ROWS rows.

    Streams one column in chunks and stops at a cap, so memory and time stay bounded.
    """
    if path.stat().st_size > MAX_CSV_COUNT_BYTES:
        return f"more than {PREVIEW_ROWS} rows (file too large to count)"
    total = 0
    try:
        with pd.read_csv(str(path), usecols=[0], chunksize=CSV_CHUNK_ROWS) as reader:
            for chunk in reader:
                total += len(chunk)
                if total > MAX_COUNTED_ROWS:
                    return f"more than {MAX_COUNTED_ROWS:,} rows"
    except Exception as e:
        logger.debug("csv row count failed error=%s", e)
        return f"more than {PREVIEW_ROWS} rows (could not count)"
    return f"{total:,} rows"


def _summarize_spreadsheet(file_path: str) -> str:
    """Columns, a row count and the first rows of a CSV/Excel file (blocking)."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        return f"File not found: {path}"

    suffix = path.suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        return f"Unsupported spreadsheet format: {suffix}"

    try:
        if suffix != ".csv":
            error = _size_error(path, MAX_EXCEL_BYTES, "Spreadsheet")
            if error:
                return error

        import pandas as pd  # heavy; loaded only when a spreadsheet is read

        # One row more than we show tells us whether the file has more.
        if suffix == ".csv":
            df = pd.read_csv(str(path), nrows=PREVIEW_ROWS + 1)
        else:
            df = pd.read_excel(str(path), nrows=PREVIEW_ROWS + 1)

        if len(df) <= PREVIEW_ROWS:
            rows = f"{len(df)} rows"
        elif suffix == ".csv":
            rows = _count_csv_rows(pd, path)
        else:
            rows = f"more than {PREVIEW_ROWS} rows (total not counted for Excel files)"

        summary = f"Spreadsheet: {rows} x {len(df.columns)} columns\n\n"
        summary += f"Columns: {', '.join(str(c) for c in df.columns)}\n\n"
        summary += f"First {PREVIEW_ROWS} rows:\n"
        summary += df.head(PREVIEW_ROWS).to_string(index=False)
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS] + "\n\n[...truncated...]"
        return summary
    except Exception as e:
        logger.error("spreadsheet read failed error=%s", e)
        return f"Error reading spreadsheet: {e}"


class VisionClient:
    """Helper for vision-based tasks using the multimodal LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def _screenshot_to_b64(self) -> str:
        """Capture screen, downsize for vision model, and return base64 PNG."""
        return await asyncio.to_thread(capture_screen_b64)

    async def _ask_vision(
        self,
        prompt: str,
        image_b64: str | None = None,
        system_hint: str = "",
    ) -> str:
        """Send a vision query to the LLM."""
        if image_b64 is None:
            image_b64 = await self._screenshot_to_b64()

        messages: list[dict[str, str]] = []
        if system_hint:
            messages.append({"role": "system", "content": system_hint})
        messages.append({"role": "user", "content": prompt})

        resp = await self.llm.chat(messages=messages, images=[image_b64])
        return resp.get("message", {}).get("content", "")

    async def find_elements(self, query: str) -> list[dict[str, Any]]:
        """
        Find multiple UI elements matching a query.
        Returns list of {name, x, y} in pixel coordinates.
        """
        screen_w, screen_h = await asyncio.to_thread(_screen_size)

        prompt = (
            f'Find all visible "{query}" in the screenshot. '
            "List each with normalized center coordinates. "
            'Format: name: x=0.XX y=0.YY'
        )
        system_hint = (
            "You are a precise UI automation assistant. "
            "List only elements you can clearly identify."
        )

        content = await self._ask_vision(prompt, system_hint=system_hint)
        logger.debug("find elements response=%s", content)

        results = parse_element_coords(content, query, screen_w, screen_h)
        logger.info("found elements count=%d query=%r", len(results), query)
        return results

    async def analyze_image(self, image_path: str, question: str = "") -> str:
        """Analyze an image file and answer a question about it."""
        img_b64, error = await asyncio.to_thread(_read_image_b64, image_path)
        if img_b64 is None:
            return error

        prompt = question if question else "Describe this image in detail."
        return await self._ask_vision(prompt, image_b64=img_b64)

    async def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        return await asyncio.to_thread(_read_pdf_text, file_path)

    async def read_spreadsheet(self, file_path: str) -> str:
        """Read an Excel or CSV file and return a summary."""
        return await asyncio.to_thread(_summarize_spreadsheet, file_path)


def register_vision_tools(registry: ToolRegistry, llm: LLMClient) -> None:
    """Register vision-based tools that require the LLM client."""
    vision = VisionClient(llm)

    @registry.tool(
        name="find_ui_elements",
        description="Find all visible UI elements matching a description and return their locations.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of elements to find (e.g., 'buttons', 'icons', 'menu items').",
                },
            },
            "required": ["query"],
        },
    )
    async def find_ui_elements(query: str) -> str:
        results = await vision.find_elements(query)
        if not results:
            return f'No elements matching "{query}" found on screen.'
        lines = [f"{r['name']}: ({r['x']}, {r['y']})" for r in results]
        return "Found elements:\n" + "\n".join(lines)

    @registry.tool(
        name="read_pdf",
        description="Extract text from a PDF file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file.",
                },
            },
            "required": ["path"],
        },
    )
    async def read_pdf(path: str) -> str:
        return await vision.read_pdf(path)

    @registry.tool(
        name="read_spreadsheet",
        description="Read an Excel (.xlsx) or CSV file and return a summary with columns and sample rows.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the spreadsheet file.",
                },
            },
            "required": ["path"],
        },
    )
    async def read_spreadsheet(path: str) -> str:
        return await vision.read_spreadsheet(path)

    @registry.tool(
        name="read_image",
        description="Analyze an image file and describe its contents, or answer a question about it.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image file.",
                },
                "question": {
                    "type": "string",
                    "description": "Optional question to answer about the image.",
                },
            },
            "required": ["path"],
        },
    )
    async def read_image(path: str, question: str = "") -> str:
        return await vision.analyze_image(path, question)
