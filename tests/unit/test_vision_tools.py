"""Vision tools: bounded reads, strict coordinate parsing, and a free event loop.

pyautogui, pdfplumber and pandas are heavy and absent from CI, so fakes are injected
into sys.modules; the tools import them lazily inside their worker functions.
"""

import asyncio
import base64
import csv
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from jarvis.agent.tool_registry import ToolRegistry
from jarvis.tools import vision_tools
from jarvis.tools.vision_tools import VisionClient, parse_element_coords, register_vision_tools

SCREEN = (1920, 1080)


class FakeLLM:
    def __init__(self, content: str = "") -> None:
        self.content = content
        self.calls: list[dict] = []

    async def chat(self, messages, images=None):
        self.calls.append({"messages": messages, "images": images})
        return {"message": {"content": self.content}}


def make_registry(content: str = "") -> tuple[ToolRegistry, FakeLLM]:
    llm = FakeLLM(content)
    registry = ToolRegistry()
    register_vision_tools(registry, llm)
    return registry, llm


def install_module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


# -- fakes -------------------------------------------------------------------


def install_fake_pyautogui(monkeypatch, delay: float = 0.0, size=SCREEN) -> SimpleNamespace:
    state = SimpleNamespace(shots=0, resized_to=None, threads=set())

    class Image:
        def resize(self, new_size):
            state.resized_to = new_size
            return self

        def save(self, buf, format, optimize):
            buf.write(b"PNGDATA")

    def screenshot():
        state.shots += 1
        state.threads.add(threading.get_ident())
        time.sleep(delay)
        return Image()

    install_module(monkeypatch, "pyautogui", screenshot=screenshot, size=lambda: size)
    return state


class FakeFrame:
    def __init__(self, columns, rows):
        self.columns = list(columns)
        self.rows = [list(r) for r in rows]

    def __len__(self):
        return len(self.rows)

    def head(self, n):
        return FakeFrame(self.columns, self.rows[:n])

    def to_string(self, index=True):
        return "\n".join(" ".join(str(c) for c in row) for row in self.rows)


class FakeReader:
    def __init__(self, frames):
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.frames)


def install_fake_pandas(monkeypatch, delay: float = 0.0, excel_rows=()) -> SimpleNamespace:
    """A pandas stand-in that refuses unbounded reads."""
    state = SimpleNamespace(csv_calls=[], excel_calls=[], threads=set())

    def read_csv(path, nrows=None, usecols=None, chunksize=None):
        state.threads.add(threading.get_ident())
        time.sleep(delay)
        state.csv_calls.append({"nrows": nrows, "usecols": usecols, "chunksize": chunksize})
        assert nrows is not None or chunksize is not None, "unbounded read_csv"
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, body = rows[0], rows[1:]
        if usecols is not None:
            header = [header[i] for i in usecols]
            body = [[r[i] for i in usecols] for r in body]
        if chunksize:
            return FakeReader(
                [FakeFrame(header, body[i:i + chunksize]) for i in range(0, len(body), chunksize)]
            )
        return FakeFrame(header, body[:nrows])

    def read_excel(path, nrows=None):
        state.threads.add(threading.get_ident())
        time.sleep(delay)
        state.excel_calls.append({"nrows": nrows})
        assert nrows is not None, "unbounded read_excel"
        return FakeFrame(["a", "b"], list(excel_rows)[:nrows])

    install_module(monkeypatch, "pandas", read_csv=read_csv, read_excel=read_excel)
    return state


def install_fake_pdfplumber(
    monkeypatch, page_texts, accept_pages: bool = True, delay: float = 0.0
) -> SimpleNamespace:
    state = SimpleNamespace(opened_with=None, extracted=[], threads=set())

    class Page:
        def __init__(self, number, text):
            self.number, self.text = number, text

        def extract_text(self):
            state.extracted.append(self.number)
            return self.text

    class Pdf:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def build(path, wanted):
        state.threads.add(threading.get_ident())
        time.sleep(delay)
        state.opened_with = wanted
        pages = [
            Page(i, t) for i, t in enumerate(page_texts, 1) if wanted is None or i in wanted
        ]
        return Pdf(pages)

    if accept_pages:
        def open_(path, pages=None):
            return build(path, pages)
    else:
        def open_(path):
            return build(path, None)

    install_module(monkeypatch, "pdfplumber", open=open_)
    return state


# -- coordinate parsing ------------------------------------------------------

W, H = SCREEN


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # The sentence-final period used to raise ValueError and drop everything.
        ("Save: x=0.5, y=0.25.", [("Save", 960, 270)]),
        ("Save: x=0.5, y=0.25", [("Save", 960, 270)]),
        ("Save button (x=0.5, y=0.25).", [("Save button", 960, 270)]),
        ("x=0.5 y=0.25", [("buttons", 960, 270)]),
        ("x: 0.75; y: 0.125", [("buttons", 1440, 135)]),
        ("x = 0.5 , y = 0.25", [("buttons", 960, 270)]),
        ("X=.5, Y=.25", [("buttons", 960, 270)]),
        ("1. **OK** x: 0.5; y: 0.25", [("OK", 960, 270)]),
        ("- Button 2 (x=0.5, y=0.25)", [("Button 2", 960, 270)]),
        # The right and bottom edges are the last pixel, not one past the screen.
        ("Close: x=1.0, y=1.0", [("Close", W - 1, H - 1)]),
        ("Origin: x=0, y=0", [("Origin", 0, 0)]),
        # Pixels or out-of-range values are rejected, not clamped to a corner.
        ("Save: x=640, y=360", []),
        ("Save: x=1.5, y=0.2", []),
        ("Save: x=0.2, y=1.01", []),
        ("Save: x=-0.2, y=0.2", []),
        # Malformed or unrelated text is skipped.
        ("Save: x=1.5.3, y=0.2", []),
        ("Inbox: 0.5, y=0.25", []),
        ("I could not find any buttons.", []),
        ("", []),
    ],
)
def test_parse_element_coords(line, expected):
    results = parse_element_coords(line, "buttons", W, H)
    assert [(r["name"], r["x"], r["y"]) for r in results] == expected


def test_one_bad_line_does_not_drop_the_others():
    content = "\n".join([
        "Found 3 buttons:",
        "A: x=0.5, y=0.25.",
        "B: x=700, y=300",
        "C: x=0.75, y=0.125",
    ])
    results = parse_element_coords(content, "buttons", W, H)
    assert [(r["name"], r["x"], r["y"]) for r in results] == [
        ("A", 960, 270),
        ("C", 1440, 135),
    ]


async def test_find_ui_elements_survives_sentence_final_period(monkeypatch):
    install_fake_pyautogui(monkeypatch)
    registry, llm = make_registry("Save: x=0.5, y=0.25.\nCancel: x=0.75, y=0.125.")
    out = await registry.execute("find_ui_elements", query="buttons")
    assert out == "Found elements:\nSave: (960, 270)\nCancel: (1440, 135)"
    assert llm.calls[0]["images"] == [base64.b64encode(b"PNGDATA").decode()]


async def test_find_ui_elements_reports_nothing_for_pixel_answers(monkeypatch):
    install_fake_pyautogui(monkeypatch)
    registry, _ = make_registry("Save: x=640, y=360")
    out = await registry.execute("find_ui_elements", query="buttons")
    assert out == 'No elements matching "buttons" found on screen.'


# -- screenshots -------------------------------------------------------------


async def test_screenshot_is_downsized_and_encoded(monkeypatch):
    state = install_fake_pyautogui(monkeypatch)
    b64 = await VisionClient(FakeLLM())._screenshot_to_b64()
    assert base64.b64decode(b64) == b"PNGDATA"
    assert state.resized_to == vision_tools.SCREENSHOT_SIZE


# -- images ------------------------------------------------------------------


async def test_read_image_sends_the_file(tmp_path):
    image = tmp_path / "pic.PNG"
    image.write_bytes(b"\x89PNGfake")
    registry, llm = make_registry("a cat")
    out = await registry.execute("read_image", path=str(image), question="what is it?")
    assert out == "a cat"
    assert llm.calls[0]["images"] == [base64.b64encode(b"\x89PNGfake").decode()]
    assert llm.calls[0]["messages"][-1]["content"] == "what is it?"


async def test_read_image_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "MAX_IMAGE_BYTES", 10)
    image = tmp_path / "big.png"
    image.write_bytes(b"x" * 11)
    registry, llm = make_registry()
    out = await registry.execute("read_image", path=str(image))
    assert "too large" in out and "limit" in out
    assert llm.calls == []


async def test_read_image_rejects_non_image_and_missing(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    registry, llm = make_registry()
    assert "Unsupported image type '.txt'" in await registry.execute("read_image", path=str(text))
    assert "not found" in await registry.execute("read_image", path=str(tmp_path / "nope.png"))
    assert "not found" in await registry.execute("read_image", path=str(tmp_path))
    assert llm.calls == []


# -- PDFs --------------------------------------------------------------------


async def test_read_pdf_stops_parsing_at_the_character_cap(tmp_path, monkeypatch):
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(b"%PDF")
    state = install_fake_pdfplumber(monkeypatch, ["x" * 1000] * 30)
    registry, _ = make_registry()
    out = await registry.execute("read_pdf", path=str(pdf))
    assert out.endswith("[...truncated...]")
    assert len(out) < vision_tools.MAX_PDF_CHARS + 100
    assert len(state.extracted) <= 10  # not all 20 allowed pages


async def test_read_pdf_asks_for_only_the_first_pages(tmp_path, monkeypatch):
    pdf = tmp_path / "many.pdf"
    pdf.write_bytes(b"%PDF")
    state = install_fake_pdfplumber(monkeypatch, [f"text {i}" for i in range(1, 41)])
    registry, _ = make_registry()
    out = await registry.execute("read_pdf", path=str(pdf))
    assert max(state.opened_with) == vision_tools.MAX_PDF_PAGES + 1
    assert max(state.extracted) == vision_tools.MAX_PDF_PAGES
    assert "--- Page 20 ---" in out and "--- Page 21 ---" not in out
    assert "only the first 20 pages" in out


async def test_read_pdf_without_pages_argument_still_works(tmp_path, monkeypatch):
    pdf = tmp_path / "old.pdf"
    pdf.write_bytes(b"%PDF")
    install_fake_pdfplumber(monkeypatch, ["hello"], accept_pages=False)
    registry, _ = make_registry()
    assert "hello" in await registry.execute("read_pdf", path=str(pdf))


async def test_read_pdf_rejects_oversized_file_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "MAX_PDF_BYTES", 10)
    pdf = tmp_path / "huge.pdf"
    pdf.write_bytes(b"x" * 11)
    state = install_fake_pdfplumber(monkeypatch, ["text"])
    registry, _ = make_registry()
    out = await registry.execute("read_pdf", path=str(pdf))
    assert "too large" in out
    assert state.opened_with is None and state.extracted == []


async def test_read_pdf_empty_and_missing(tmp_path, monkeypatch):
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF")
    install_fake_pdfplumber(monkeypatch, [None, ""])
    registry, _ = make_registry()
    assert await registry.execute("read_pdf", path=str(pdf)) == "No text found in PDF."
    assert "not found" in await registry.execute("read_pdf", path=str(tmp_path / "nope.pdf"))


# -- spreadsheets ------------------------------------------------------------


def write_csv(path, rows: int, cols=("id", "name")) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(rows):
            writer.writerow([i, f"n{i}"])


async def test_csv_preview_reads_only_the_preview_rows(tmp_path, monkeypatch):
    path = tmp_path / "small.csv"
    write_csv(path, 3)
    state = install_fake_pandas(monkeypatch)
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert out.startswith("Spreadsheet: 3 rows x 2 columns")
    assert "Columns: id, name" in out
    assert state.csv_calls == [{"nrows": 11, "usecols": None, "chunksize": None}]


async def test_csv_row_count_is_streamed_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "CSV_CHUNK_ROWS", 10)
    path = tmp_path / "medium.csv"
    write_csv(path, 25)
    state = install_fake_pandas(monkeypatch)
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert out.startswith("Spreadsheet: 25 rows x 2 columns")
    assert out.count("\n0 n0") == 1  # preview rows come from the head only
    preview, count = state.csv_calls
    assert preview["nrows"] == 11
    assert count == {"nrows": None, "usecols": [0], "chunksize": 10}


async def test_csv_row_count_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "CSV_CHUNK_ROWS", 10)
    monkeypatch.setattr(vision_tools, "MAX_COUNTED_ROWS", 15)
    path = tmp_path / "long.csv"
    write_csv(path, 100)
    install_fake_pandas(monkeypatch)
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert out.startswith("Spreadsheet: more than 15 rows x 2 columns")


async def test_huge_csv_is_previewed_but_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "MAX_CSV_COUNT_BYTES", 10)
    path = tmp_path / "huge.csv"
    write_csv(path, 30)
    state = install_fake_pandas(monkeypatch)
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert "more than 10 rows (file too large to count)" in out
    assert len(state.csv_calls) == 1


async def test_excel_preview_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"PK")
    state = install_fake_pandas(monkeypatch, excel_rows=[(i, i) for i in range(50)])
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert "more than 10 rows" in out and "x 2 columns" in out
    assert state.excel_calls == [{"nrows": 11}]


async def test_excel_over_the_size_limit_is_rejected_unread(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "MAX_EXCEL_BYTES", 5)
    path = tmp_path / "big.xlsx"
    path.write_bytes(b"PK" * 10)
    state = install_fake_pandas(monkeypatch, excel_rows=[(1, 1)])
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert "too large" in out
    assert state.excel_calls == []


async def test_spreadsheet_unsupported_and_missing(tmp_path):
    other = tmp_path / "data.json"
    other.write_text("{}")
    registry, _ = make_registry()
    assert "Unsupported spreadsheet format: .json" in await registry.execute(
        "read_spreadsheet", path=str(other))
    assert "not found" in await registry.execute(
        "read_spreadsheet", path=str(tmp_path / "nope.csv"))


async def test_spreadsheet_summary_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "MAX_SUMMARY_CHARS", 50)
    path = tmp_path / "wide.csv"
    write_csv(path, 3)
    install_fake_pandas(monkeypatch)
    registry, _ = make_registry()
    out = await registry.execute("read_spreadsheet", path=str(path))
    assert out.endswith("[...truncated...]")


# -- the event loop stays free ----------------------------------------------


async def ticks_while(awaitable, interval: float = 0.01):
    """Await ``awaitable`` while a ticker task counts event-loop turns."""
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(interval)
            ticks += 1

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        result = await awaitable
    finally:
        stop.set()
        await task
    return result, ticks


SLOW = 0.4
MIN_TICKS = 5  # ~40 expected when the loop is free, ~0 when it is blocked


async def test_screenshot_does_not_block_the_loop(monkeypatch):
    state = install_fake_pyautogui(monkeypatch, delay=SLOW)
    _, ticks = await ticks_while(VisionClient(FakeLLM())._screenshot_to_b64())
    assert ticks >= MIN_TICKS
    assert state.threads and threading.get_ident() not in state.threads


async def test_find_elements_does_not_block_the_loop(monkeypatch):
    install_fake_pyautogui(monkeypatch, delay=SLOW)
    client = VisionClient(FakeLLM("Save: x=0.5, y=0.25"))
    results, ticks = await ticks_while(client.find_elements("buttons"))
    assert ticks >= MIN_TICKS
    assert [r["name"] for r in results] == ["Save"]


async def test_read_pdf_does_not_block_the_loop(tmp_path, monkeypatch):
    pdf = tmp_path / "slow.pdf"
    pdf.write_bytes(b"%PDF")
    state = install_fake_pdfplumber(monkeypatch, ["hello"], delay=SLOW)
    out, ticks = await ticks_while(VisionClient(FakeLLM()).read_pdf(str(pdf)))
    assert ticks >= MIN_TICKS
    assert "hello" in out
    assert threading.get_ident() not in state.threads


async def test_read_spreadsheet_does_not_block_the_loop(tmp_path, monkeypatch):
    path = tmp_path / "slow.csv"
    write_csv(path, 2)
    state = install_fake_pandas(monkeypatch, delay=SLOW)
    out, ticks = await ticks_while(VisionClient(FakeLLM()).read_spreadsheet(str(path)))
    assert ticks >= MIN_TICKS
    assert out.startswith("Spreadsheet: 2 rows")
    assert threading.get_ident() not in state.threads


async def test_read_image_does_not_block_the_loop(tmp_path, monkeypatch):
    seen = set()

    def slow_read(image_path):
        seen.add(threading.get_ident())
        time.sleep(SLOW)
        return "QUJD", ""

    monkeypatch.setattr(vision_tools, "_read_image_b64", slow_read)
    client = VisionClient(FakeLLM("fine"))
    out, ticks = await ticks_while(client.analyze_image(str(tmp_path / "any.png")))
    assert ticks >= MIN_TICKS
    assert out == "fine"
    assert threading.get_ident() not in seen
