"""
Vision-based desktop automation tools.
Uses the multimodal LLM (qwen3.5:4b) for visual grounding and document understanding.
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)


class VisionClient:
    """Helper for vision-based tasks using the multimodal LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def _screenshot_to_b64(self) -> str:
        """Capture screen, downsize for vision model, and return base64 PNG."""
        import pyautogui
        img = pyautogui.screenshot()
        # Downsize to reduce vision encoder latency
        img = img.resize((672, 448))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

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
        import pyautogui
        screen_w, screen_h = pyautogui.size()

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

        results = []
        for line in content.splitlines():
            match = re.search(r"x[=:]\s*([0-9.]+)[,;\s]+y[=:]\s*([0-9.]+)", line, re.IGNORECASE)
            if match:
                nx, ny = float(match.group(1)), float(match.group(2))
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                # Extract name from before the coordinate
                name_part = line[:match.start()].strip(" -*•1234567890.")
                name_part = name_part.strip(": ")
                results.append({
                    "name": name_part or query,
                    "x": int(nx * screen_w),
                    "y": int(ny * screen_h),
                })

        logger.info("found elements count=%d query=%r", len(results), query)
        return results

    async def analyze_image(self, image_path: str, question: str = "") -> str:
        """Analyze an image file and answer a question about it."""
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            return f"Image not found: {path}"

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = question if question else "Describe this image in detail."
        return await self._ask_vision(prompt, image_b64=img_b64)

    async def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"PDF not found: {path}"

        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages[:20]):  # Limit to first 20 pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {i+1} ---\n{text}")
            full_text = "\n\n".join(text_parts)
            if len(full_text) > 8000:
                full_text = full_text[:8000] + "\n\n[...truncated...]"
            return full_text if full_text else "No text found in PDF."
        except Exception as e:
            logger.error("pdf read failed error=%s", e)
            return f"Error reading PDF: {e}"

    async def read_spreadsheet(self, file_path: str) -> str:
        """Read an Excel or CSV file and return a summary."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"File not found: {path}"

        try:
            import pandas as pd  # heavy; loaded only when a spreadsheet is read

            suffix = path.suffix.lower()

            if suffix == ".csv":
                df = pd.read_csv(str(path))
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(str(path))
            else:
                return f"Unsupported spreadsheet format: {suffix}"

            rows, cols = df.shape
            summary = f"Spreadsheet: {rows} rows x {cols} columns\n\n"
            summary += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
            summary += "First 10 rows:\n"
            summary += df.head(10).to_string(index=False)
            return summary
        except Exception as e:
            logger.error("spreadsheet read failed error=%s", e)
            return f"Error reading spreadsheet: {e}"


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
