"""
JARVIS Browser Automation Tools
=================================
Playwright-based tools for human-like browser control.
Persistent browser session across tool calls within a conversation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)


class BrowserSession:
    """
    Manages a persistent Playwright browser session.

    The browser stays open across multiple tool calls so the agent
    can perform multi-step interactions (navigate → click → type → read).
    """

    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    @property
    def is_running(self) -> bool:
        return self._page is not None and not self._page.is_closed()

    async def _ensure_browser(self) -> None:
        """Launch browser on first use (lazy init)."""
        if self.is_running:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        logger.info("🌐 Browser launched")

    @property
    def page(self) -> Any:
        return self._page

    async def shutdown(self) -> None:
        """Close browser and clean up."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("🌐 Browser closed")


# Module-level singleton so all tools share the same session
_session = BrowserSession()


async def shutdown_browser() -> None:
    """Called by Orchestrator on shutdown."""
    await _session.shutdown()


def register_browser_tools(registry: ToolRegistry) -> None:
    """Register all Playwright-based browser automation tools."""

    # ── Navigate ──

    @registry.tool(
        name="browser_navigate",
        description=(
            "Navigate the browser to a URL. Opens the browser if not already open. "
            "Use this for going to websites, Google searches, YouTube, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to navigate to (e.g., 'https://google.com').",
                },
            },
            "required": ["url"],
        },
    )
    async def browser_navigate(url: str) -> str:
        try:
            await _session._ensure_browser()
            response = await _session.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title = await _session.page.title()
            status = response.status if response else "unknown"
            return f"✅ Navigated to: {title}\nURL: {_session.page.url}\nStatus: {status}"
        except Exception as e:
            return f"❌ Navigation failed: {e}"

    # ── Google Search ──

    @registry.tool(
        name="browser_search",
        description=(
            "Search Google for a query and return the top results. "
            "Use this when the user wants to search for something online."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    )
    async def browser_search(query: str) -> str:
        try:
            await _session._ensure_browser()
            page = _session.page

            await page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)

            # Type the search query
            search_box = page.locator('textarea[name="q"], input[name="q"]').first
            await search_box.click()
            await search_box.fill(query)
            await search_box.press("Enter")
            await page.wait_for_load_state("domcontentloaded")

            # Wait for results
            await page.wait_for_selector("div#search", timeout=10000)

            # Extract top results
            results = []
            items = page.locator("div#search div.g")
            count = min(await items.count(), 8)
            for i in range(count):
                item = items.nth(i)
                try:
                    title_el = item.locator("h3").first
                    title = await title_el.inner_text() if await title_el.count() > 0 else ""
                    link_el = item.locator("a").first
                    href = await link_el.get_attribute("href") if await link_el.count() > 0 else ""
                    snippet_el = item.locator("div[data-sncf], div.VwiC3b, span.aCOpRe").first
                    snippet = await snippet_el.inner_text() if await snippet_el.count() > 0 else ""
                    if title:
                        results.append(f"{i+1}. {title}\n   {href}\n   {snippet}")
                except Exception:
                    continue

            if results:
                return f"🔍 Google results for '{query}':\n\n" + "\n\n".join(results)
            return f"🔍 Searched for '{query}' but couldn't parse results. Page: {page.url}"
        except Exception as e:
            return f"❌ Search failed: {e}"

    # ── Click Element ──

    @registry.tool(
        name="browser_click",
        description=(
            "Click an element on the current page by its text content, CSS selector, or link text. "
            "Examples: 'Submit', 'a:has-text(\"Login\")', '#my-button', 'Sign in'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": (
                        "Text of the element to click, or a CSS/Playwright selector. "
                        "For text: just use the visible text like 'Sign in'. "
                        "For CSS: use standard selectors like '#id', '.class', 'button'."
                    ),
                },
            },
            "required": ["selector"],
        },
    )
    async def browser_click(selector: str) -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            page = _session.page
            # Try as text first, then as CSS selector
            el = page.get_by_text(selector, exact=False).first
            if await el.count() > 0 and await el.is_visible():
                await el.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                title = await page.title()
                return f"✅ Clicked '{selector}'\nPage: {title}\nURL: {page.url}"

            # Try as role-based
            el = page.get_by_role("link", name=selector).first
            if await el.count() > 0 and await el.is_visible():
                await el.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                title = await page.title()
                return f"✅ Clicked link '{selector}'\nPage: {title}\nURL: {page.url}"

            el = page.get_by_role("button", name=selector).first
            if await el.count() > 0 and await el.is_visible():
                await el.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                title = await page.title()
                return f"✅ Clicked button '{selector}'\nPage: {title}\nURL: {page.url}"

            # Fall through to CSS selector
            await page.click(selector, timeout=5000)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            title = await page.title()
            return f"✅ Clicked '{selector}'\nPage: {title}\nURL: {page.url}"
        except Exception as e:
            return f"❌ Click failed for '{selector}': {e}"

    # ── Type / Fill Text ──

    @registry.tool(
        name="browser_type",
        description=(
            "Type text into an input field or text area on the current page. "
            "Provide a selector to identify the field and the text to type. "
            "Use press_enter=true to submit after typing."
        ),
        parameters={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": (
                        "Selector for the input field. Can be a CSS selector like "
                        "'input[name=\"email\"]', '#search', or a placeholder text like 'Search'."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": "The text to type into the field.",
                },
                "press_enter": {
                    "type": "boolean",
                    "description": "Press Enter after typing (default: false).",
                },
            },
            "required": ["selector", "text"],
        },
    )
    async def browser_type(selector: str, text: str, press_enter: bool = False) -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            page = _session.page

            # Try as placeholder text first
            el = page.get_by_placeholder(selector, exact=False).first
            if await el.count() > 0 and await el.is_visible():
                await el.click()
                await el.fill(text)
                if press_enter:
                    await el.press("Enter")
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                return f"✅ Typed '{text}' into field '{selector}'" + (" and pressed Enter" if press_enter else "")

            # Try as label
            el = page.get_by_label(selector, exact=False).first
            if await el.count() > 0 and await el.is_visible():
                await el.click()
                await el.fill(text)
                if press_enter:
                    await el.press("Enter")
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                return f"✅ Typed '{text}' into field '{selector}'" + (" and pressed Enter" if press_enter else "")

            # Fall through to CSS selector
            await page.fill(selector, text, timeout=5000)
            if press_enter:
                await page.press(selector, "Enter")
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            return f"✅ Typed '{text}' into '{selector}'" + (" and pressed Enter" if press_enter else "")
        except Exception as e:
            return f"❌ Type failed for '{selector}': {e}"

    # ── Read Page Content ──

    @registry.tool(
        name="browser_read_page",
        description=(
            "Read the visible text content of the current web page. "
            "Returns the page title, URL, and main text content (truncated if very long). "
            "Use this to understand what's on the page."
        ),
        parameters={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to read text from a specific section (e.g., 'article', 'main', '#content'). If empty, reads the full page body.",
                },
            },
        },
    )
    async def browser_read_page(selector: str = "") -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            page = _session.page
            title = await page.title()
            url = page.url

            if selector:
                el = page.locator(selector).first
                text = await el.inner_text(timeout=5000)
            else:
                text = await page.locator("body").inner_text(timeout=5000)

            # Truncate very long pages
            if len(text) > 5000:
                text = text[:5000] + f"\n\n... (truncated, {len(text)} total chars)"

            return f"📄 Page: {title}\nURL: {url}\n\n{text}"
        except Exception as e:
            return f"❌ Failed to read page: {e}"

    # ── Scroll ──

    @registry.tool(
        name="browser_scroll",
        description="Scroll the page up or down. Use this to see more content.",
        parameters={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Scroll direction: 'down' or 'up'.",
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (default: 500).",
                },
            },
            "required": ["direction"],
        },
    )
    async def browser_scroll(direction: str, amount: int = 500) -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            page = _session.page
            delta = amount if direction.lower() == "down" else -amount
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(0.3)  # let page settle
            return f"✅ Scrolled {direction} by {amount}px"
        except Exception as e:
            return f"❌ Scroll failed: {e}"

    # ── Get Page Info ──

    @registry.tool(
        name="browser_get_info",
        description="Get current page info: title, URL, and a list of visible links and input fields.",
        parameters={"type": "object", "properties": {}},
    )
    async def browser_get_info() -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            page = _session.page
            title = await page.title()
            url = page.url

            # Get visible links (top 20)
            links_info = await page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                return links
                    .filter(a => a.offsetParent !== null && a.innerText.trim())
                    .slice(0, 20)
                    .map(a => ({ text: a.innerText.trim().substring(0, 60), href: a.href }));
            }""")

            # Get visible input fields
            inputs_info = await page.evaluate("""() => {
                const inputs = Array.from(document.querySelectorAll('input, textarea, select'));
                return inputs
                    .filter(el => el.offsetParent !== null)
                    .slice(0, 15)
                    .map(el => ({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        name: el.name || '',
                        placeholder: el.placeholder || '',
                        id: el.id || '',
                    }));
            }""")

            result = f"📄 Page: {title}\nURL: {url}\n"

            if links_info:
                result += f"\n🔗 Links ({len(links_info)}):\n"
                for i, link in enumerate(links_info, 1):
                    result += f"  {i}. {link['text']} → {link['href']}\n"

            if inputs_info:
                result += f"\n📝 Input fields ({len(inputs_info)}):\n"
                for inp in inputs_info:
                    ident = inp['placeholder'] or inp['name'] or inp['id'] or inp['type']
                    result += f"  - <{inp['tag']}> type={inp['type']} ({ident})\n"

            return result
        except Exception as e:
            return f"❌ Failed to get page info: {e}"

    # ── Browser Screenshot ──

    @registry.tool(
        name="browser_screenshot",
        description="Take a screenshot of the current browser page.",
        parameters={
            "type": "object",
            "properties": {
                "save_path": {
                    "type": "string",
                    "description": "Path to save the screenshot (default: ~/.jarvis/browser_screenshot.png).",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture the full scrollable page (default: false, viewport only).",
                },
            },
        },
    )
    async def browser_screenshot(save_path: str = "", full_page: bool = False) -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            if not save_path:
                save_dir = Path.home() / ".jarvis"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / "browser_screenshot.png")
            await _session.page.screenshot(path=save_path, full_page=full_page)
            return f"✅ Browser screenshot saved to {save_path}"
        except Exception as e:
            return f"❌ Screenshot failed: {e}"

    # ── Back / Forward ──

    @registry.tool(
        name="browser_back",
        description="Go back to the previous page in browser history.",
        parameters={"type": "object", "properties": {}},
    )
    async def browser_back() -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            await _session.page.go_back(wait_until="domcontentloaded", timeout=10000)
            title = await _session.page.title()
            return f"✅ Went back → {title}\nURL: {_session.page.url}"
        except Exception as e:
            return f"❌ Back navigation failed: {e}"

    @registry.tool(
        name="browser_forward",
        description="Go forward to the next page in browser history.",
        parameters={"type": "object", "properties": {}},
    )
    async def browser_forward() -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            await _session.page.go_forward(wait_until="domcontentloaded", timeout=10000)
            title = await _session.page.title()
            return f"✅ Went forward → {title}\nURL: {_session.page.url}"
        except Exception as e:
            return f"❌ Forward navigation failed: {e}"

    # ── Tab Management ──

    @registry.tool(
        name="browser_new_tab",
        description="Open a new browser tab and optionally navigate to a URL.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Optional URL to open in the new tab.",
                },
            },
        },
    )
    async def browser_new_tab(url: str = "") -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            new_page = await _session._context.new_page()
            _session._page = new_page
            if url:
                await new_page.goto(url, wait_until="domcontentloaded", timeout=30000)
                title = await new_page.title()
                return f"✅ New tab opened: {title}\nURL: {new_page.url}"
            return "✅ New blank tab opened"
        except Exception as e:
            return f"❌ Failed to open new tab: {e}"

    @registry.tool(
        name="browser_list_tabs",
        description="List all open browser tabs with their titles and URLs.",
        parameters={"type": "object", "properties": {}},
    )
    async def browser_list_tabs() -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            pages = _session._context.pages
            lines = []
            for i, p in enumerate(pages):
                active = " (active)" if p == _session.page else ""
                title = await p.title()
                lines.append(f"  {i+1}. {title} — {p.url}{active}")
            return f"🗂️ Open tabs ({len(pages)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"❌ Failed to list tabs: {e}"

    @registry.tool(
        name="browser_switch_tab",
        description="Switch to a different browser tab by its number (from browser_list_tabs).",
        parameters={
            "type": "object",
            "properties": {
                "tab_number": {
                    "type": "integer",
                    "description": "Tab number (1-based) to switch to.",
                },
            },
            "required": ["tab_number"],
        },
    )
    async def browser_switch_tab(tab_number: int) -> str:
        if not _session.is_running:
            return "❌ Browser not open. Use browser_navigate first."
        try:
            pages = _session._context.pages
            idx = tab_number - 1
            if idx < 0 or idx >= len(pages):
                return f"❌ Invalid tab number. Have {len(pages)} tabs."
            _session._page = pages[idx]
            await _session._page.bring_to_front()
            title = await _session._page.title()
            return f"✅ Switched to tab {tab_number}: {title}\nURL: {_session._page.url}"
        except Exception as e:
            return f"❌ Failed to switch tab: {e}"

    # ── Close Browser ──

    @registry.tool(
        name="browser_close",
        description="Close the browser completely.",
        parameters={"type": "object", "properties": {}},
    )
    async def browser_close() -> str:
        try:
            await _session.shutdown()
            return "✅ Browser closed"
        except Exception as e:
            return f"❌ Failed to close browser: {e}"

    # ── YouTube Search (convenience) ──

    @registry.tool(
        name="browser_youtube_search",
        description=(
            "Search YouTube for a video query. Opens YouTube and types the search. "
            "Use this when the user asks to search YouTube, play a song, find a video, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for on YouTube.",
                },
            },
            "required": ["query"],
        },
    )
    async def browser_youtube_search(query: str) -> str:
        try:
            await _session._ensure_browser()
            page = _session.page
            await page.goto("https://www.youtube.com", wait_until="domcontentloaded", timeout=20000)

            search_box = page.locator('input#search, input[name="search_query"]').first
            await search_box.click()
            await search_box.fill(query)
            await search_box.press("Enter")
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(2000)  # let results load

            # Extract top results
            results = []
            items = page.locator("ytd-video-renderer, ytd-compact-video-renderer")
            count = min(await items.count(), 5)
            for i in range(count):
                try:
                    title_el = items.nth(i).locator("#video-title").first
                    title = (await title_el.inner_text()).strip() if await title_el.count() > 0 else ""
                    if title:
                        results.append(f"{i+1}. {title}")
                except Exception:
                    continue

            summary = "\n".join(results) if results else "Results loaded."
            return f"YouTube results for '{query}':\n{summary}\nURL: {page.url}"
        except Exception as e:
            return f"YouTube search failed: {e}"

    logger.info("Registered %d browser tools", 15)
