"""Browser tools: failure-safe startup and shutdown, honest results, no blind clicks.

Playwright is heavy and absent from CI, so a fake ``playwright.async_api`` is injected
into sys.modules together with fake driver/browser/context/page objects.
"""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from jarvis.agent.tool_registry import ToolRegistry
from jarvis.tools import browser_tools

# The real Playwright timeout is named TimeoutError and is not the builtin.
PlaywrightTimeout = type("TimeoutError", (Exception,), {})


# -- fakes -------------------------------------------------------------------


class World:
    """Everything the fake driver created, for assertions."""

    def __init__(self) -> None:
        self.drivers: list[FakeDriver] = []
        self.browsers: list[FakeBrowser] = []
        self.launches = 0
        self.launch_error: Exception | None = None
        self.launch_gate: asyncio.Event | None = None


class FakeDriver:
    def __init__(self, world: World) -> None:
        self.world = world
        self.stopped = False
        self.chromium = SimpleNamespace(launch=self._launch)

    async def _launch(self, **kwargs):
        self.world.launches += 1
        if self.world.launch_gate is not None:
            await self.world.launch_gate.wait()
        if self.world.launch_error is not None:
            raise self.world.launch_error
        browser = FakeBrowser(self.world)
        self.world.browsers.append(browser)
        return browser

    async def stop(self):
        self.stopped = True


class FakeDriverManager:
    def __init__(self, world: World) -> None:
        self.world = world

    async def start(self) -> FakeDriver:
        await asyncio.sleep(0)  # let concurrent starters interleave
        driver = FakeDriver(self.world)
        self.world.drivers.append(driver)
        return driver


class FakeBrowser:
    def __init__(self, world: World) -> None:
        self.world = world
        self.connected = True
        self.closed = False
        self.contexts: list[FakeContext] = []

    def is_connected(self) -> bool:
        return self.connected

    async def new_context(self, **kwargs):
        context = FakeContext()
        self.contexts.append(context)
        return context

    async def close(self):
        self.closed = True
        self.connected = False


class FakeContext:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []
        self.closed = False
        self.close_error: Exception | None = None

    async def new_page(self) -> "FakePage":
        page = FakePage(self)
        self.pages.append(page)
        return page

    async def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeElement:
    def __init__(self, text="", *, visible=True, role=None, name=None, placeholder=None,
                 opens=None) -> None:
        self.text = text
        self.visible = visible
        self.role = role
        self.name = name
        self.placeholder = placeholder
        self.opens = opens
        self.page: FakePage | None = None
        self.value = None
        self.pressed: list[str] = []

    async def click(self, timeout=None):
        page = self.page
        page.clicked.append(self.text or self.name)
        if self.opens:
            popup = FakePage(page.context, url=self.opens, title="Popup")
            page.context.pages.append(popup)


class FakeLocator:
    def __init__(self, elements) -> None:
        self.elements = list(elements)

    async def count(self) -> int:
        return len(self.elements)

    def nth(self, i: int) -> "FakeLocator":
        return FakeLocator([self.elements[i]])

    @property
    def first(self) -> "FakeLocator":
        return FakeLocator(self.elements[:1])

    async def is_visible(self) -> bool:
        return bool(self.elements) and self.elements[0].visible

    async def click(self, timeout=None):
        await self.elements[0].click(timeout=timeout)

    async def fill(self, text):
        self.elements[0].value = text

    async def press(self, key):
        self.elements[0].pressed.append(key)

    async def inner_text(self, timeout=None) -> str:
        return self.elements[0].text


class FakeVideo:
    def __init__(self, title: str) -> None:
        self.title = title

    def locator(self, css):
        return FakeLocator([FakeElement(self.title)])


class FakeVideoList:
    def __init__(self, titles) -> None:
        self.videos = [FakeVideo(t) for t in titles]

    async def count(self) -> int:
        return len(self.videos)

    def nth(self, i: int) -> FakeVideo:
        return self.videos[i]


class FakePage:
    def __init__(self, context: FakeContext, url="https://example.test/", title="Example"):
        self.context = context
        self.url = url
        self._title = title
        self.closed = False
        self.elements: list[FakeElement] = []
        self.clicked: list[str] = []
        self.css_clicked: list[str] = []
        self.click_error: Exception | None = None
        self.load_error: Exception | None = None
        self.load_waits = 0
        self.wheel_calls: list[tuple[int, int]] = []
        self.mouse = SimpleNamespace(wheel=self._wheel)
        self.video_titles: list[str] = []

    def add(self, *elements: FakeElement) -> "FakePage":
        for el in elements:
            el.page = self
            self.elements.append(el)
        return self

    def is_closed(self) -> bool:
        return self.closed

    async def title(self) -> str:
        return self._title

    async def wait_for_load_state(self, state="load", timeout=None):
        self.load_waits += 1
        if self.load_error is not None:
            raise self.load_error

    def get_by_text(self, text, exact=False) -> FakeLocator:
        def hit(el):
            if exact:
                return el.text == text
            return bool(el.text) and text.lower() in el.text.lower()
        return FakeLocator(e for e in self.elements if hit(e))

    def get_by_role(self, role, name="") -> FakeLocator:
        return FakeLocator(
            e for e in self.elements
            if e.role == role and e.name and name.lower() in e.name.lower()
        )

    def get_by_placeholder(self, text, exact=False) -> FakeLocator:
        return FakeLocator(
            e for e in self.elements if e.placeholder and text.lower() in e.placeholder.lower()
        )

    def get_by_label(self, text, exact=False) -> FakeLocator:
        return FakeLocator([])

    def locator(self, css):
        if "search" in css:
            box = FakeElement("searchbox")
            box.page = self
            return FakeLocator([box])
        return FakeVideoList(self.video_titles)

    async def click(self, selector, timeout=None):
        if self.click_error is not None:
            raise self.click_error
        self.css_clicked.append(selector)

    async def goto(self, url, wait_until=None, timeout=None):
        self.url = url
        return SimpleNamespace(status=200)

    async def wait_for_timeout(self, ms):
        return None

    async def _wheel(self, dx, dy):
        self.wheel_calls.append((dx, dy))


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def world(monkeypatch) -> World:
    world = World()
    api = types.ModuleType("playwright.async_api")
    api.async_playwright = lambda: FakeDriverManager(world)
    api.TimeoutError = PlaywrightTimeout
    package = types.ModuleType("playwright")
    package.async_api = api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.async_api", api)
    return world


@pytest.fixture
def session(monkeypatch, world) -> browser_tools.BrowserSession:
    fresh = browser_tools.BrowserSession()
    monkeypatch.setattr(browser_tools, "_session", fresh)
    return fresh


@pytest.fixture
def registry(session) -> ToolRegistry:
    tools = ToolRegistry()
    browser_tools.register_browser_tools(tools)
    return tools


async def open_page(session) -> FakePage:
    await session._ensure_browser()
    return session.page


# -- B1: startup is failure-safe, idempotent and serialised ------------------


async def test_launch_failure_stops_the_driver(session, world):
    world.launch_error = RuntimeError("no chrome")
    for attempt in (1, 2):
        with pytest.raises(RuntimeError, match="no chrome"):
            await session._ensure_browser()
        assert len(world.drivers) == attempt
    assert all(d.stopped for d in world.drivers)  # no node process left per retry
    assert session._playwright is None and session._browser is None
    assert not session.is_running


async def test_navigate_reports_launch_failure_without_leaking(registry, world):
    world.launch_error = RuntimeError("no chrome")
    out = await registry.execute("browser_navigate", url="https://example.test")
    assert out == "Navigation failed: no chrome"
    assert world.drivers[0].stopped


async def test_cancelled_startup_stops_the_driver_and_frees_the_lock(session, world):
    world.launch_gate = asyncio.Event()
    task = asyncio.create_task(session._ensure_browser())
    while world.launches == 0:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.drivers[0].stopped
    assert session._playwright is None and session._browser is None

    world.launch_gate.set()  # the next start must not be wedged by the cancelled one
    await session._ensure_browser()
    assert session.is_running
    assert len(world.drivers) == 2 and not world.drivers[1].stopped


async def test_closing_one_tab_reuses_the_browser(session, world):
    first = await open_page(session)
    first.closed = True  # the window stays open, only this tab went away
    assert not session.is_running

    await session._ensure_browser()

    assert session.is_running and session.page is not first
    assert len(world.drivers) == 1 and world.launches == 1
    assert session.page in session._context.pages


async def test_dead_browser_is_replaced_and_its_driver_stopped(session, world):
    page = await open_page(session)
    page.closed = True
    world.browsers[0].connected = False  # the user closed the whole browser

    await session._ensure_browser()

    assert session.is_running
    assert len(world.drivers) == 2
    assert world.drivers[0].stopped and not world.drivers[1].stopped


async def test_concurrent_startups_launch_once(session, world):
    await asyncio.gather(*(session._ensure_browser() for _ in range(4)))
    assert world.launches == 1 and len(world.drivers) == 1
    assert session.is_running


async def test_lock_is_rebuilt_for_a_new_event_loop(session):
    await session._ensure_browser()
    old_lock = session._lock
    session._lock_loop = object()  # as if a previous loop had created it
    await session.shutdown()
    await session._ensure_browser()
    assert session._lock is not old_lock


# -- B2: shutdown cleans up every step ---------------------------------------


async def test_shutdown_continues_after_context_close_fails(session, world):
    await open_page(session)
    context = session._context
    context.close_error = RuntimeError("context boom")

    failed = await session.shutdown()

    assert failed == ["context"]
    assert world.browsers[0].closed
    assert world.drivers[0].stopped
    assert session._page is None and session._context is None
    assert session._browser is None and session._playwright is None


async def test_browser_close_reports_failure_honestly(registry, session, world):
    await open_page(session)
    session._context.close_error = RuntimeError("context boom")
    out = await registry.execute("browser_close")
    assert "Browser closed" not in out
    assert "incomplete" in out and "context" in out
    assert world.drivers[0].stopped


async def test_browser_close_clean(registry, session, world):
    await open_page(session)
    assert await registry.execute("browser_close") == "Browser closed"
    assert world.drivers[0].stopped
    assert await session.shutdown() == []  # closing twice is harmless


async def test_browser_can_be_reopened_after_close(registry, session, world):
    await open_page(session)
    await registry.execute("browser_close")
    out = await registry.execute("browser_navigate", url="https://example.test/x")
    assert "https://example.test/x" in out
    assert len(world.drivers) == 2 and not world.drivers[1].stopped


# -- B3: a slow page after a performed action is not a failed action ---------


async def test_click_then_slow_page_is_reported_as_performed(registry, session):
    page = (await open_page(session)).add(FakeElement("Buy now"))
    page.load_error = PlaywrightTimeout("Timeout 10000ms exceeded.")

    out = await registry.execute("browser_click", selector="Buy now")

    assert page.clicked == ["Buy now"]
    assert out.startswith("Clicked 'Buy now'")
    assert "still loading" in out and "do not repeat" in out
    assert "failed" not in out.lower()


async def test_click_then_unexpected_wait_error_is_still_performed(registry, session):
    page = (await open_page(session)).add(FakeElement("Close"))
    page.load_error = RuntimeError("Target page has been closed")

    out = await registry.execute("browser_click", selector="Close")

    assert out.startswith("Clicked 'Close'")
    assert "could not confirm" in out and "do not repeat" in out
    assert "failed" not in out.lower()


async def test_click_on_a_normal_page_reports_title_and_url(registry, session):
    await open_page(session)
    session.page.add(FakeElement("Buy now"))
    out = await registry.execute("browser_click", selector="Buy now")
    assert out == "Clicked 'Buy now'\nPage: Example\nURL: https://example.test/"


async def test_failed_click_is_still_a_failure(registry, session):
    page = await open_page(session)
    page.click_error = RuntimeError("no such element")
    out = await registry.execute("browser_click", selector="#missing")
    assert out == "Click failed for '#missing': no such element"


async def test_type_enter_then_slow_page_is_reported_as_performed(registry, session):
    field = FakeElement(placeholder="Search")
    page = (await open_page(session)).add(field)
    page.load_error = PlaywrightTimeout("Timeout 10000ms exceeded.")

    out = await registry.execute(
        "browser_type", selector="Search", text="hello", press_enter=True)

    assert field.value == "hello" and field.pressed == ["Enter"]
    assert out.startswith("Typed 'hello' into field 'Search' and pressed Enter")
    assert "still loading" in out and "failed" not in out.lower()


async def test_type_without_enter_does_not_wait_for_the_page(registry, session):
    field = FakeElement(placeholder="Search")
    page = (await open_page(session)).add(field)
    out = await registry.execute("browser_type", selector="Search", text="hello")
    assert out == "Typed 'hello' into field 'Search'"
    assert page.load_waits == 0


# -- B4: text clicks must be unambiguous -------------------------------------


async def test_ambiguous_text_click_lists_candidates_and_does_not_click(registry, session):
    page = (await open_page(session)).add(
        *(FakeElement(f"Read more {i}") for i in range(8)))

    out = await registry.execute("browser_click", selector="Read more")

    assert page.clicked == [] and page.css_clicked == []
    assert out.startswith("Nothing was clicked")
    assert "more specific selector" in out
    listed = [line for line in out.splitlines() if line.startswith("  ") and ". Read more" in line]
    assert len(listed) == browser_tools.MAX_CANDIDATES
    assert "Read more 0" in out and "Read more 7" not in out


async def test_exact_match_wins_over_longer_text(registry, session):
    page = (await open_page(session)).add(FakeElement("Login help"), FakeElement("Login"))
    out = await registry.execute("browser_click", selector="Login")
    assert page.clicked == ["Login"]
    assert out.startswith("Clicked 'Login'")


async def test_two_exact_matches_are_ambiguous(registry, session):
    page = (await open_page(session)).add(FakeElement("Login"), FakeElement("Login"))
    out = await registry.execute("browser_click", selector="Login")
    assert page.clicked == []
    assert out.startswith("Nothing was clicked")


async def test_unique_substring_match_is_clicked(registry, session):
    page = (await open_page(session)).add(FakeElement("Sign in to continue"))
    out = await registry.execute("browser_click", selector="Sign in")
    assert page.clicked == ["Sign in to continue"]
    assert out.startswith("Clicked 'Sign in'")


async def test_hidden_matches_are_ignored(registry, session):
    page = (await open_page(session)).add(
        FakeElement("Next", visible=False), FakeElement("Next"))
    out = await registry.execute("browser_click", selector="Next")
    assert page.clicked == ["Next"]
    assert out.startswith("Clicked 'Next'")


async def test_button_role_is_used_when_no_text_matches(registry, session):
    page = (await open_page(session)).add(FakeElement(role="button", name="Submit order"))
    out = await registry.execute("browser_click", selector="Submit")
    assert page.clicked == ["Submit order"]
    assert out.startswith("Clicked button 'Submit'")


async def test_css_selector_still_falls_through(registry, session):
    page = (await open_page(session)).add(FakeElement("Something else"))
    out = await registry.execute("browser_click", selector="#my-button")
    assert page.css_clicked == ["#my-button"]
    assert out.startswith("Clicked '#my-button'")


# -- B5: honest strings ------------------------------------------------------


async def test_click_that_opens_a_tab_switches_to_it(registry, session):
    page = (await open_page(session)).add(FakeElement("Open docs", opens="https://docs.test/"))

    out = await registry.execute("browser_click", selector="Open docs")

    assert session.page is not page
    assert session.page.url == "https://docs.test/"
    assert "new tab" in out and "Page: Popup" in out and "URL: https://docs.test/" in out


async def test_click_without_popup_keeps_the_active_tab(registry, session):
    page = (await open_page(session)).add(FakeElement("Plain"))
    await registry.execute("browser_click", selector="Plain")
    assert session.page is page


@pytest.mark.parametrize("direction", ["sideways", "", "left", "forward"])
async def test_scroll_rejects_unknown_direction(registry, session, direction):
    page = await open_page(session)
    out = await registry.execute("browser_scroll", direction=direction)
    assert out.startswith("Invalid scroll direction")
    assert page.wheel_calls == []


@pytest.mark.parametrize("amount", [0, -200])
async def test_scroll_rejects_non_positive_amount(registry, session, amount):
    page = await open_page(session)
    out = await registry.execute("browser_scroll", direction="down", amount=amount)
    assert out.startswith("Invalid scroll amount")
    assert page.wheel_calls == []


async def test_scroll_directions(registry, session):
    page = await open_page(session)
    assert await registry.execute("browser_scroll", direction="DOWN") == "Scrolled down by 500px"
    assert await registry.execute(
        "browser_scroll", direction=" up ", amount=300) == "Scrolled up by 300px"
    assert page.wheel_calls == [(0, 500), (0, -300)]


async def test_scroll_needs_an_open_browser(registry):
    out = await registry.execute("browser_scroll", direction="down")
    assert out == "Browser not open. Use browser_navigate first."


async def test_youtube_zero_results_is_not_reported_as_loaded(registry, session):
    await open_page(session)
    out = await registry.execute("browser_youtube_search", query="lofi")
    assert "Results loaded" not in out
    assert "could not read any video titles" in out
    assert "URL: https://www.youtube.com" in out


async def test_youtube_results_are_listed(registry, session):
    page = await open_page(session)
    page.video_titles = ["Lofi beats", "Rain sounds"]
    out = await registry.execute("browser_youtube_search", query="lofi")
    assert out.startswith("YouTube results for 'lofi':\n1. Lofi beats\n2. Rain sounds\n")
