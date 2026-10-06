"""Run scenarios against a live agent and judge what happened.

The agent is driven through ``run_stream``, the path the voice loop uses, so the first-token
time is the one a person would wait for. Every tool call is recorded where it is dispatched
(``ToolRegistry.execute``) together with how it ended, and confirmation prompts are answered by
the scenario instead of a person.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import aclosing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from jarvis.agent.tool_registry import CANCELLED_MESSAGE, PERMISSION_DENIED_PREFIX, ToolRegistry
from jarvis.bench.scenarios import Scenario
from jarvis.core.orchestrator import Orchestrator
from jarvis.utils.config import DesktopConfig, JarvisConfig, SecurityConfig
from jarvis.utils.logger import begin_turn, end_turn, get_logger

logger = get_logger(__name__)

# ask(message, history) streams the reply to one user message.
AskFn = Callable[[str, list[dict[str, str]]], AsyncIterator[str]]

_REPLY_KEPT_CHARS = 2000
_ARG_KEPT_CHARS = 200
_GPU_QUERY_TIMEOUT_S = 5.0


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    outcome: str  # ok | invalid (arguments rejected) | denied (policy) | cancelled (user said no) | error
    seconds: float


@dataclass
class Trace:
    """What the model did during one scenario."""

    approve: bool = True  # the answer given to confirmation prompts
    calls: list[ToolCall] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)  # tools the user was asked about


@dataclass
class RunResult:
    scenario: str
    run: int
    passed: bool
    failures: list[str]
    seconds: float
    first_token_s: float | None
    reply: str
    tool_calls: list[ToolCall]


@dataclass
class Resources:
    """Memory in use; None where it cannot be measured (no NVIDIA GPU, server not Jarvis's own)."""

    gpu_baseline_mb: int | None = None  # whole-GPU use before the model was loaded
    gpu_peak_mb: int | None = None
    jarvis_mb: float | None = None
    server_mb: float | None = None  # the llama-server Jarvis launched


@dataclass
class Report:
    provider: str
    model: str
    repeat: int
    startup_s: float
    results: list[RunResult]
    resources: Resources
    config: str | None = None


def bench_config(config: JarvisConfig, root: Path) -> JarvisConfig:
    """``config`` with the parts a benchmark must control.

    Security is the standard profile confined to ``root`` whatever the machine's own profile
    allows, so results compare across machines and a model's mistakes stay in the sandbox.
    """
    return config.model_copy(update={
        "security": SecurityConfig(allowed_directories=[root.as_posix()]),
        "memory": config.memory.model_copy(update={"enabled": False}),
        "desktop": DesktopConfig(),
    })


def _outcome(result: Any) -> str:
    text = result if isinstance(result, str) else ""
    if text.startswith("Invalid arguments for"):
        return "invalid"
    if text.startswith(PERMISSION_DENIED_PREFIX):
        return "denied"
    if text == CANCELLED_MESSAGE:
        return "cancelled"
    return "ok"


def _brief(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value[:_ARG_KEPT_CHARS] + "..." if isinstance(value, str)
            and len(value) > _ARG_KEPT_CHARS else value for key, value in args.items()}


@contextmanager
def recording(trace: Trace) -> Iterator[None]:
    """Record every tool dispatch into ``trace`` and answer confirmations as it says."""
    original = ToolRegistry.execute

    async def answer(tool_name: str, details: dict[str, Any]) -> bool:
        trace.confirmations.append(tool_name)
        return trace.approve

    async def execute(self: ToolRegistry, tool_name: str, **kwargs: Any) -> Any:
        self.set_confirm_callback(answer)
        started = time.monotonic()
        outcome = "error"
        try:
            result = await original(self, tool_name, **kwargs)
            outcome = _outcome(result)
            return result
        finally:
            trace.calls.append(ToolCall(tool_name, _brief(kwargs), outcome,
                                        time.monotonic() - started))

    ToolRegistry.execute = execute  # type: ignore[method-assign]
    try:
        yield
    finally:
        ToolRegistry.execute = original  # type: ignore[method-assign]


def evaluate(scenario: Scenario, workspace: Path, trace: Trace, reply: str,
             seconds: float) -> list[str]:
    """Everything that does not hold, one sentence each; empty means the scenario passed."""
    expect = scenario.expect
    problems: list[str] = []
    if not reply.strip():
        problems.append("no reply")

    called = [call.name for call in trace.calls]
    if expect.no_tools and called:
        problems.append(f"used tools although none were needed: {', '.join(called)}")
    for wanted in expect.tools_called:
        if not any(name in wanted.split("|") for name in called):
            problems.append(f"never called {wanted}" + (f" (called: {', '.join(called)})"
                                                         if called else ""))
    for forbidden in expect.tools_not_called:
        if forbidden in called:
            problems.append(f"called {forbidden}, which it should not")
    if expect.max_tool_calls is not None and len(called) > expect.max_tool_calls:
        problems.append(f"{len(called)} tool calls, at most {expect.max_tool_calls} expected")

    for pattern in expect.reply_matches:
        if not re.search(pattern, reply, re.IGNORECASE | re.DOTALL):
            problems.append(f"reply does not match {pattern!r}")
    for pattern in expect.reply_excludes:
        if re.search(pattern, reply, re.IGNORECASE | re.DOTALL):
            problems.append(f"reply matches {pattern!r}, which it should not")

    for relative, wanted_file in expect.files.items():
        target = workspace / relative
        if not wanted_file.exists:
            if target.exists():
                problems.append(f"{relative} exists, but should not")
            continue
        if not target.is_file():
            problems.append(f"{relative} was not created")
        elif wanted_file.contains is not None:
            text = target.read_text(encoding="utf-8", errors="replace")
            if wanted_file.contains.lower() not in text.lower():
                problems.append(f"{relative} does not contain {wanted_file.contains!r}")

    if expect.max_seconds is not None and seconds > expect.max_seconds:
        problems.append(f"took {seconds:.1f}s, limit {expect.max_seconds:g}s")
    return problems


async def _converse(ask: AskFn, messages: list[str], workspace: Path) -> tuple[str, float | None]:
    """Send each message in turn; returns the last reply and the time to its first words."""
    history: list[dict[str, str]] = []
    reply, first_token_s = "", None
    for message in messages:
        text = message.replace("{workspace}", workspace.as_posix())
        began = time.monotonic()
        parts: list[str] = []
        async with aclosing(ask(text, history)) as stream:
            async for token in stream:
                if first_token_s is None and token.strip():
                    first_token_s = time.monotonic() - began
                parts.append(token)
        reply = "".join(parts).strip()
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
    return reply, first_token_s


async def run_scenario(scenario: Scenario, ask: AskFn, root: Path, run: int = 0) -> RunResult:
    """One attempt at ``scenario`` in a fresh workspace under ``root``."""
    workspace = root / f"{scenario.id}-{run}"
    workspace.mkdir(parents=True)
    for relative, content in scenario.files.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.replace("{workspace}", workspace.as_posix()), encoding="utf-8")

    trace = Trace(approve=scenario.confirm == "approve")
    reply, first_token_s = "", None
    failures: list[str] = []
    started = time.monotonic()
    try:
        with recording(trace):
            reply, first_token_s = await asyncio.wait_for(
                _converse(ask, scenario.messages, workspace), scenario.timeout)
    except TimeoutError:
        failures.append(f"no answer within {scenario.timeout:g}s")
    except Exception as exc:  # a crash is a failed scenario, not the end of the benchmark
        logger.warning("scenario crashed id=%s error=%s: %s", scenario.id, type(exc).__name__, exc)
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    seconds = time.monotonic() - started
    if not failures:
        failures = evaluate(scenario, workspace, trace, reply, seconds)
    return RunResult(
        scenario=scenario.id, run=run + 1, passed=not failures, failures=failures,
        seconds=seconds, first_token_s=first_token_s, reply=reply[:_REPLY_KEPT_CHARS],
        tool_calls=trace.calls)


def _gpu_memory_mb_now() -> int | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=_GPU_QUERY_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = [int(word) for word in out.split() if word.isdigit()]
    return sum(used) if used else None


async def gpu_memory_mb() -> int | None:
    """Memory in use on the NVIDIA GPU(s) in MB, or None without nvidia-smi."""
    return await asyncio.to_thread(_gpu_memory_mb_now)


def process_memory_mb(pid: int | None, *, with_children: bool = True) -> float | None:
    """Resident memory of a process (and, by default, its children) in MB; None if not running."""
    if pid is None:
        return None
    try:
        process = psutil.Process(pid)
        total = process.memory_info().rss
        if with_children:
            for child in process.children(recursive=True):
                total += child.memory_info().rss
    except psutil.Error:
        return None
    return total / (1024 * 1024)


def _larger(old: float | None, new: float | None) -> Any:
    return new if old is None else old if new is None else max(old, new)


async def _sample(resources: Resources, orchestrator: Orchestrator) -> None:
    resources.gpu_peak_mb = _larger(resources.gpu_peak_mb, await gpu_memory_mb())
    # Jarvis alone: the llama-server it launched is its child and is counted on its own line.
    resources.jarvis_mb = _larger(
        resources.jarvis_mb, process_memory_mb(psutil.Process().pid, with_children=False))
    server_pid = orchestrator.llm.server_pid if orchestrator.llm else None
    resources.server_mb = _larger(resources.server_mb, process_memory_mb(server_pid))


async def run_bench(config: JarvisConfig, scenarios: list[Scenario], *, repeat: int = 1,
                    on_result: Callable[[RunResult], None] | None = None) -> Report:
    """Start Jarvis as ``--text`` does, run every scenario ``repeat`` times, and shut down.

    Raises RuntimeError when Jarvis cannot start (model not available, server will not load...).
    """
    root = Path(tempfile.mkdtemp(prefix="jarvis-bench-")).resolve()
    resources = Resources(gpu_baseline_mb=await gpu_memory_mb())
    orchestrator = Orchestrator(bench_config(config, root))
    results: list[RunResult] = []
    try:
        began = time.monotonic()
        await orchestrator.startup(voice_enabled=False)
        startup_s = time.monotonic() - began
        agent = orchestrator.agent
        if agent is None:
            raise RuntimeError("Jarvis started without an agent")

        def ask(message: str, history: list[dict[str, str]]) -> AsyncIterator[str]:
            if not history:  # the first message of a scenario: nothing carries over from the last
                agent.new_conversation()
            return agent.run_stream(message, history, None)

        try:  # the first request of a fresh server is slower; keep that out of the numbers
            await _converse(ask, ["Hello."], root)
        except Exception as exc:
            logger.warning("warm-up request failed error=%s: %s", type(exc).__name__, exc)
        await _sample(resources, orchestrator)

        for scenario in scenarios:
            for index in range(repeat):
                begin_turn()
                try:
                    result = await run_scenario(scenario, ask, root, index)
                finally:
                    end_turn()
                results.append(result)
                await _sample(resources, orchestrator)
                if on_result is not None:
                    on_result(result)
        return Report(provider=config.llm.provider, model=config.llm.model, repeat=repeat,
                      startup_s=startup_s, results=results, resources=resources)
    finally:
        await orchestrator.shutdown()
        shutil.rmtree(root, ignore_errors=True)
