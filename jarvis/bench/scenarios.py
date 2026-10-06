"""Scenario files for ``python -m jarvis.bench``.

A scenario is a prompt, the files that exist before it, and what must be true after the
answer: which tools were (not) used, what the reply says, which files exist, how long it took.
Unknown keys are errors, so a typo cannot silently turn a check off.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios.yaml")


class ScenarioError(Exception):
    """A scenario file that cannot be used, described in one readable message."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _inside_workspace(relative: str) -> str:
    """A path that stays under the scenario's workspace (no drive, no root, no ``..``)."""
    parts = PurePosixPath(relative.replace("\\", "/")).parts
    if not parts or PureWindowsPath(relative).drive or relative.startswith(("/", "\\")) \
            or ".." in parts:
        raise ValueError(f"{relative!r} must be a path inside the workspace")
    return relative


class FileExpect(_Strict):
    exists: bool = True
    contains: str | None = None  # a substring of the file's text


class Expect(_Strict):
    no_tools: bool = False
    # Each entry must have been called at least once; "a|b" accepts either tool.
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = Field(default=None, ge=0)
    # Case-insensitive regular expressions searched in the final reply.
    reply_matches: list[str] = Field(default_factory=list)
    reply_excludes: list[str] = Field(default_factory=list)
    files: dict[str, FileExpect] = Field(default_factory=dict)  # relative to the workspace
    max_seconds: float | None = Field(default=None, gt=0)  # the whole scenario, all turns

    @field_validator("reply_matches", "reply_excludes")
    @classmethod
    def _patterns_compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"bad regular expression {pattern!r}: {exc}") from None
        return patterns

    @field_validator("files")
    @classmethod
    def _files_stay_inside(cls, files: dict[str, FileExpect]) -> dict[str, FileExpect]:
        for relative in files:
            _inside_workspace(relative)
        return files

    @model_validator(mode="after")
    def _consistent(self) -> Expect:
        if self.no_tools and self.tools_called:
            raise ValueError("no_tools and tools_called contradict each other")
        return self


class Scenario(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    prompt: str | None = None  # one user message...
    turns: list[str] = Field(default_factory=list)  # ...or several, answered in order
    # Files created in the workspace first; "{workspace}" in a prompt or file is its path.
    files: dict[str, str] = Field(default_factory=dict)
    # How a confirmation prompt (a tool that changes something) is answered.
    confirm: Literal["approve", "deny"] = "approve"
    timeout: float = Field(default=120.0, gt=0)  # give up on a model that never answers
    expect: Expect = Field(default_factory=Expect)

    @field_validator("files")
    @classmethod
    def _files_stay_inside(cls, files: dict[str, str]) -> dict[str, str]:
        for relative in files:
            _inside_workspace(relative)
        return files

    @model_validator(mode="after")
    def _one_way_to_ask(self) -> Scenario:
        if (self.prompt is None) == (not self.turns):
            raise ValueError("give either prompt or turns, not both and not neither")
        return self

    @property
    def messages(self) -> list[str]:
        return [self.prompt] if self.prompt is not None else list(self.turns)


def load_scenarios(path: Path | str | None = None) -> list[Scenario]:
    """Scenarios from a YAML file with a top-level ``scenarios:`` list."""
    path = Path(path) if path else DEFAULT_SCENARIOS
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path} is not valid YAML: {exc}") from exc
    items = document.get("scenarios") if isinstance(document, dict) else None
    if not isinstance(items, list) or not items:
        raise ScenarioError(f"{path} needs a top-level 'scenarios:' list")

    scenarios: list[Scenario] = []
    for position, item in enumerate(items, start=1):
        label = item.get("id") if isinstance(item, dict) and item.get("id") else f"#{position}"
        try:
            scenarios.append(Scenario.model_validate(item))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'scenario'}: {error['msg']}"
                for error in exc.errors())
            raise ScenarioError(f"{path}: scenario {label}: {problems}") from None
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            raise ScenarioError(f"{path}: scenario id {scenario.id!r} is used twice")
        seen.add(scenario.id)
    return scenarios
