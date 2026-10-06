"""The scenario benchmark: files are validated, tool calls recorded, outcomes judged, reports readable."""

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from jarvis.agent.permissions import PermissionGateway
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.bench import __main__ as bench_cli
from jarvis.bench import runner
from jarvis.bench.report import render, summarize, to_json_data
from jarvis.bench.runner import (
    Report,
    Resources,
    RunResult,
    ToolCall,
    Trace,
    bench_config,
    evaluate,
    recording,
    run_bench,
    run_scenario,
)
from jarvis.bench.scenarios import DEFAULT_SCENARIOS, Scenario, ScenarioError, load_scenarios
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import JarvisConfig, SecurityConfig


def write_scenarios(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "scenarios.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class TestScenarioFiles:
    def test_the_shipped_scenarios_are_valid_and_name_real_tools(self):
        scenarios = load_scenarios(DEFAULT_SCENARIOS)
        assert len(scenarios) >= 10
        registry = ToolRegistry()
        register_system_tools(registry, SecurityConfig(), llm_client=object())
        known = set(registry.list_tools())
        for scenario in scenarios:
            named = [*scenario.expect.tools_called, *scenario.expect.tools_not_called]
            for entry in named:
                for tool in entry.split("|"):
                    assert tool in known, f"{scenario.id} names a tool that does not exist: {tool}"

    def test_a_scenario_has_messages_from_prompt_or_turns(self):
        assert Scenario(id="a", prompt="hi").messages == ["hi"]
        assert Scenario(id="a", turns=["one", "two"]).messages == ["one", "two"]

    @pytest.mark.parametrize("body, problem", [
        ("scenarios:\n  - id: a\n    prompt: hi\n    expct: {}\n", "expct"),  # a typo is not ignored
        ("scenarios:\n  - id: a\n    prompt: hi\n    expect: {tools_calld: [x]}\n", "tools_calld"),
        ("scenarios:\n  - id: a\n", "either prompt or turns"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    turns: [x]\n", "either prompt or turns"),
        ("scenarios:\n  - id: Bad Id\n    prompt: hi\n", "id"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    files: {'../x': y}\n", "inside the workspace"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    files: {'C:/x': y}\n", "inside the workspace"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    files: {'/etc/x': y}\n", "inside the workspace"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    expect: {files: {'a/../../x': {}}}\n",
         "inside the workspace"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    expect: {reply_matches: ['(']}\n",
         "regular expression"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    expect: {no_tools: true, tools_called: [x]}\n",
         "contradict"),
        ("scenarios:\n  - id: a\n    prompt: hi\n    confirm: maybe\n", "confirm"),
        ("scenarios:\n  - id: a\n    prompt: hi\n  - id: a\n    prompt: again\n", "used twice"),
        ("scenarios: []\n", "scenarios:"),
        ("just: text\n", "scenarios:"),
        ("scenarios: [unclosed\n", "not valid YAML"),
    ])
    def test_a_bad_file_is_refused_with_the_reason(self, tmp_path, body, problem):
        with pytest.raises(ScenarioError, match=problem):
            load_scenarios(write_scenarios(tmp_path, body))

    def test_the_error_names_the_scenario(self, tmp_path):
        path = write_scenarios(tmp_path, """
            scenarios:
              - id: fine
                prompt: hi
              - id: broken
                prompt: hi
                timeout: -1
        """)
        with pytest.raises(ScenarioError, match="scenario broken"):
            load_scenarios(path)

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(ScenarioError, match="cannot read"):
            load_scenarios(tmp_path / "nope.yaml")


def scenario(**expect) -> Scenario:
    return Scenario(id="s", prompt="p", expect=expect)


def call(name: str, outcome: str = "ok") -> ToolCall:
    return ToolCall(name, {}, outcome, 0.1)


class TestEvaluate:
    def check(self, tmp_path, spec: Scenario, calls=(), reply="An answer.", seconds=1.0):
        return evaluate(spec, tmp_path, Trace(calls=list(calls)), reply, seconds)

    def test_everything_satisfied_is_no_problems(self, tmp_path):
        (tmp_path / "out.txt").write_text("Hello THERE", encoding="utf-8")
        spec = scenario(tools_called=["write_file"], reply_matches=["answer"],
                        files={"out.txt": {"contains": "hello there"}}, max_seconds=5)
        assert self.check(tmp_path, spec, [call("write_file")]) == []

    def test_an_empty_reply_is_a_failure(self, tmp_path):
        assert self.check(tmp_path, scenario(), reply="  \n") == ["no reply"]

    def test_tools_when_none_were_wanted(self, tmp_path):
        [problem] = self.check(tmp_path, scenario(no_tools=True), [call("a"), call("b")])
        assert "a, b" in problem

    def test_a_required_tool_can_be_one_of_several(self, tmp_path):
        spec = scenario(tools_called=["search_files|list_directory"])
        assert self.check(tmp_path, spec, [call("list_directory")]) == []
        [problem] = self.check(tmp_path, spec, [call("read_file")])
        assert "search_files|list_directory" in problem and "read_file" in problem
        [problem] = self.check(tmp_path, spec, [])
        assert "never called" in problem

    def test_forbidden_tools_and_too_many_calls(self, tmp_path):
        spec = scenario(tools_not_called=["delete_file"], max_tool_calls=1)
        problems = self.check(tmp_path, spec, [call("delete_file"), call("read_file")])
        assert any("delete_file" in p for p in problems) and any("2 tool calls" in p for p in problems)

    def test_reply_patterns_ignore_case_and_line_breaks(self, tmp_path):
        spec = scenario(reply_matches=["red.*blue"], reply_excludes=["secret"])
        assert self.check(tmp_path, spec, reply="RED\nthen Blue") == []
        problems = self.check(tmp_path, spec, reply="green and a SECRET")
        assert len(problems) == 2

    def test_files_that_must_exist_or_must_not(self, tmp_path):
        (tmp_path / "kept.txt").write_text("keep", encoding="utf-8")
        spec = scenario(files={"kept.txt": {}, "missing.txt": {}, "gone.txt": {"exists": False}})
        assert self.check(tmp_path, spec) == ["missing.txt was not created"]
        (tmp_path / "gone.txt").write_text("x", encoding="utf-8")
        assert any("gone.txt exists" in p for p in self.check(tmp_path, spec))

    def test_file_content_is_checked(self, tmp_path):
        (tmp_path / "f.txt").write_text("something else", encoding="utf-8")
        spec = scenario(files={"f.txt": {"contains": "hi there"}})
        assert "does not contain" in self.check(tmp_path, spec)[0]

    def test_too_slow_is_a_failure(self, tmp_path):
        [problem] = self.check(tmp_path, scenario(max_seconds=2), seconds=3.5)
        assert "3.5s" in problem and "2s" in problem


@pytest.fixture
def sandbox(tmp_path):
    """A registry with the real file tools, confined to ``root`` by the real permission gateway."""
    root = tmp_path.resolve()
    security = SecurityConfig(allowed_directories=[root.as_posix()])
    registry = ToolRegistry()
    register_system_tools(registry, security, llm_client=object())
    registry.set_permission_gateway(PermissionGateway(security))
    return registry, root


def acting(registry: ToolRegistry, *steps, reply="Done."):
    """An agent that makes the given tool calls, ``(tool, {kwargs})``, then answers."""
    async def ask(message, history):
        for tool, kwargs in steps:
            await registry.execute(tool, **kwargs)
        yield reply
    return ask


class TestRunScenario:
    async def test_an_approved_write_is_recorded_and_checked(self, sandbox):
        registry, root = sandbox
        target = (root / "write-0" / "hello.txt").as_posix()
        spec = Scenario(id="write", prompt="make hello.txt in {workspace}", expect={
            "tools_called": ["write_file"], "files": {"hello.txt": {"contains": "hi"}}})
        ask = acting(registry, ("write_file", {"file_path": target, "content": "hi there"}))
        result = await run_scenario(spec, ask, root)
        assert result.passed, result.failures
        assert [(c.name, c.outcome) for c in result.tool_calls] == [("write_file", "ok")]
        assert result.first_token_s is not None and result.run == 1

    async def test_a_declined_action_does_not_happen_and_is_recorded(self, sandbox):
        registry, root = sandbox
        victim = (root / "decline-0" / "old.txt").as_posix()
        spec = Scenario(id="decline", prompt="delete it", confirm="deny",
                        files={"old.txt": "keep me"},
                        expect={"tools_called": ["delete_file"], "files": {"old.txt": {}}})
        result = await run_scenario(spec, acting(registry, ("delete_file", {"path": victim})), root)
        assert result.passed, result.failures
        assert result.tool_calls[0].outcome == "cancelled"
        assert (root / "decline-0" / "old.txt").read_text(encoding="utf-8") == "keep me"

    async def test_a_target_outside_the_workspace_is_denied(self, sandbox, tmp_path_factory):
        registry, root = sandbox
        outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
        outside.write_text("hidden", encoding="utf-8")
        spec = Scenario(id="scope", prompt="read it")
        ask = acting(registry, ("read_file", {"file_path": outside.as_posix()}))
        result = await run_scenario(spec, ask, root)
        assert [c.outcome for c in result.tool_calls] == ["denied"]

    async def test_bad_arguments_are_counted_as_invalid(self, sandbox):
        registry, root = sandbox
        target = (root / "args-0" / "x.txt").as_posix()
        spec = Scenario(id="args", prompt="write")
        result = await run_scenario(
            spec, acting(registry, ("write_file", {"file_path": target})), root)  # no content
        assert [c.outcome for c in result.tool_calls] == ["invalid"]

    async def test_prompts_and_files_get_the_workspace_path(self, sandbox):
        registry, root = sandbox
        seen = []

        async def ask(message, history):
            seen.append(message)
            yield "ok"

        spec = Scenario(id="paths", prompt="look in {workspace}", files={"n.txt": "at {workspace}"})
        await run_scenario(spec, ask, root)
        workspace = (root / "paths-0").as_posix()
        assert seen == [f"look in {workspace}"]
        assert (root / "paths-0" / "n.txt").read_text(encoding="utf-8") == f"at {workspace}"

    async def test_turns_see_the_conversation_so_far_and_the_last_reply_is_judged(self, sandbox):
        _, root = sandbox
        histories = []

        async def ask(message, history):
            histories.append([dict(item) for item in history])
            yield f"re: {message}"

        spec = Scenario(id="talk", turns=["first", "second"], expect={"reply_matches": ["re: second"]})
        result = await run_scenario(spec, ask, root)
        assert result.passed, result.failures
        assert histories[0] == []
        assert histories[1] == [{"role": "user", "content": "first"},
                                {"role": "assistant", "content": "re: first"}]

    async def test_a_crash_is_a_failed_run_not_an_exception(self, sandbox):
        _, root = sandbox

        async def ask(message, history):
            raise RuntimeError("boom")
            yield ""  # pragma: no cover

        result = await run_scenario(Scenario(id="crash", prompt="x"), ask, root)
        assert not result.passed and result.failures == ["crashed: RuntimeError: boom"]

    async def test_a_model_that_never_answers_times_out(self, sandbox):
        _, root = sandbox

        async def ask(message, history):
            await asyncio.sleep(30)
            yield "late"

        result = await run_scenario(Scenario(id="slow", prompt="x", timeout=0.05), ask, root)
        assert not result.passed and "no answer within" in result.failures[0]

    async def test_recording_is_removed_afterwards_even_after_a_failure(self, sandbox):
        registry, root = sandbox
        original = ToolRegistry.execute

        async def ask(message, history):
            await registry.execute("no_such_tool")
            yield "unreachable"

        result = await run_scenario(Scenario(id="unknown", prompt="x"), ask, root)
        assert ToolRegistry.execute is original
        assert not result.passed and "Unknown tool" in result.failures[0]
        assert [c.outcome for c in result.tool_calls] == ["error"]

    async def test_long_arguments_are_shortened_in_the_record(self, sandbox):
        registry, root = sandbox
        trace = Trace()
        with recording(trace):
            await registry.execute(
                "write_file", file_path=(root / "big.txt").as_posix(), content="x" * 5000)
        assert len(trace.calls[0].args["content"]) < 300
        assert trace.confirmations == ["write_file"]  # the scenario answered the prompt


class FakeAgent:
    def __init__(self):
        self.fresh_conversations = 0

    def new_conversation(self):
        self.fresh_conversations += 1

    async def run_stream(self, message, history, context):
        yield "All good."


class FakeOrchestrator:
    instances: list = []
    fail_startup = False

    def __init__(self, config):
        self.config = config
        self.agent = FakeAgent()
        self.llm = None
        self.closed = False
        self.voice_enabled = None
        FakeOrchestrator.instances.append(self)

    async def startup(self, *, voice_enabled=True):
        self.voice_enabled = voice_enabled
        if FakeOrchestrator.fail_startup:
            raise RuntimeError("model is not available")

    async def shutdown(self):
        self.closed = True


@pytest.fixture
def fake_jarvis(monkeypatch):
    FakeOrchestrator.instances = []
    FakeOrchestrator.fail_startup = False
    monkeypatch.setattr(runner, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(runner, "_gpu_memory_mb_now", lambda: 3000)
    return FakeOrchestrator


class TestRunBench:
    SCENARIOS = [Scenario(id="one", prompt="x", expect={"reply_matches": ["good"]}),
                 Scenario(id="two", prompt="y", expect={"reply_matches": ["bad"]})]

    def permissive(self) -> JarvisConfig:
        return JarvisConfig.model_validate({
            "security": {"confirm_dangerous_ops": False, "command_whitelist": False,
                         "allowed_directories": ["C:\\", "D:\\"]},
            "memory": {"enabled": True},
            "desktop": {"backend": "cua"},
        })

    async def test_it_runs_every_scenario_repeat_times_and_reports(self, fake_jarvis):
        seen = []
        report = await run_bench(self.permissive(), self.SCENARIOS, repeat=2, on_result=seen.append)
        assert [(r.scenario, r.run, r.passed) for r in report.results] == [
            ("one", 1, True), ("one", 2, True), ("two", 1, False), ("two", 2, False)]
        assert seen == report.results
        assert report.repeat == 2 and report.resources.gpu_peak_mb == 3000
        assert fake_jarvis.instances[0].closed

    async def test_the_machines_own_permissions_do_not_leak_into_the_benchmark(self, fake_jarvis):
        await run_bench(self.permissive(), self.SCENARIOS)
        config = fake_jarvis.instances[0].config
        assert config.security.confirm_dangerous_ops and config.security.command_whitelist
        assert len(config.security.allowed_directories) == 1
        assert "jarvis-bench-" in config.security.allowed_directories[0]
        assert not config.memory.enabled and config.desktop.backend == "legacy"
        assert fake_jarvis.instances[0].voice_enabled is False

    async def test_each_scenario_starts_a_fresh_conversation_for_the_router(self, fake_jarvis):
        multi = Scenario(id="multi", turns=["one", "two", "three"])
        await run_bench(self.permissive(), [*self.SCENARIOS, multi])
        # the warm-up, the two single-message scenarios, and once (not per turn) for the multi-turn one
        assert fake_jarvis.instances[0].agent.fresh_conversations == 4

    async def test_the_workspace_is_removed_afterwards(self, fake_jarvis):
        await run_bench(self.permissive(), self.SCENARIOS)
        root = Path(fake_jarvis.instances[0].config.security.allowed_directories[0])
        assert not root.exists()

    async def test_a_failed_start_is_reported_and_still_cleans_up(self, fake_jarvis):
        fake_jarvis.fail_startup = True
        with pytest.raises(RuntimeError, match="model is not available"):
            await run_bench(self.permissive(), self.SCENARIOS)
        instance = fake_jarvis.instances[0]
        assert instance.closed
        assert not Path(instance.config.security.allowed_directories[0]).exists()

    def test_bench_config_leaves_the_rest_of_the_config_alone(self, tmp_path):
        config = JarvisConfig.model_validate({"llm": {"provider": "llamacpp", "model": "m"},
                                              "voice": {"input_device": "JBL"}})
        derived = bench_config(config, tmp_path)
        assert derived.llm == config.llm and derived.voice == config.voice
        assert config.memory.enabled  # the caller's config is not modified


def result(scenario_id, run=1, passed=True, seconds=2.0, first=0.5, failures=(), calls=(),
           reply="ok") -> RunResult:
    return RunResult(scenario_id, run, passed, list(failures), seconds, first, reply, list(calls))


class TestReport:
    def report(self) -> Report:
        results = [
            result("read", seconds=2.0, calls=[call("read_file")]),
            result("read", run=2, seconds=4.0, calls=[call("read_file", "invalid"), call("read_file")]),
            result("write", passed=False, seconds=9.0, first=None,
                   failures=["hello.txt was not created"], reply="I made the file.",
                   calls=[call("write_file", "cancelled")]),
        ]
        return Report("llamacpp", "qwen3.5-4b", 2, 7.5, results,
                      Resources(gpu_baseline_mb=500, gpu_peak_mb=3500, jarvis_mb=400.0, server_mb=1024.0))

    def test_summary_per_scenario(self):
        read, write = summarize(self.report().results)
        assert (read.scenario, read.passed, read.total, read.median_s) == ("read", 2, 2, 3.0)
        assert read.tools == ["read_file"] and read.invalid_calls == 1
        assert write.passed == 0 and write.first_token_s is None

    def test_the_text_report_shows_what_failed_and_why(self):
        text = render(self.report())
        assert "llamacpp / qwen3.5-4b" in text
        assert "2/2" in text and "0/1" in text
        assert "write (run 1): hello.txt was not created" in text
        assert "reply: I made the file." in text and "write_file (cancelled)" in text
        assert "Passed 2/3 runs (66%)" in text
        assert "rejected for bad arguments: 1" in text
        assert "GPU in use 3.4 GB (+2.9 GB since start)" in text and "llama-server 1.0 GB" in text

    def test_missing_measurements_are_left_out_not_invented(self):
        report = self.report()
        report.resources = Resources()
        assert "Peak memory" not in render(report)

    def test_json_has_runs_and_summary(self):
        data = json.loads(json.dumps(to_json_data(self.report())))
        assert data["model"] == "qwen3.5-4b" and len(data["results"]) == 3
        assert data["results"][2]["tool_calls"][0]["outcome"] == "cancelled"
        assert data["summary"][0]["scenario"] == "read"


class TestCli:
    @pytest.fixture
    def config_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  provider: ollama\n  model: test-model\n", encoding="utf-8")
        return str(path)

    @pytest.fixture
    def scenarios_file(self, tmp_path):
        return write_scenarios(tmp_path, """
            scenarios:
              - id: first
                prompt: hello
              - id: second
                prompt: bye
        """)

    @pytest.fixture(autouse=True)
    def quiet(self, monkeypatch):
        monkeypatch.setattr(bench_cli, "setup_logging", lambda **kwargs: None)

    def fake_run(self, monkeypatch, *, passed=True, error=None, seen=None):
        async def run(config, scenarios, *, repeat, on_result):
            if seen is not None:
                seen.update(ids=[s.id for s in scenarios], repeat=repeat, model=config.llm.model)
            if error:
                raise error
            results = [result(s.id, passed=passed, failures=[] if passed else ["nope"])
                       for s in scenarios]
            for item in results:
                on_result(item)
            return Report("ollama", "test-model", repeat, 1.0, results, Resources())

        monkeypatch.setattr(bench_cli, "run_bench", run)

    async def test_list_prints_the_scenarios_without_starting_anything(
            self, config_file, scenarios_file, capsys, monkeypatch):
        self.fake_run(monkeypatch, error=AssertionError("must not run"))
        code = await bench_cli.main(
            ["--config", config_file, "--scenarios", str(scenarios_file), "--list"])
        assert code == bench_cli.EXIT_OK
        assert "first: hello" in capsys.readouterr().out

    async def test_a_passing_run_exits_zero_and_can_write_json(
            self, config_file, scenarios_file, tmp_path, capsys, monkeypatch):
        seen = {}
        self.fake_run(monkeypatch, seen=seen)
        out = tmp_path / "out.json"
        code = await bench_cli.main(["--config", config_file, "--scenarios", str(scenarios_file),
                                     "--repeat", "3", "--report", str(out)])
        assert code == bench_cli.EXIT_OK
        assert seen == {"ids": ["first", "second"], "repeat": 3, "model": "test-model"}
        assert json.loads(out.read_text(encoding="utf-8"))["model"] == "test-model"
        shown = capsys.readouterr().out
        assert "ok   first (run 1)" in shown and "Passed 2/2" in shown

    async def test_failed_runs_exit_three(self, config_file, scenarios_file, capsys, monkeypatch):
        self.fake_run(monkeypatch, passed=False)
        code = await bench_cli.main(["--config", config_file, "--scenarios", str(scenarios_file)])
        assert code == bench_cli.EXIT_FAILED_RUNS
        assert "FAIL first (run 1)" in capsys.readouterr().out

    async def test_only_selects_scenarios(self, config_file, scenarios_file, monkeypatch):
        seen = {}
        self.fake_run(monkeypatch, seen=seen)
        await bench_cli.main(["--config", config_file, "--scenarios", str(scenarios_file),
                              "--only", "second"])
        assert seen["ids"] == ["second"]

    async def test_an_unknown_scenario_name_lists_the_choices(
            self, config_file, scenarios_file, capsys, monkeypatch):
        self.fake_run(monkeypatch)
        code = await bench_cli.main(["--config", config_file, "--scenarios", str(scenarios_file),
                                     "--only", "frist"])
        assert code == bench_cli.EXIT_USAGE
        err = capsys.readouterr().err
        assert "frist" in err and "first, second" in err

    async def test_a_bad_scenario_file_is_a_usage_error(self, config_file, tmp_path, capsys):
        bad = write_scenarios(tmp_path, "scenarios:\n  - id: a\n")
        code = await bench_cli.main(["--config", config_file, "--scenarios", str(bad)])
        assert code == bench_cli.EXIT_USAGE
        assert "either prompt or turns" in capsys.readouterr().err

    async def test_a_jarvis_that_cannot_start_exits_one_with_the_reason(
            self, config_file, scenarios_file, capsys, monkeypatch):
        self.fake_run(monkeypatch, error=RuntimeError("model is not pulled"))
        code = await bench_cli.main(["--config", config_file, "--scenarios", str(scenarios_file)])
        assert code == bench_cli.EXIT_STARTUP_FAILED
        assert "model is not pulled" in capsys.readouterr().err

    def test_repeat_must_be_positive(self, config_file):
        with pytest.raises(SystemExit):
            bench_cli.parse_args(["--config", config_file, "--repeat", "0"])
