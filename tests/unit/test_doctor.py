"""``jarvis --doctor``: each check reports an actionable result without touching hardware."""

import json
from pathlib import Path

import httpx

import jarvis.__main__ as cli
from jarvis import doctor
from jarvis.doctor import FAIL, OK, WARN, Check
from jarvis.utils.config import JarvisConfig


def config(**sections) -> JarvisConfig:
    return JarvisConfig.model_validate(sections)


def by_name(checks):
    return {c.name: c for c in checks}


class FakeSounddevice:
    def __init__(self, devices):
        self.devices = devices

    def query_devices(self):
        return self.devices


MIC = {"name": "Headset Mic", "max_input_channels": 1, "max_output_channels": 0}
SPEAKER = {"name": "Headset Out", "max_input_channels": 0, "max_output_channels": 2}


def ollama(models, status=200):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(status, content=json.dumps({"models": [{"name": m} for m in models]}))
    return httpx.MockTransport(handler)


class TestOllama:
    async def test_everything_pulled(self):
        cfg = config(llm={"model": "qwen3.5:2b"}, memory={"enabled": True})
        checks = by_name(await doctor.check_ollama(
            cfg, ollama(["qwen3.5:2b", "nomic-embed-text:latest"])))
        assert {c.status for c in checks.values()} == {OK}

    async def test_unreachable_server_says_how_to_start_it(self):
        def refuse(request):
            raise httpx.ConnectError("refused")

        [check] = await doctor.check_ollama(config(), httpx.MockTransport(refuse))
        assert check.status == FAIL and "ollama serve" in check.hint

    async def test_missing_chat_model_names_the_pull_command(self):
        checks = by_name(await doctor.check_ollama(
            config(llm={"model": "qwen3.5:2b"}), ollama(["other:1b"])))
        assert checks["ollama"].status == OK
        assert checks["chat model"].status == FAIL
        assert checks["chat model"].hint == "run: ollama pull qwen3.5:2b"

    async def test_missing_embedding_model_is_only_a_warning(self):
        checks = by_name(await doctor.check_ollama(
            config(llm={"model": "m"}, memory={"enabled": True}), ollama(["m"])))
        assert checks["embedding model"].status == WARN

    async def test_embedding_model_is_not_checked_when_memory_is_off(self):
        checks = by_name(await doctor.check_ollama(
            config(llm={"model": "m"}, memory={"enabled": False}), ollama(["m"])))
        assert "embedding model" not in checks

    async def test_http_error_from_the_server_is_a_failure(self):
        [check] = await doctor.check_ollama(config(), ollama([], status=500))
        assert check.status == FAIL


class TestAudio:
    def test_default_microphone(self):
        checks = by_name(doctor.check_audio(config(), FakeSounddevice([MIC, SPEAKER])))
        assert checks["microphone"].status == OK and checks["speaker"].status == OK

    def test_configured_device_that_is_not_a_microphone_lists_the_real_ones(self):
        checks = by_name(doctor.check_audio(
            config(voice={"input_device": 1}), FakeSounddevice([MIC, SPEAKER])))
        assert checks["microphone"].status == FAIL
        assert "0=Headset Mic" in checks["microphone"].detail
        assert "config.local.yaml" in checks["microphone"].hint

    def test_configured_device_that_exists(self):
        checks = by_name(doctor.check_audio(
            config(voice={"input_device": 0}), FakeSounddevice([MIC, SPEAKER])))
        assert checks["microphone"].status == OK and "Headset Mic" in checks["microphone"].detail

    def test_no_devices_at_all(self):
        checks = by_name(doctor.check_audio(config(), FakeSounddevice([])))
        assert checks["microphone"].status == FAIL and checks["speaker"].status == FAIL

    def test_bad_output_device_index(self):
        checks = by_name(doctor.check_audio(
            config(voice={"output_device": 0}), FakeSounddevice([MIC, SPEAKER])))
        assert checks["speaker"].status == FAIL

    def test_device_listing_error_is_reported(self):
        class Broken:
            def query_devices(self):
                raise OSError("PortAudio not initialized")

        [check] = doctor.check_audio(config(), Broken())
        assert check.status == FAIL and "PortAudio not initialized" in check.detail


class TestPackagesAndSecurity:
    def finder(self, installed):
        return lambda name: object() if name in installed else None

    def test_missing_speech_package_is_a_failure_with_an_install_hint(self):
        checks = doctor.check_packages(config(), self.finder({"pyaudio", "sounddevice", "kokoro"}))
        missing = [c for c in checks if c.status == FAIL]
        assert {c.hint for c in missing} == {"pip install faster-whisper", "pip install openwakeword"}

    def test_optional_features_only_warn(self):
        checks = by_name(doctor.check_packages(
            config(memory={"enabled": True}, desktop={"backend": "cua"}), self.finder(set())))
        assert checks["memory"].status == WARN
        assert checks["desktop control"].status == WARN
        assert checks["browser tools"].status == WARN

    def test_piper_engine_checks_piper_not_kokoro(self):
        checks = doctor.check_packages(config(voice={"tts": {"engine": "piper"}}),
                                       self.finder(set()))
        hints = {c.hint for c in checks}
        assert "pip install piper-tts" in hints and "pip install kokoro" not in hints

    def test_disabled_features_are_not_checked(self):
        checks = doctor.check_packages(
            config(memory={"enabled": False}, desktop={"backend": "legacy"}), self.finder(set()))
        assert not {"memory", "desktop control"} & {c.name for c in checks}

    def test_standard_profile_is_ok_and_full_access_warns(self):
        assert doctor.check_security(config()).status == OK
        relaxed = doctor.check_security(config(security={
            "confirm_dangerous_ops": False, "command_whitelist": False}))
        assert relaxed.status == WARN
        assert "confirmations off" in relaxed.detail and "whitelist off" in relaxed.detail


class TestDirectories:
    def test_writable_directories(self, tmp_path):
        cfg = config(system={"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path / "logs")})
        assert {c.status for c in doctor.check_directories(cfg)} == {OK}
        assert not (tmp_path / "data" / ".doctor-probe").exists()

    def test_unwritable_directory_fails(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        cfg = config(system={"data_dir": str(blocker / "data"), "log_dir": str(tmp_path / "logs")})
        statuses = {c.name: c.status for c in doctor.check_directories(cfg)}
        assert statuses == {"data directory": FAIL, "log directory": OK}


def test_browser_binary_is_checked_only_when_playwright_is_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    assert doctor.check_browser_binary(lambda name: None) is None
    missing = doctor.check_browser_binary(lambda name: object())
    assert missing.status == WARN and "playwright install chromium" in missing.hint
    (tmp_path / "chromium-1234").mkdir()
    assert doctor.check_browser_binary(lambda name: object()).status == OK


class TestReport:
    def test_hints_appear_only_under_problems_and_the_summary_counts_them(self):
        text = doctor.render([
            Check("python", OK, "3.12", hint="ignored"),
            Check("ollama", FAIL, "cannot reach", "start it with: ollama serve"),
            Check("memory", WARN, "mem0 missing", "pip install mem0ai"),
        ])
        assert "ignored" not in text
        assert "-> start it with: ollama serve" in text and "-> pip install mem0ai" in text
        assert text.rstrip().endswith("1 problem(s), 1 warning(s)")

    def test_clean_report(self):
        assert doctor.render([Check("python", OK, "3.12")]).rstrip().endswith("everything looks good")

    def test_exit_code_only_fails_on_failures(self):
        assert doctor.exit_code([Check("a", OK, ""), Check("b", WARN, "")]) == 0
        assert doctor.exit_code([Check("a", FAIL, "")]) == 1


async def test_run_checks_skips_ollama_for_the_huggingface_provider(tmp_path):
    cfg = config(llm={"provider": "huggingface", "model": "x"},
                 system={"data_dir": str(tmp_path), "log_dir": str(tmp_path)})
    checks = by_name(await doctor.run_checks(
        cfg, None, find=lambda name: object(), sd=FakeSounddevice([MIC, SPEAKER])))
    assert "ollama" not in checks and "chat model" not in checks
    assert checks["config"].status == WARN


async def test_cli_doctor_prints_the_report_and_returns_its_exit_code(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("system:\n  log_level: INFO\n", encoding="utf-8")

    async def fake_run_checks(cfg, config_path, **kwargs):
        assert Path(config_path) == path.resolve()
        return [Check("ollama", FAIL, "cannot reach", "start it with: ollama serve")]

    monkeypatch.setattr(doctor, "run_checks", fake_run_checks)
    assert await cli.main(["--config", str(path), "--doctor"]) == 1
    out = capsys.readouterr().out
    assert "Jarvis doctor" in out and "start it with: ollama serve" in out
