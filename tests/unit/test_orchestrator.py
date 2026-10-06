"""Orchestrator turn handling, voice-loop control flow, and service wiring (all fakes)."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import jarvis.core.orchestrator as orch
from jarvis.core.orchestrator import Orchestrator, _clip, _Heard, _split_sentence
from jarvis.utils.config import JarvisConfig


class FakeAgent:
    def __init__(self, tokens=None, reply="Done.", error=None):
        self.tokens = tokens if tokens is not None else [reply]
        self.reply = reply
        self.error = error
        self.calls = []
        self.produced = 0  # tokens the consumer actually pulled
        self.closed = False

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply

    async def run_stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        try:
            for token in self.tokens:
                self.produced += 1
                yield token
        finally:
            self.closed = True


def make_app(agent=None, *, barge_in=False, latency_report=True):
    config = JarvisConfig()
    config.voice.barge_in = barge_in
    config.system.latency_report = latency_report
    app = Orchestrator(config)
    app.agent = agent or FakeAgent()
    app.memory = SimpleNamespace(get_context=AsyncMock(return_value=[]),
                                 add_conversation=AsyncMock(), shutdown=AsyncMock())
    app.tts = SimpleNamespace(speak=AsyncMock(), abort=MagicMock())
    app.wake_word = SimpleNamespace(suppress=MagicMock(), unsuppress=MagicMock())
    app._running = True
    return app


@pytest.fixture(autouse=True)
def silent_cues(monkeypatch):
    monkeypatch.setattr(orch, "play_listening_cue", AsyncMock())
    monkeypatch.setattr(orch, "play_wake_cue", AsyncMock())


def spoken(app):
    return [call.args[0] for call in app.tts.speak.await_args_list]


def stream_sentences(tokens):
    """Feed tokens the way _respond_streaming does; return (sentences cut mid-stream, leftover)."""
    pending, sentences = "", []
    for token in tokens:
        pending += token
        while (split := _split_sentence(pending)) is not None:
            sentence, pending = split
            sentences.append(sentence)
    return sentences, pending


class TestHelpers:
    def test_split_sentence_cuts_at_the_first_terminator(self):
        assert _split_sentence("Hello there. How are") == ("Hello there.", "How are")
        assert _split_sentence("no terminator yet") is None

    def test_a_terminator_needs_whitespace_after_it(self):
        assert _split_sentence("Done!") is None  # may still be "Done!!" or "3.5": flushed at stream end
        assert _split_sentence("Done! ") == ("Done!", "")
        assert _split_sentence("Is it? Yes") == ("Is it?", "Yes")
        assert _split_sentence("line one\nline two") == ("line one", "line two")

    @pytest.mark.parametrize("text", ["3.5 degrees", "v1.2 is out", "It is 3.", "pi is 3.14159"])
    def test_decimals_and_versions_are_not_sentence_ends(self, text):
        assert _split_sentence(text) is None

    def test_a_list_number_is_not_a_sentence_of_its_own(self):
        assert _split_sentence("1. Open the file. Then") == ("1. Open the file.", "Then")
        assert _split_sentence("Steps:\n2. Save it.\n3. Close") == ("Steps:", "2. Save it.\n3. Close")
        assert _split_sentence("12. Done. ") == ("12. Done.", "")
        assert _split_sentence("It is 3. Next") == ("It is 3.", "Next")  # a number ending a sentence

    def test_a_streamed_list_is_spoken_item_by_item(self):
        sentences, leftover = stream_sentences(
            ["Steps:", "\n1", ". Open", " it.\n", "2", ". Save it", ".\n"])
        assert sentences == ["Steps:", "1. Open it.", "2. Save it."]
        assert leftover == ""

    def test_decimal_split_across_tokens_is_spoken_whole(self):
        sentences, leftover = stream_sentences(["It is 3", ".", "5 degrees", " out. ", "Bring a coat."])
        assert sentences == ["It is 3.5 degrees out."]
        assert leftover == "Bring a coat."  # spoken by the end-of-stream flush

    def test_version_split_across_tokens_is_spoken_whole(self):
        sentences, leftover = stream_sentences(["Install v1", ".", "2", ".", "3 now", ". "])
        assert sentences == ["Install v1.2.3 now."]
        assert leftover == ""

    def test_ordinary_sentences_still_split_token_by_token(self):
        sentences, leftover = stream_sentences(["Hello", " there", ".", " How are", " you", "?", " Fine."])
        assert sentences == ["Hello there.", "How are you?"]
        assert leftover == "Fine."

    def test_clip_collapses_whitespace_and_truncates(self):
        assert _clip("a\n b   c") == "a b c"
        assert _clip("x" * 50, limit=10) == "x" * 10 + "..."


class TestTextTurn:
    async def test_ask_returns_reply_and_records_history_and_memory(self):
        app = make_app(FakeAgent(reply="Sure thing."))
        assert await app.ask("what is my schedule") == "Sure thing."
        assert app._conversation_history == [
            {"role": "user", "content": "what is my schedule"},
            {"role": "assistant", "content": "Sure thing."}]
        app._memory_tasks and await next(iter(app._memory_tasks))
        app.memory.add_conversation.assert_awaited_once_with("what is my schedule", "Sure thing.")

    async def test_history_is_bounded(self):
        app = make_app()
        app.config.memory.max_conversation_turns = 2
        for i in range(5):
            await app.ask(f"question {i}")
        assert len(app._conversation_history) == 4
        assert app._conversation_history[-2]["content"] == "question 4"

    async def test_ask_before_startup_fails_clearly(self):
        with pytest.raises(RuntimeError, match="startup"):
            await Orchestrator(JarvisConfig()).ask("hello")

    async def test_records_carry_the_turn_tag_only_during_the_turn(self, caplog):
        app = make_app()
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            await app.ask("open the file list")
        from jarvis.utils.logger import current_turn
        assert current_turn() is None
        assert any(r.getMessage().startswith("heard text=") for r in caplog.records)
        assert any(r.getMessage().startswith("reply text=") for r in caplog.records)


class TestStreamingTurn:
    async def test_sentences_are_spoken_as_they_complete(self):
        app = make_app(FakeAgent(tokens=["Hello", " there.", " How are", " you?"]))
        result = await app._respond_streaming(_Heard("hi jarvis please", 1.0, 0.5))
        assert result == "Hello there. How are you?"
        assert spoken(app) == ["Hello there.", "How are you?"]
        app.wake_word.suppress.assert_called_once()
        app.wake_word.unsuppress.assert_called_once()

    async def test_trailing_text_without_a_terminator_is_still_spoken(self):
        app = make_app(FakeAgent(tokens=["First one. ", "and a tail"]))
        await app._respond_streaming(_Heard("say things", 0, 0))
        assert spoken(app) == ["First one.", "and a tail"]

    async def test_final_sentence_is_flushed_when_the_stream_ends(self):
        app = make_app(FakeAgent(tokens=["It is 3", ".", "5 degrees", " out."]))
        await app._respond_streaming(_Heard("weather", 0, 0))
        assert spoken(app) == ["It is 3.5 degrees out."]

    async def test_decimals_are_spoken_in_one_piece(self):
        app = make_app(FakeAgent(tokens=["It is 3", ".", "5 degrees", " today. ", "Stay warm."]))
        await app._respond_streaming(_Heard("weather", 0, 0))
        assert spoken(app) == ["It is 3.5 degrees today.", "Stay warm."]

    async def test_barge_in_on_the_final_sentence_is_recorded(self, caplog):
        app = make_app(FakeAgent(tokens=["First one. ", "and a tail"]))
        app._speak_interruptible = AsyncMock(side_effect=[False, True])
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            result = await app._respond_streaming(_Heard("say things", 0, 0))
        assert result == "First one. and a tail"
        assert app._speak_interruptible.await_count == 2
        assert any("barge_in=yes" in r.getMessage() for r in caplog.records)
        assert any("cut short by barge-in" in r.getMessage() for r in caplog.records)
        assert app._conversation_history[-1]["content"] == (
            "First one. and a tail [interrupted by the user]")

    async def test_latency_line_reports_each_stage(self, caplog):
        app = make_app(FakeAgent(tokens=["Okay."]))
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            await app._respond_streaming(_Heard("do something", 1.25, 0.5))
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("latency"))
        for field in ("record=1.25s", "stt=0.50s", "memory=", "first_token=", "agent=", "speak=", "total="):
            assert field in line

    async def test_latency_line_can_be_disabled(self, caplog):
        app = make_app(FakeAgent(tokens=["Okay."]), latency_report=False)
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            await app._respond_streaming(_Heard("do something", 0, 0))
        assert not any(r.getMessage().startswith("latency") for r in caplog.records)

    async def test_barge_in_stops_the_agent_and_marks_the_history(self, caplog):
        agent = FakeAgent(tokens=["One. ", "Two. ", "Three."])
        app = make_app(agent)
        app._speak_interruptible = AsyncMock(side_effect=[True])
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            result = await app._respond_streaming(_Heard("count", 0, 0))
        assert result == "One. "
        assert app._speak_interruptible.await_count == 1
        assert agent.produced == 1 and agent.closed  # nothing was generated after the interruption
        assert any("barge_in=yes" in r.getMessage() for r in caplog.records)
        assert app._conversation_history[-1]["content"] == "One. [interrupted by the user]"

    async def test_interrupted_reply_is_stored_in_memory_without_the_marker(self):
        app = make_app(FakeAgent(tokens=["Sure thing. ", "More."]))
        app._speak_interruptible = AsyncMock(side_effect=[True])
        await app._respond_streaming(_Heard("remember my dentist is on friday", 0, 0))
        await next(iter(app._memory_tasks))
        app.memory.add_conversation.assert_awaited_once_with(
            "remember my dentist is on friday", "Sure thing. ")

    async def test_uninterrupted_reply_is_stored_verbatim(self):
        app = make_app(FakeAgent(tokens=["All done."]))
        await app._respond_streaming(_Heard("do the thing", 0, 0))
        assert app._conversation_history[-1]["content"] == "All done."

    async def test_wake_word_is_resumed_even_if_the_agent_fails(self):
        app = make_app(FakeAgent(error=RuntimeError("model crashed")))
        with pytest.raises(RuntimeError):
            await app._respond_streaming(_Heard("do it", 0, 0))
        app.wake_word.unsuppress.assert_called_once()
        assert app._conversation_history == []


class TestSpokenTurns:
    async def test_exit_phrase_stops_the_app(self):
        app = make_app()
        assert await app._run_spoken_turn(_Heard("Goodbye!", 0, 0)) is True
        assert app._running is False
        assert spoken(app) == ["Goodbye!"]
        assert not app.agent.calls

    async def test_dismissal_ends_the_session_but_keeps_running(self):
        app = make_app()
        assert await app._run_spoken_turn(_Heard("Thank you.", 0, 0)) is True
        assert app._running is True
        assert not app.agent.calls

    async def test_ordinary_request_goes_to_the_agent_and_continues(self):
        app = make_app(FakeAgent(tokens=["Opened."]))
        assert await app._run_spoken_turn(_Heard("open notepad", 0, 0)) is False
        assert spoken(app) == ["Opened."]
        orch.play_listening_cue.assert_awaited_once()

    async def test_agent_failure_is_apologised_for_not_raised(self, caplog):
        app = make_app(FakeAgent(error=RuntimeError("boom")))
        with caplog.at_level(logging.ERROR, logger="jarvis.core.orchestrator"):
            assert await app._run_spoken_turn(_Heard("open notepad", 0, 0)) is False
        assert "something went wrong" in spoken(app)[-1]
        assert any("turn failed" in r.getMessage() for r in caplog.records)


class TestSession:
    async def test_two_silent_recordings_end_the_session(self):
        app = make_app()
        app._record_and_transcribe = AsyncMock(return_value=None)
        await app._conversation_session()
        assert app._record_and_transcribe.await_count == 2
        assert orch.play_listening_cue.await_count == 1

    async def test_speech_resets_the_silence_counter(self):
        app = make_app(FakeAgent(tokens=["Ok."]))
        heard = _Heard("open notepad", 0, 0)
        app._record_and_transcribe = AsyncMock(side_effect=[None, heard, None, None])
        await app._conversation_session()
        assert app._record_and_transcribe.await_count == 4

    async def test_exit_command_leaves_the_loop(self):
        app = make_app()
        app._record_and_transcribe = AsyncMock(return_value=_Heard("quit", 0, 0))
        await app._conversation_session()
        assert app._running is False
        assert app._record_and_transcribe.await_count == 1


class TestRecordAndTranscribe:
    @pytest.fixture
    def app(self, monkeypatch):
        app = make_app()
        monkeypatch.setattr("jarvis.voice.implementations.record_until_silence",
                            AsyncMock(return_value=b"audio"))
        return app

    def hear(self, app, text):
        app.stt = SimpleNamespace(transcribe=AsyncMock(return_value=text))

    async def test_accepted_text_carries_timings(self, app):
        self.hear(app, "  open notepad  ")
        heard = await app._record_and_transcribe()
        assert heard.text == "open notepad"
        assert heard.record_s >= 0 and heard.stt_s >= 0

    async def test_empty_transcript_is_ignored(self, app):
        self.hear(app, "   ")
        assert await app._record_and_transcribe() is None

    async def test_stuttering_noise_is_ignored(self, app, caplog):
        self.hear(app, "oh. oh. oh. oh.")
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            assert await app._record_and_transcribe() is None
        assert any("reason=noise" in r.getMessage() for r in caplog.records)

    async def test_recording_failure_is_logged_not_raised(self, app, monkeypatch, caplog):
        monkeypatch.setattr("jarvis.voice.implementations.record_until_silence",
                            AsyncMock(side_effect=OSError("no mic")))
        self.hear(app, "unused")
        with caplog.at_level(logging.ERROR, logger="jarvis.core.orchestrator"):
            assert await app._record_and_transcribe() is None
        assert any("recording failed" in r.getMessage() for r in caplog.records)

    async def test_transcription_failure_is_logged_not_raised(self, app, caplog):
        app.stt = SimpleNamespace(transcribe=AsyncMock(side_effect=RuntimeError("model crashed")))
        with caplog.at_level(logging.ERROR, logger="jarvis.core.orchestrator"):
            assert await app._record_and_transcribe() is None
        failure = next(r for r in caplog.records if "transcription failed" in r.getMessage())
        assert "model crashed" in failure.getMessage()


class TestStartupWiring:
    @pytest.fixture
    def wired(self, monkeypatch, tmp_path):
        config = JarvisConfig()
        config.system.data_dir = str(tmp_path)
        config.memory.enabled = False
        config.desktop.backend = "legacy"
        fake_llm = SimpleNamespace(provider="ollama", startup=AsyncMock(), shutdown=AsyncMock(),
                                   config=config.llm)
        monkeypatch.setattr(orch, "LLMClient", lambda cfg, **kwargs: fake_llm)
        return config

    async def test_text_mode_starts_without_voice_and_wires_confirmation(self, wired, caplog):
        wired.security.confirm_dangerous_ops = True
        app = Orchestrator(wired)
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            await app.startup(voice_enabled=False)
        assert app.agent is not None and app.tts is None and app.wake_word is None
        assert app.tools._confirm_callback == app._confirm_text_tool
        ready = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("ready"))
        assert "voice=off" in ready and "confirm=text" in ready and "desktop=legacy" in ready

    async def test_confirmation_off_leaves_tools_unwrapped(self, wired):
        wired.security.confirm_dangerous_ops = False
        app = Orchestrator(wired)
        await app.startup(voice_enabled=False)
        assert app.tools._confirm_callback is None

    async def test_gateway_is_attached_to_every_registry(self, wired):
        app = Orchestrator(wired)
        await app.startup(voice_enabled=False)
        assert app.tools._permission_gateway is not None
        assert app.tools._permission_gateway is app._browser_tools._permission_gateway

    async def test_run_requires_the_voice_stack(self):
        with pytest.raises(RuntimeError, match="Voice stack"):
            await Orchestrator(JarvisConfig()).run()


def fake_engine(startup_error=None, speak_error=None):
    """A voice engine class whose startup (and speak) can be made to fail."""
    class Engine:
        def __init__(self, voice_config):
            self.startup = AsyncMock(side_effect=startup_error)
            self.shutdown = AsyncMock()
            self.speak = AsyncMock(side_effect=speak_error)
            self.abort = MagicMock()
    return Engine


class TestVoiceStartupFailures:
    """Anything that goes wrong booting the voice stack must reach main() as a RuntimeError."""

    @pytest.fixture
    def app(self, monkeypatch, tmp_path):
        config = JarvisConfig()
        config.system.data_dir = str(tmp_path)
        config.memory.enabled = False
        config.desktop.backend = "legacy"
        fake_llm = SimpleNamespace(provider="ollama", startup=AsyncMock(), shutdown=AsyncMock(),
                                   config=config.llm)
        monkeypatch.setattr(orch, "LLMClient", lambda cfg, **kwargs: fake_llm)
        return Orchestrator(config)

    @staticmethod
    def stack(monkeypatch, tts=None, stt=None, wake=None):
        monkeypatch.setattr("jarvis.voice.implementations.KokoroTTS", tts or fake_engine())
        monkeypatch.setattr("jarvis.voice.implementations.WhisperSTT", stt or fake_engine())
        monkeypatch.setattr("jarvis.voice.implementations.OpenWakeWordEngine",
                            wake or fake_engine())

    async def test_a_healthy_stack_greets_the_user(self, app, monkeypatch):
        self.stack(monkeypatch)
        await app.startup(voice_enabled=True)
        app.tts.speak.assert_awaited_once_with("Jarvis online. How can I help?")

    @pytest.mark.parametrize("which, error, label", [
        ("stt", OSError("download failed: connection reset"), "speech recognition"),
        ("wake", OSError("No Default Input Device Available"), "wake word"),
        ("tts", ImportError("No module named 'kokoro'"), "text-to-speech"),
    ])
    async def test_engine_failures_become_one_line_startup_errors(
            self, app, monkeypatch, which, error, label):
        self.stack(monkeypatch, **{which: fake_engine(startup_error=error)})
        with pytest.raises(RuntimeError) as raised:
            await app.startup(voice_enabled=True)
        message = str(raised.value)
        assert f"{label} failed to start" in message
        assert str(error) in message  # the underlying reason is not lost
        assert "\n" not in message
        assert raised.value.__cause__ is error

    async def test_device_names_are_resolved_once_and_used_everywhere(self, app, monkeypatch):
        """Engines, recording, barge-in monitoring and cues all get the resolved indexes."""
        app.config.voice.input_device = "JBL"
        app.config.voice.output_device = "JBL"
        seen = {}

        def engine(label):
            class Engine:
                def __init__(self, voice_config):
                    seen[label] = voice_config
                    self.startup, self.shutdown = AsyncMock(), AsyncMock()
                    self.speak, self.abort = AsyncMock(), MagicMock()
            return Engine

        monkeypatch.setattr(orch, "resolve_voice_devices",
                            lambda voice: voice.model_copy(update={"input_device": 3,
                                                                    "output_device": 5}))
        monkeypatch.setattr("jarvis.voice.implementations.KokoroTTS", engine("tts"))
        monkeypatch.setattr("jarvis.voice.implementations.WhisperSTT", engine("stt"))
        monkeypatch.setattr("jarvis.voice.implementations.OpenWakeWordEngine", engine("wake"))
        await app.startup(voice_enabled=True)

        assert {label: (v.input_device, v.output_device) for label, v in seen.items()} == {
            "tts": (3, 5), "stt": (3, 5), "wake": (3, 5)}
        assert app.config.voice.input_device == "JBL", "the loaded config keeps the name"

        record = AsyncMock(return_value=b"audio")
        monkeypatch.setattr("jarvis.voice.implementations.record_until_silence", record)
        app.stt = SimpleNamespace(transcribe=AsyncMock(return_value="hello there"))
        await app._record_and_transcribe()
        assert record.await_args.kwargs["input_device"] == 3

        # The wake cue is played on the resolved output device.
        app.wake_word = SimpleNamespace(suppress=MagicMock(), unsuppress=MagicMock())
        app._conversation_session = AsyncMock()
        await app._run_session()
        orch.play_wake_cue.assert_awaited_once_with(5)

    async def test_an_unresolvable_device_name_is_a_startup_error(self, app, monkeypatch):
        app.config.voice.input_device = "JBL"
        self.stack(monkeypatch)
        monkeypatch.setattr(orch, "resolve_voice_devices",
                            MagicMock(side_effect=RuntimeError("cannot list audio devices")))
        with pytest.raises(RuntimeError, match="cannot list audio devices"):
            await app.startup(voice_enabled=True)

    async def test_startup_runtime_errors_pass_through_unchanged(self, app, monkeypatch):
        error = RuntimeError("Piper voice model missing and auto-download failed: offline")
        self.stack(monkeypatch, tts=fake_engine(startup_error=error))
        with pytest.raises(RuntimeError) as raised:
            await app.startup(voice_enabled=True)
        assert raised.value is error

    async def test_a_failing_greeting_becomes_a_startup_error(self, app, monkeypatch):
        error = OSError("Error opening OutputStream: Invalid device")
        self.stack(monkeypatch, tts=fake_engine(speak_error=error))
        with pytest.raises(RuntimeError, match="speech output failed during startup") as raised:
            await app.startup(voice_enabled=True)
        assert "Invalid device" in str(raised.value)
        assert raised.value.__cause__ is error


class TestSpeechOutput:
    """_speak_interruptible and _speak_safe: failures are visible, bounded, and never fatal."""

    @pytest.fixture
    def app(self):
        return make_app(barge_in=True)

    @pytest.fixture
    def monitor(self, monkeypatch):
        """Replace wait_for_speech; set ``monitor.result`` (value or exception) and ``.delay``."""
        class Monitor:
            result = False
            delay = 0.0
            started = 0

            async def __call__(self, input_device=None, cancel_event=None, **kwargs):
                self.started += 1
                if self.delay:
                    await asyncio.sleep(self.delay)
                if isinstance(self.result, BaseException):
                    raise self.result
                if self.result is False:  # idle until playback finishes
                    await cancel_event.wait()
                return self.result

        fake = Monitor()
        monkeypatch.setattr("jarvis.voice.implementations.wait_for_speech", fake)
        return fake

    @staticmethod
    def messages(caplog, level):
        return [r.getMessage() for r in caplog.records if r.levelno == level]

    async def test_user_speech_aborts_playback_and_reports_it(self, app, monitor):
        released = asyncio.Event()

        async def speak(text):
            await released.wait()

        app.tts.speak = AsyncMock(side_effect=speak)
        app.tts.abort = MagicMock(side_effect=released.set)
        monitor.result = True
        assert await app._speak_interruptible("A long sentence.") is True
        app.tts.abort.assert_called_once()

    async def test_markup_is_not_read_aloud(self, app, monitor):
        await app._speak_interruptible("- **Name:** see [the docs](https://example.com/x).")
        assert spoken(app) == ["Name: see the docs."]

    async def test_a_sentence_of_only_markup_says_nothing(self, app, monitor):
        assert await app._speak_interruptible("---") is False
        app.tts.speak.assert_not_awaited()
        assert monitor.started == 0

    async def test_markup_is_not_read_aloud_without_barge_in(self):
        app = make_app(barge_in=False)
        await app._speak_interruptible("## **Done** — `3` files.")
        await app._speak_safe("* * *")
        assert spoken(app) == ["Done — 3 files."]

    async def test_finished_speech_stops_the_monitor_and_is_not_an_interruption(self, app, monitor):
        assert await app._speak_interruptible("Short.") is False
        app.tts.abort.assert_not_called()
        assert monitor.started == 1

    async def test_speaker_failure_is_a_warning_once_then_debug(self, app, monitor, caplog):
        app.tts.speak = AsyncMock(side_effect=OSError("speaker unplugged"))
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            for _ in range(3):
                assert await app._speak_interruptible("Hello.") is False
        warnings = self.messages(caplog, logging.WARNING)
        assert len(warnings) == 1 and "speech output failed" in warnings[0]
        assert "speaker unplugged" in warnings[0]
        assert sum("failed again" in m for m in self.messages(caplog, logging.DEBUG)) == 2

    async def test_a_success_rearms_the_warning(self, app, monitor, caplog):
        app.tts.speak = AsyncMock(side_effect=[OSError("dead"), OSError("dead"), None, OSError("dead")])
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            for _ in range(4):
                await app._speak_interruptible("Hello.")
        assert len(self.messages(caplog, logging.WARNING)) == 2

    async def test_a_different_error_warns_again(self, app, monitor, caplog):
        app.tts.speak = AsyncMock(side_effect=[OSError("dead"), OSError("other")])
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            await app._speak_interruptible("One.")
            await app._speak_interruptible("Two.")
        assert len(self.messages(caplog, logging.WARNING)) == 2

    async def test_monitor_failure_is_visible_and_speech_still_completes(self, app, monitor, caplog):
        monitor.result = OSError("cannot open microphone")
        spoke = asyncio.Event()

        async def speak(text):
            await asyncio.sleep(0.02)  # the monitor fails first, mid-sentence
            spoke.set()

        app.tts.speak = AsyncMock(side_effect=speak)
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            for _ in range(2):
                assert await app._speak_interruptible("Hello.") is False
        assert spoke.is_set()
        warnings = self.messages(caplog, logging.WARNING)
        assert len(warnings) == 1 and "barge-in monitor failed" in warnings[0]
        assert "cannot open microphone" in warnings[0]
        app.tts.abort.assert_not_called()

    async def test_cancelling_the_turn_stops_playback_at_once(self, app, monitor):
        released = asyncio.Event()

        async def speak(text):
            await released.wait()  # a sentence that would otherwise play for seconds

        app.tts.speak = AsyncMock(side_effect=speak)
        app.tts.abort = MagicMock(side_effect=released.set)
        task = asyncio.create_task(app._speak_interruptible("A very long sentence."))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        app.tts.abort.assert_called_once()

    async def test_cancelling_without_barge_in_also_stops_playback(self):
        app = make_app(barge_in=False)
        released = asyncio.Event()

        async def speak(text):
            await released.wait()

        app.tts.speak = AsyncMock(side_effect=speak)
        app.tts.abort = MagicMock(side_effect=released.set)
        task = asyncio.create_task(app._speak_interruptible("A very long sentence."))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        app.tts.abort.assert_called_once()

    async def test_speak_safe_reports_success(self, app):
        assert await app._speak_safe("Hi.") is True
        app.wake_word.suppress.assert_called_once()
        app.wake_word.unsuppress.assert_called_once()

    async def test_speak_safe_never_raises(self, app, caplog):
        app.tts.speak = AsyncMock(side_effect=OSError("speaker unplugged"))
        with caplog.at_level(logging.WARNING, logger="jarvis.core.orchestrator"):
            assert await app._speak_safe("Hi.") is False
        assert "speaker unplugged" in self.messages(caplog, logging.WARNING)[0]
        app.wake_word.unsuppress.assert_called_once()  # detection resumes regardless

    async def test_speak_safe_lets_cancellation_through_after_aborting(self, app):
        released = asyncio.Event()

        async def speak(text):
            await released.wait()

        app.tts.speak = AsyncMock(side_effect=speak)
        app.tts.abort = MagicMock(side_effect=released.set)
        task = asyncio.create_task(app._speak_safe("Goodbye!"))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        app.tts.abort.assert_called_once()
        app.wake_word.unsuppress.assert_called_once()


class FakeWake:
    """A wake word detector: each listen() plays the next script (booleans, or an exception)."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.suppress = MagicMock()
        self.unsuppress = MagicMock()
        self.listens = 0
        self.closed = 0

    async def listen(self):
        self.listens += 1
        script = self.scripts.pop(0) if self.scripts else []
        try:
            for event in script:
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            self.closed += 1

    async def shutdown(self):
        pass


class TestVoiceLoopSupervision:
    @pytest.fixture
    def waits(self, monkeypatch):
        waited = []

        async def fake_sleep(seconds):
            waited.append(seconds)

        monkeypatch.setattr(orch, "_sleep", fake_sleep)
        return waited

    @staticmethod
    def voice_app(*scripts, agent=None, **kwargs):
        app = make_app(agent, **kwargs)
        app.wake_word = FakeWake(*scripts)
        app.stt = SimpleNamespace(transcribe=AsyncMock(return_value=""))
        return app

    @staticmethod
    def errors(caplog):
        return [r for r in caplog.records if r.levelno == logging.ERROR]

    async def test_a_failed_session_is_logged_and_the_loop_carries_on(self, waits, caplog):
        app = self.voice_app([True, True])
        app._conversation_session = AsyncMock(side_effect=[OSError("mic unplugged"), None])
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            await app.run()
        assert app._conversation_session.await_count == 2
        assert waits == [1.0]
        (failure,) = self.errors(caplog)
        assert "session failed" in failure.getMessage()
        assert "mic unplugged" in failure.getMessage()
        assert app.wake_word.unsuppress.call_count == 2  # detection is never left suppressed
        assert app.wake_word.closed == 1

    async def test_the_error_is_one_line_with_the_traceback_at_debug_only(self, waits, caplog):
        app = self.voice_app([True])
        app._conversation_session = AsyncMock(side_effect=OSError("mic\nunplugged"))
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            await app.run()
        (failure,) = self.errors(caplog)
        assert "\n" not in failure.getMessage() and not failure.exc_info
        details = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
        assert details and details[0].exc_info[1].args == ("mic\nunplugged",)

    async def test_back_off_doubles_to_a_cap_and_resets_after_a_healthy_session(self, waits):
        app = self.voice_app([True] * 8)
        boom = OSError("boom")
        app._conversation_session = AsyncMock(
            side_effect=[boom, boom, boom, boom, boom, None, boom, None])
        await app.run()
        assert waits == [1.0, 2.0, 4.0, 8.0, 10.0, 1.0]

    async def test_cancellation_is_not_swallowed(self, waits):
        app = self.voice_app([True])
        app._conversation_session = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await app.run()
        assert waits == []
        app.wake_word.unsuppress.assert_called_once()
        assert app.wake_word.closed == 1

    async def test_keyboard_interrupt_is_not_swallowed(self, waits):
        app = self.voice_app([True])
        app._conversation_session = AsyncMock(side_effect=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            await app.run()
        assert waits == []

    async def test_a_recording_error_outside_the_try_is_contained_and_exit_still_works(self, waits):
        app = self.voice_app([True, True, True])
        app._record_and_transcribe = AsyncMock(
            side_effect=[OSError("device lost"), _Heard("goodbye", 0, 0)])
        await app.run()
        assert app._running is False
        assert spoken(app) == ["Goodbye!"]
        assert waits == [1.0]
        assert app.wake_word.closed == 1  # the detector was released on the way out

    async def test_stt_errors_do_not_end_the_loop(self, waits, monkeypatch, caplog):
        monkeypatch.setattr("jarvis.voice.implementations.record_until_silence",
                            AsyncMock(return_value=b"audio"))
        app = self.voice_app([True, True])
        app.stt.transcribe = AsyncMock(side_effect=[RuntimeError("model crashed"),
                                                    RuntimeError("model crashed"), "goodbye"])
        with caplog.at_level(logging.ERROR, logger="jarvis.core.orchestrator"):
            await app.run()
        assert app._running is False and spoken(app) == ["Goodbye!"]
        assert sum("transcription failed" in r.getMessage() for r in caplog.records) == 2

    async def test_tts_errors_do_not_end_the_loop_and_goodbye_still_exits(self, waits, caplog):
        app = self.voice_app([True, True], agent=FakeAgent(tokens=["Sure thing."]))
        app.tts.speak = AsyncMock(side_effect=OSError("speaker unplugged"))
        app._record_and_transcribe = AsyncMock(
            side_effect=[_Heard("what time is it", 0, 0), _Heard("goodbye", 0, 0)])
        with caplog.at_level(logging.DEBUG, logger="jarvis.core.orchestrator"):
            await app.run()
        assert app._running is False
        assert waits == []  # handled where it happened; the supervisor never needed to step in
        assert any("turn failed" in r.getMessage() for r in self.errors(caplog))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "speech output failed" in warnings[0]

    async def test_a_dead_detector_stream_is_restarted_with_back_off(self, waits, caplog):
        app = self.voice_app([False, OSError("stream dead")], [True])
        app._record_and_transcribe = AsyncMock(return_value=_Heard("quit", 0, 0))
        with caplog.at_level(logging.ERROR, logger="jarvis.core.orchestrator"):
            await app.run()
        assert app.wake_word.listens == 2
        assert waits == [1.0]
        assert app._running is False
        (failure,) = self.errors(caplog)
        assert "voice loop failed" in failure.getMessage()
        assert "stream dead" in failure.getMessage()

    async def test_a_detector_that_keeps_dying_backs_off_without_spinning(self, waits):
        app = self.voice_app(*[[OSError("dead")]] * 4)
        await app.run()
        assert app.wake_word.listens == 5  # four failures, then a clean (empty) listen ends run()
        assert waits == [1.0, 2.0, 4.0, 8.0]

    async def test_detector_is_closed_when_the_app_stops(self, waits):
        app = self.voice_app([True, False, False])
        app._record_and_transcribe = AsyncMock(return_value=_Heard("quit", 0, 0))
        await app.run()
        assert app.wake_word.closed == 1

    async def test_no_back_off_wait_once_the_app_is_stopping(self, waits):
        app = self.voice_app([True])

        async def session():
            app._running = False
            raise OSError("late failure")

        app._conversation_session = session
        await app.run()
        assert waits == []

    async def test_cues_play_on_the_configured_output_device(self, waits):
        app = self.voice_app([True])
        app.config.voice.output_device = 3
        app._record_and_transcribe = AsyncMock(side_effect=[None, None])
        await app.run()
        orch.play_wake_cue.assert_awaited_once_with(3)
        orch.play_listening_cue.assert_awaited_once_with(3)
