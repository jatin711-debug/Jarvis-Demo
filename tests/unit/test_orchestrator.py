"""Orchestrator turn handling, voice-loop control flow, and service wiring (all fakes)."""

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

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply

    async def run_stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        for token in self.tokens:
            yield token


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


class TestHelpers:
    def test_split_sentence_cuts_at_the_first_terminator(self):
        assert _split_sentence("Hello there. How are") == ("Hello there.", "How are")
        assert _split_sentence("Done!") == ("Done!", "")
        assert _split_sentence("no terminator yet") is None

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

    async def test_barge_in_stops_further_speech_but_keeps_the_full_reply(self, caplog):
        app = make_app(FakeAgent(tokens=["One. ", "Two. ", "Three."]))
        app._speak_interruptible = AsyncMock(side_effect=[True])
        with caplog.at_level(logging.INFO, logger="jarvis.core.orchestrator"):
            result = await app._respond_streaming(_Heard("count", 0, 0))
        assert result == "One. Two. Three."
        assert app._speak_interruptible.await_count == 1
        assert any("barge_in=yes" in r.getMessage() for r in caplog.records)

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


class TestStartupWiring:
    @pytest.fixture
    def wired(self, monkeypatch, tmp_path):
        config = JarvisConfig()
        config.system.data_dir = str(tmp_path)
        config.memory.enabled = False
        config.desktop.backend = "legacy"
        fake_llm = SimpleNamespace(provider="ollama", startup=AsyncMock(), shutdown=AsyncMock(),
                                   config=config.llm)
        monkeypatch.setattr(orch, "LLMClient", lambda cfg: fake_llm)
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
