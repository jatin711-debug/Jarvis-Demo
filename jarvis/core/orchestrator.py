"""
JARVIS Orchestrator
===================
Boots the services and drives the voice loop:
wake word -> record -> STT -> agent -> TTS -> speaker.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.permissions import PermissionGateway
from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.routing import is_small_talk
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.desktop.cua import CuaDesktop, register_desktop_tools
from jarvis.memory.manager import MemoryManager
from jarvis.tools.browser_tools import register_browser_tools, shutdown_browser
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import expand_path
from jarvis.utils.logger import begin_turn, end_turn, get_logger
from jarvis.voice.cues import play_listening_cue, play_wake_cue
from jarvis.voice.devices import resolve_voice_devices
from jarvis.voice.search_monitor import monitor_search
from jarvis.voice.speech_text import speakable
from jarvis.voice.transcript import is_explicit_yes, is_garbage, is_hallucination

if TYPE_CHECKING:
    from jarvis.utils.config import JarvisConfig
    from jarvis.voice.engines import STTEngine, TTSEngine
    from jarvis.voice.implementations import OpenWakeWordEngine

logger = get_logger(__name__)

# A sentence ends at a terminator followed by whitespace, or at a newline. A terminator
# at the very end of the buffer is not enough: streamed tokens can split "3.5" into
# "3" and ".5", so the last sentence is flushed when the stream ends instead.
_SENTENCE_END = re.compile(r"[.!?]\s+|\n")
# A list item's own number ("2" in "\n2. Open it"), at the end of the text before a ".".
_LIST_NUMBER = re.compile(r"(?:^|\n)[ \t]*\d{1,2}$")

# After this many empty recordings in a row the session drops back to wake word.
_MAX_SILENT_FOLLOWUPS = 2

# A failed conversation session (or voice loop) is retried after this long, doubling
# up to the cap; a session that finishes cleanly resets the delay.
_RETRY_MIN_SECONDS = 1.0
_RETRY_MAX_SECONDS = 10.0

_sleep = asyncio.sleep  # indirected so tests can skip the waits

# Said alone, these end the whole application.
_EXIT_PHRASES = frozenset({
    "exit", "quit", "bye", "goodbye", "shut down", "stop",
    "thats all", "that is all", "never mind", "go to sleep",
})
# Said alone, these end the conversation but keep Jarvis running.
_DISMISS_PHRASES = frozenset({
    "thanks", "thank you", "ok thanks", "thats it",
    "nothing else", "im good", "all done", "done",
})

# Recorded after a reply the user talked over, so the next turn knows it was cut short.
_INTERRUPTED_NOTE = " [interrupted by the user]"


@dataclass
class _Heard:
    """An accepted transcript plus the time spent capturing it."""

    text: str
    record_s: float
    stt_s: float


def _split_sentence(buffer: str) -> tuple[str, str] | None:
    """Split off the first complete sentence of ``buffer``, if there is one."""
    for match in _SENTENCE_END.finditer(buffer):
        if buffer[match.start()] == "." and _LIST_NUMBER.search(buffer, 0, match.start()):
            continue  # the "1." that opens a list item is not the end of a sentence
        return buffer[:match.end()].strip(), buffer[match.end():]
    return None


def _describe(error: BaseException) -> str:
    """One-line ``Type: message`` text for a log field."""
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


class Orchestrator:
    """
    The brain of JARVIS: boots services and coordinates the voice loop.

    Lifecycle:
        1. startup()  - initialize all services (+ voice stack if enabled)
        2. run()      - voice loop: wake word -> record -> STT -> agent -> TTS
        3. shutdown() - clean teardown

    Logging is configured by the entry point, not here.
    """

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        # Voice settings with device names turned into indexes (see _init_voice).
        self._voice = config.voice
        self._running = False

        # Core services
        self.llm: LLMClient | None = None
        self.tools: ToolRegistry | None = None
        self._browser_tools: ToolRegistry | None = None
        self.agent: ReActAgent | None = None
        self.memory: MemoryManager | None = None
        self.desktop: CuaDesktop | None = None
        self._memory_tasks: set[asyncio.Task] = set()
        self._memory_lookup: asyncio.Task | None = None

        # Voice services
        self.stt: STTEngine | None = None
        self.tts: TTSEngine | None = None
        self.wake_word: OpenWakeWordEngine | None = None

        # Conversation state
        self._conversation_history: list[dict[str, str]] = []

        # Voice loop supervision
        self._retry_delay = _RETRY_MIN_SECONDS
        self._faults: dict[str, str] = {}  # last error per audio component, to log repeats quietly

    # ── Startup ─────────────────────────────────────────────

    async def startup(self, *, voice_enabled: bool = True) -> None:
        """Initialize all services in dependency order."""
        cfg = self.config
        logger.info("starting services provider=%s model=%s", cfg.llm.provider, cfg.llm.model)

        expand_path(cfg.system.data_dir).mkdir(parents=True, exist_ok=True)

        self.llm = LLMClient(cfg.llm, log_dir=expand_path(cfg.system.log_dir))
        await self.llm.startup()

        # System tools are always offered per request; browser and desktop
        # tools are attached only when the request needs them.
        self.tools = ToolRegistry()
        register_system_tools(self.tools, cfg.security, llm_client=self.llm)

        self._browser_tools = ToolRegistry()
        register_browser_tools(self._browser_tools)

        desktop_tools = None
        if cfg.desktop.backend == "cua":
            self.desktop = CuaDesktop(cfg.desktop)
            await self.desktop.startup()
            desktop_tools = ToolRegistry()
            register_desktop_tools(desktop_tools, self.desktop)

        gateway = PermissionGateway(cfg.security)
        registries = [r for r in (self.tools, self._browser_tools, desktop_tools) if r is not None]
        for registry in registries:
            registry.set_permission_gateway(gateway)

        self.agent = ReActAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=cfg.llm.system_prompt,
            browser_tools=self._browser_tools,
            desktop_tools=desktop_tools,
        )

        self.memory = MemoryManager(cfg.memory, cfg.llm)
        await self.memory.startup()

        # The voice stack is optional for text-driven tasks and diagnostics.
        if voice_enabled:
            await self._init_voice(cfg)

        confirm_mode = "off"
        if cfg.security.confirm_dangerous_ops:
            confirm = self._confirm_tool if voice_enabled else self._confirm_text_tool
            for registry in registries:
                registry.set_confirm_callback(confirm)
            confirm_mode = "voice" if voice_enabled else "text"

        self._running = True
        logger.info(
            "ready tools=%d browser_tools=%d desktop=%s memory=%s voice=%s confirm=%s",
            len(self.tools), len(self._browser_tools), cfg.desktop.backend,
            "on" if cfg.memory.enabled else "off", "on" if voice_enabled else "off",
            confirm_mode,
        )

        if self.tts:
            try:
                await self.tts.speak("Jarvis online. How can I help?")
            except Exception as e:
                raise RuntimeError(
                    f"speech output failed during startup ({_describe(e)}); check the speakers "
                    "and voice.output_device in config.yaml, or run with --text"
                ) from e

    async def _init_voice(self, cfg: JarvisConfig) -> None:
        """Initialize TTS, STT, and wake word engines; any failure is a startup RuntimeError."""
        from jarvis.voice.implementations import KokoroTTS, OpenWakeWordEngine, PiperTTS, WhisperSTT

        # Device names ("JBL") become indexes now, against the devices present at this moment.
        self._voice = await asyncio.to_thread(resolve_voice_devices, cfg.voice)
        voice = self._voice

        self.tts = PiperTTS(voice) if voice.tts.engine.lower() == "piper" else KokoroTTS(voice)
        self.stt = WhisperSTT(voice)
        self.wake_word = OpenWakeWordEngine(voice)

        stages = (
            ("text-to-speech", "check voice.tts in config.yaml and that the engine is installed",
             self.tts),
            ("speech recognition", "check voice.stt in config.yaml and your network connection "
             "for the first model download", self.stt),
            ("wake word", "check the microphone, voice.input_device and voice.wake_word in "
             "config.yaml", self.wake_word),
        )
        for label, hint, engine in stages:
            try:
                await engine.startup()
            except RuntimeError:
                raise  # already a one-line, actionable startup error
            except Exception as e:
                raise RuntimeError(
                    f"{label} failed to start ({_describe(e)}); {hint}, or run with --text"
                ) from e

    def _require_agent(self) -> ReActAgent:
        if self.agent is None or self.memory is None:
            raise RuntimeError("Orchestrator.startup() must complete before handling messages")
        return self.agent

    # ── Turn handling ───────────────────────────────────────

    async def ask(self, text: str) -> str:
        """Run one text turn through memory and the agent; no voice hardware needed."""
        agent = self._require_agent()
        begin_turn()
        try:
            logger.info("heard chars=%d", len(text))
            context = await self._get_memory_context(text)
            response = await agent.run(
                user_message=text,
                conversation_history=self._conversation_history,
                context=context,
            )
            logger.info("reply chars=%d", len(response))
            self._remember_exchange(text, response)
            return response
        finally:
            end_turn()

    async def _respond_streaming(self, heard: _Heard) -> str:
        """Run the agent and speak each sentence as soon as it is generated."""
        agent = self._require_agent()
        text = heard.text

        t_turn = time.monotonic()
        context = await self._get_memory_context(text)
        memory_s = time.monotonic() - t_turn

        full_response = ""
        pending = ""
        interrupted = False
        first_token_s: float | None = None
        speak_s = 0.0
        t_agent = time.monotonic()

        async def speak(sentence: str) -> bool:
            nonlocal speak_s
            started = time.monotonic()
            try:
                return await self._speak_interruptible(sentence)
            finally:
                speak_s += time.monotonic() - started

        self.wake_word.suppress()
        if self.config.voice.barge_in:
            agent.search_monitor = self._monitor_voice_search
        try:
            # Closing the stream on barge-in stops the model and, because the agent
            # only advances when asked for the next token, any further tool calls.
            async with aclosing(agent.run_stream(
                user_message=text,
                conversation_history=self._conversation_history,
                context=context,
            )) as stream:
                async for token in stream:
                    if first_token_s is None:
                        first_token_s = time.monotonic() - t_agent
                    full_response += token
                    pending += token

                    split = _split_sentence(pending)
                    if split is not None:
                        sentence, pending = split
                        if sentence:
                            interrupted = await speak(sentence)
                            if interrupted:
                                break

            leftover = pending.strip()
            if leftover and not interrupted:
                interrupted = await speak(leftover)
        finally:
            agent.search_monitor = None
            self.wake_word.unsuppress(cooldown=0.5)

        if self.config.system.latency_report:
            agent_s = time.monotonic() - t_agent - speak_s
            logger.info(
                "latency record=%.2fs stt=%.2fs memory=%.2fs first_token=%.2fs "
                "agent=%.2fs speak=%.2fs total=%.2fs%s",
                heard.record_s, heard.stt_s, memory_s, first_token_s or 0.0,
                agent_s, speak_s, time.monotonic() - t_turn,
                " barge_in=yes" if interrupted else "",
            )

        if interrupted:
            logger.info("turn cut short by barge-in; agent stopped chars=%d", len(full_response))
        self._remember_exchange(text, full_response, interrupted=interrupted)
        return full_response

    def _remember_exchange(self, user_text: str, response: str, *, interrupted: bool = False) -> None:
        """Append to the rolling history and queue a background memory write."""
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append({
            "role": "assistant",
            "content": response.rstrip() + _INTERRUPTED_NOTE if interrupted else response})
        limit = self.config.memory.max_conversation_turns * 2
        if limit == 0:
            self._conversation_history.clear()
        elif len(self._conversation_history) > limit:
            self._conversation_history = self._conversation_history[-limit:]
        self._save_conversation(user_text, response)

    # ── Speech output ───────────────────────────────────────

    async def _monitor_voice_search(self, operation):
        """Run one search while listening; never dispatch a second action concurrently."""

        async def confirmation() -> str | None:
            heard = await self._record_and_transcribe()
            return heard.text if heard else None

        return await monitor_search(
            operation,
            input_device=self._voice.input_device,
            speak=self._speak_plain,
            record_confirmation=confirmation,
            note_fault=self._note_fault,
        )

    def _note_fault(self, component: str, error: Exception) -> None:
        """Log an audio failure: WARNING the first time, DEBUG while the same error repeats."""
        text = _describe(error)
        if self._faults.get(component) == text:
            logger.debug("%s failed again error=%s", component, text)
        else:
            self._faults[component] = text
            logger.warning("%s failed error=%s (identical repeats are logged at debug "
                           "until it works again)", component, text)

    def _note_outcome(self, component: str, result: object) -> None:
        """Record how an audio task ended: an exception is a fault, a normal end clears it."""
        if isinstance(result, Exception):
            self._note_fault(component, result)
        elif not isinstance(result, BaseException):  # not a cancellation
            self._faults.pop(component, None)

    async def _speak_plain(self, text: str) -> None:
        """tts.speak, but a cancelled turn stops playback instead of finishing the sentence."""
        text = speakable(text)
        if not text:
            return
        try:
            await self.tts.speak(text)
        except asyncio.CancelledError:
            self.tts.abort()
            raise

    async def _speak_safe(self, text: str) -> bool:
        """
        Speak with wake word suppression to prevent echo triggers.
        Never raises: a dead speaker must not take the voice loop down with it.
        Returns False if the text could not be spoken.
        """
        self.wake_word.suppress()
        try:
            await self._speak_plain(text)
        except Exception as e:
            self._note_fault("speech output", e)
            logger.debug("speech failure detail", exc_info=True)
            return False
        else:
            self._faults.pop("speech output", None)
            return True
        finally:
            # Resume detection with a cooldown so residual mic audio is ignored.
            self.wake_word.unsuppress(cooldown=1.0)

    async def _speak_interruptible(self, text: str) -> bool:
        """
        Speak a sentence, allowing the user to barge in by talking over it.
        Returns True if playback was interrupted by user speech.

        A failing speaker or microphone does not fail the turn: it is logged
        (see _note_fault) and the reply carries on without that part.
        """
        text = speakable(text)
        if not text:
            return False  # nothing but markup, e.g. a "---" divider
        if not self.config.voice.barge_in:
            await self._speak_plain(text)
            return False

        from jarvis.voice.implementations import wait_for_speech

        cancel = asyncio.Event()
        speak_task = asyncio.create_task(self.tts.speak(text))
        monitor_task = asyncio.create_task(
            wait_for_speech(self._voice.input_device, cancel_event=cancel)
        )

        interrupted = False
        try:
            done, _ = await asyncio.wait({speak_task, monitor_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if monitor_task in done and not speak_task.done():
                if not monitor_task.cancelled() and monitor_task.exception() is None:
                    interrupted = monitor_task.result()
                if interrupted:
                    self.tts.abort()
        except asyncio.CancelledError:
            self.tts.abort()  # shutting down: do not wait out the rest of the sentence
            raise
        finally:
            cancel.set()
            speech, monitor = await asyncio.gather(speak_task, monitor_task,
                                                   return_exceptions=True)
            self._note_outcome("speech output", speech)
            self._note_outcome("barge-in monitor", monitor)

        if interrupted:
            logger.info("barge-in playback interrupted")
        return interrupted

    # ── Confirmation ────────────────────────────────────────

    async def _confirm_tool(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Ask the user to confirm a dangerous tool call by voice; only an explicit yes passes."""
        from jarvis.voice.implementations import record_until_silence

        if not self.tts or not self.stt:
            logger.warning("confirmation denied tool=%s reason=voice_unavailable", tool_name)
            return False

        if {"effect", "target", "action_id"} <= args.keys():
            request = (f"This will {args['effect']} using {tool_name} on {args['target']}. "
                       f"Action code {args['action_id']}. Say yes to confirm.")
        else:
            arg_summary = ", ".join(f"{k}={v}" for k, v in list(args.items())[:2])
            request = f"This will run {tool_name} with {arg_summary}. Say yes to confirm."
        await self._speak_safe(request)

        try:
            audio = await record_until_silence(
                sample_rate=16000,
                silence_duration=1.2,
                max_duration=6.0,
                min_speech_duration=0.3,
                input_device=self._voice.input_device,
            )
            heard = (await self.stt.transcribe(audio)).strip().lower()
        except Exception as e:
            logger.error("confirmation denied tool=%s reason=recording_failed error=%s", tool_name, e)
            return False

        approved = is_explicit_yes(heard)
        logger.info("confirmation tool=%s approved=%s", tool_name, approved)
        if not approved:
            await self._speak_safe("Cancelled.")
        return approved

    async def _confirm_text_tool(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Ask for an exact yes in interactive text mode; deny in pipelines."""
        if not sys.stdin.isatty():
            logger.warning("confirmation denied tool=%s reason=no_interactive_terminal", tool_name)
            return False
        summary = ", ".join(f"{key}={value!r}" for key, value in args.items())
        answer = await asyncio.to_thread(input, f"Approve {tool_name}({summary})? Type yes: ")
        return is_explicit_yes(answer)

    # ── Voice loop ──────────────────────────────────────────

    async def _record_and_transcribe(self) -> _Heard | None:
        """Record and transcribe one utterance; None if empty, hallucinated, or garbage."""
        from jarvis.voice.implementations import record_until_silence

        t_record = time.monotonic()
        try:
            audio_bytes = await record_until_silence(
                sample_rate=16000,
                input_device=self._voice.input_device,
            )
        except Exception as e:
            logger.error("recording failed error=%s", e)
            return None
        record_s = time.monotonic() - t_record

        t_stt = time.monotonic()
        try:
            text = (await self.stt.transcribe(audio_bytes)).strip()
        except Exception as e:
            logger.error("transcription failed error=%s", _describe(e))
            logger.debug("transcription failure detail", exc_info=True)
            return None
        stt_s = time.monotonic() - t_stt

        if not text:
            logger.debug("transcript ignored reason=empty record=%.2fs stt=%.2fs", record_s, stt_s)
            return None
        if is_hallucination(text, audio_bytes) or is_garbage(text):
            logger.info("transcript ignored reason=noise chars=%d", len(text))
            return None
        return _Heard(text=text, record_s=record_s, stt_s=stt_s)

    async def _conversation_session(self) -> None:
        """
        Run an interactive conversation session.

        After each response Jarvis listens for a follow-up (with a soft cue).
        If the user stays silent it drops back to wake word mode:
        one wake word -> multi-turn conversation -> silence -> idle.
        """
        silent = 0
        while self._running:
            logger.info("listening for command")
            heard = await self._record_and_transcribe()

            if heard is None:
                silent += 1
                if silent >= _MAX_SILENT_FOLLOWUPS:
                    logger.info("session ended reason=silence")
                    return
                await play_listening_cue(self._voice.output_device)
                continue

            silent = 0
            begin_turn()
            try:
                if await self._run_spoken_turn(heard):
                    return
            finally:
                end_turn()

    async def _run_spoken_turn(self, heard: _Heard) -> bool:
        """Handle one accepted utterance. Returns True when the session should end."""
        logger.info("heard chars=%d", len(heard.text))
        phrase = re.sub(r"[^\w\s']", "", heard.text.lower()).strip()

        if phrase in _EXIT_PHRASES:
            logger.info("command exit")
            await self._speak_safe("Goodbye!")
            self._running = False
            return True

        if phrase in _DISMISS_PHRASES:
            logger.info("command dismiss")
            await self._speak_safe("Alright, I'll be here if you need me.")
            return True

        try:
            response = await self._respond_streaming(heard)
        except Exception as e:
            logger.error("turn failed error=%s", e, exc_info=True)
            response = "Sorry, something went wrong while processing your request."
            await self._speak_safe(response)

        logger.info("reply chars=%d", len(response))
        await play_listening_cue(self._voice.output_device)
        return False

    async def _back_off(self, what: str, error: Exception) -> None:
        """Log a contained failure (one line; traceback at DEBUG) and wait before retrying."""
        delay = self._retry_delay
        self._retry_delay = min(delay * 2, _RETRY_MAX_SECONDS)
        logger.error("%s failed error=%s; retrying in %.0fs", what, _describe(error), delay)
        logger.debug("%s failure detail", what, exc_info=error)
        if self._running:
            await _sleep(delay)

    async def _run_session(self) -> None:
        """
        One wake-word-initiated conversation. An unexpected failure (mic unplugged,
        speech engine crash, ...) ends only this session: it is logged and the
        voice loop backs off briefly, then goes back to waiting for the wake word.
        """
        failure: Exception | None = None
        self.wake_word.suppress()
        try:
            await play_wake_cue(self._voice.output_device)
            await self._conversation_session()
        except Exception as e:
            failure = e
        finally:
            if self._running:
                self.wake_word.unsuppress(cooldown=1.0)

        if failure is None:
            self._retry_delay = _RETRY_MIN_SECONDS
        else:
            await self._back_off("session", failure)

    async def _listen_for_wake_word(self) -> None:
        wake_word = self.config.voice.wake_word
        logger.info("waiting for wake word=%r", wake_word)

        # aclosing: leaving the loop (exit phrase, cancellation) releases the mic at once.
        async with aclosing(self.wake_word.listen()) as detections:
            async for detected in detections:
                if not self._running:
                    break
                if not detected:
                    continue

                logger.info("wake word detected")
                await self._run_session()

                if self._running:
                    logger.info("waiting for wake word=%r", wake_word)

    async def run(self) -> None:
        """
        Main voice loop: wake word -> conversation session -> repeat.

        Failures are contained so one bad device or engine call cannot end the
        process: a failed session is logged and retried after a growing delay
        (1s doubling to 10s, reset by a session that ends cleanly), and a wake word
        stream that dies is restarted the same way. Only cancellation and
        KeyboardInterrupt leave this loop abnormally.
        """
        if self.wake_word is None or self.stt is None or self.tts is None:
            raise RuntimeError("Voice stack is not initialized; start with voice_enabled=True")

        self._retry_delay = _RETRY_MIN_SECONDS
        while self._running:
            try:
                await self._listen_for_wake_word()
                return  # the detector ended on its own; nothing left to supervise
            except Exception as e:
                await self._back_off("voice loop", e)

    # ── Shutdown ────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully tear down all services."""
        self._running = False
        logger.info("stopping services")

        pending = tuple(self._memory_tasks)
        if self._memory_lookup is not None:
            pending += (self._memory_lookup,)
        if pending:
            try:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True),
                                       self.config.memory.shutdown_timeout)
            except TimeoutError:
                logger.warning("memory writes exceeded shutdown deadline pending=%d", len(pending))

        services = [shutdown_browser]
        services.extend(service.shutdown for service in
                        (self.wake_word, self.stt, self.tts, self.desktop, self.memory, self.llm)
                        if service is not None)
        for close in services:
            try:
                await close()
            except Exception:
                logger.exception("service cleanup failed; continuing shutdown")

        logger.info("stopped")

    # ── Memory ──────────────────────────────────────────────

    def _save_conversation(self, text: str, response: str) -> None:
        if is_small_talk(text):
            return
        if self.memory is None or not self.config.memory.auto_save:
            return
        if len(self._memory_tasks) >= self.config.memory.pending_writes:
            logger.warning("memory write skipped reason=queue_full pending=%d", len(self._memory_tasks))
            return
        task = asyncio.create_task(self.memory.add_conversation(text, response))
        self._memory_tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self._memory_tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error("memory write failed error=%s", done.exception())

        task.add_done_callback(completed)

    async def _get_memory_context(self, text: str) -> list[str]:
        if is_small_talk(text) or not self.config.memory.enabled or self.memory is None:
            logger.debug("memory lookup skipped")
            return []
        if self._memory_lookup is not None and not self._memory_lookup.done():
            logger.warning("memory lookup skipped reason=previous_lookup_running")
            return []
        task = asyncio.create_task(self.memory.get_context(text))
        self._memory_lookup = task
        # A synchronous embedding call in a worker cannot be forcibly stopped.
        # Retain it until completion so timeouts cannot build an unbounded queue.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        started = time.monotonic()
        try:
            context = await asyncio.wait_for(asyncio.shield(task), self.config.memory.retrieval_timeout)
        except TimeoutError:
            logger.warning("memory lookup timed out limit=%.1fs; continuing without memory",
                           self.config.memory.retrieval_timeout)
            return []
        except Exception:
            logger.warning("memory lookup failed; continuing without memory", exc_info=True)
            return []
        logger.debug("memory lookup memories=%d elapsed=%.2fs", len(context), time.monotonic() - started)
        return context
