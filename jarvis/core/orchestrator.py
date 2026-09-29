"""
JARVIS Orchestrator — Voice-Only
=================================
Central coordinator — boots services, manages the voice event loop.
Wake word → Record → STT → Agent → TTS → Speaker
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
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
from jarvis.utils.logger import get_logger, setup_logging

if TYPE_CHECKING:
    from jarvis.utils.config import JarvisConfig
    from jarvis.voice.engines import STTEngine, TTSEngine
    from jarvis.voice.implementations import OpenWakeWordEngine


# Known Whisper hallucination phrases on near-silent audio
_HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "you", "bye", "the end", "see you next time", "subscribe",
    "please subscribe", "like and subscribe", "goodbye",
    "thank you so much", "thanks for listening",
}

# Repeated words / stuttering patterns that indicate noise/gibberish
_GIBBERISH_RE = re.compile(r"(\b\w+\b)(\W+\1\b){2,}")  # same word 3+ times


def _is_hallucination(text: str, audio_bytes: bytes) -> bool:
    """Detect likely Whisper hallucinations on very short / quiet audio."""
    import io

    import numpy as np
    import soundfile as sf

    cleaned = text.strip().lower().rstrip(".!?,")
    if cleaned in _HALLUCINATION_PHRASES:
        # Verify audio is actually very quiet (confirms hallucination)
        try:
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
            rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
            if rms < 100:
                logger.debug("Whisper hallucination filtered: '%s' (rms=%.0f)", text, rms)
                return True
        except Exception:
            pass
    return False


def _is_garbage(text: str) -> bool:
    """Detect nonsensical / stuttered transcripts from ambient noise."""
    stripped = text.strip()
    if not stripped:
        return True

    # Heavy repetition / stuttering (e.g. "oh. oh. oh.")
    if _GIBBERISH_RE.search(stripped):
        logger.debug("Garbage filtered (repetition): '%s'", text)
        return True

    # NOTE: no length filtering — short commands like "open notepad" or
    # "take screenshot" are legitimate voice commands. Whisper
    # hallucinations are handled separately by _is_hallucination().
    return False


def _is_explicit_yes(text: str) -> bool:
    """Voice approval accepts only the requested single-word answer."""
    return text.strip().casefold().rstrip(".!?").strip() == "yes"

logger = get_logger(__name__)


class Orchestrator:
    """
    The brain of JARVIS — voice-only coordinator.

    Lifecycle:
        1. startup()  — initialize all services + voice stack
        2. run()      — voice loop: wake word → record → STT → agent → TTS
        3. shutdown() — clean teardown
    """

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self._running = False

        # Core services
        self.llm: LLMClient | None = None
        self.tools: ToolRegistry | None = None
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

        # (Echo cancellation is now handled via wake_word.suppress/unsuppress)

    async def startup(self, *, voice_enabled: bool = True) -> None:
        """Initialize all services in dependency order."""
        cfg = self.config

        # 1. Logging
        setup_logging(
            level=cfg.system.log_level,
            log_dir=str(expand_path(cfg.system.log_dir)),
        )
        logger.info("📋 Config: model=%s", cfg.llm.model)

        # 2. Data directories
        data_dir = expand_path(cfg.system.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # 3. LLM
        self.llm = LLMClient(cfg.llm)
        await self.llm.startup()

        # 4. Tools — system tools always loaded, browser tools on-demand
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
        for registry in (self.tools, self._browser_tools, desktop_tools):
            if registry is not None:
                registry.set_permission_gateway(gateway)

        # 5. Agent
        self.agent = ReActAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=cfg.llm.system_prompt,
            browser_tools=self._browser_tools,
            desktop_tools=desktop_tools,
        )
        logger.info("🤖 Agent ready with %d tools (+%d browser on-demand)",
                    len(self.tools), len(self._browser_tools))

        # 6. Memory
        self.memory = MemoryManager(cfg.memory, cfg.llm)
        await self.memory.startup()

        # 7. Voice stack is optional for text-driven tasks and diagnostics.
        if voice_enabled:
            await self._init_voice(cfg)

        # 8. Wire confirmation for dangerous tools (voice yes/no)
        if cfg.security.confirm_dangerous_ops:
            confirm = self._confirm_tool if voice_enabled else self._confirm_text_tool
            self.tools.set_confirm_callback(confirm)
            self._browser_tools.set_confirm_callback(confirm)
            if desktop_tools is not None:
                desktop_tools.set_confirm_callback(confirm)
            logger.info("🔒 %s confirmation enabled for dangerous tools",
                        "Voice" if voice_enabled else "Text")

        self._running = True
        logger.info("✅ All services initialized")

        # Startup announcement
        if self.tts:
            await self.tts.speak("Jarvis online. How can I help?")

    async def _init_voice(self, cfg: JarvisConfig) -> None:
        """Initialize TTS, STT, and WakeWord engines."""
        from jarvis.voice.implementations import KokoroTTS, OpenWakeWordEngine, PiperTTS, WhisperSTT

        engine = cfg.voice.tts.engine.lower()
        if engine == "piper":
            self.tts = PiperTTS(cfg.voice)
        else:
            self.tts = KokoroTTS(cfg.voice)
        await self.tts.startup()

        self.stt = WhisperSTT(cfg.voice)
        await self.stt.startup()

        self.wake_word = OpenWakeWordEngine(cfg.voice)
        await self.wake_word.startup()

    async def _handle_user_message(self, text: str) -> str:
        """Process a user message through the full pipeline. Returns response."""
        assert self.agent is not None
        assert self.memory is not None

        # 1. Retrieve relevant memories (parallelized inside agent if passed as task)
        context = await self._get_memory_context(text)
        if context:
            logger.debug("Retrieved %d memories", len(context))

        # 2. Run through ReAct agent
        response = await self.agent.run(
            user_message=text,
            conversation_history=self._conversation_history,
            context=context,
        )

        # 3. Update conversation history
        self._conversation_history.append({"role": "user", "content": text})
        self._conversation_history.append({"role": "assistant", "content": response})

        max_turns = self.config.memory.max_conversation_turns * 2
        if len(self._conversation_history) > max_turns:
            self._conversation_history = self._conversation_history[-max_turns:]

        # 4. Store in memory (non-blocking)
        self._save_conversation(text, response)

        return response

    async def _handle_user_message_stream(self, text: str) -> str:
        """Process user message and speak sentences as they are generated."""
        assert self.agent is not None
        assert self.memory is not None

        t_turn0 = time.monotonic()
        context = await self._get_memory_context(text)
        t_memory = time.monotonic() - t_turn0

        full_response = ""
        sentence_buf = ""
        interrupted = False
        ttft: float | None = None
        tts_total = 0.0
        t_agent0 = time.monotonic()
        _SENTENCE_ENDERS = re.compile(r"[.!?]\s+|[.!?]$|\n")

        self.wake_word.suppress()
        try:
            async for token in self.agent.run_stream(
                user_message=text,
                conversation_history=self._conversation_history,
                context=context,
            ):
                if ttft is None:
                    ttft = time.monotonic() - t_agent0
                full_response += token
                sentence_buf += token

                # Check if we have a complete sentence to speak
                match = _SENTENCE_ENDERS.search(sentence_buf)
                if match:
                    # Speak everything up to and including the sentence ender
                    end = match.end()
                    sentence = sentence_buf[:end].strip()
                    sentence_buf = sentence_buf[end:]
                    if sentence and not interrupted:
                        logger.debug("🔊 Streaming sentence: %s", sentence[:80])
                        t_tts = time.monotonic()
                        interrupted = await self._speak_interruptible(sentence)
                        tts_total += time.monotonic() - t_tts

            # Speak any remaining text
            leftover = sentence_buf.strip()
            if leftover and not interrupted:
                t_tts = time.monotonic()
                await self._speak_interruptible(leftover)
                tts_total += time.monotonic() - t_tts
        finally:
            self.wake_word.unsuppress(cooldown=0.5)

        agent_total = time.monotonic() - t_agent0
        e2e = time.monotonic() - t_turn0
        if self.config.system.latency_report:
            logger.info(
                "⏱️  memory=%.2fs llm_ttft=%.2fs agent=%.2fs tts=%.2fs e2e(agent+tts)=%.2fs%s",
                t_memory, ttft or 0.0, agent_total, tts_total, e2e,
                " [barge-in]" if interrupted else "",
            )

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": text})
        self._conversation_history.append({"role": "assistant", "content": full_response})

        max_turns = self.config.memory.max_conversation_turns * 2
        if len(self._conversation_history) > max_turns:
            self._conversation_history = self._conversation_history[-max_turns:]

        self._save_conversation(text, full_response)
        return full_response

    async def _speak_safe(self, text: str) -> None:
        """Speak with wake word suppression to prevent echo triggers."""
        self.wake_word.suppress()
        try:
            await self.tts.speak(text)
        finally:
            # Resume detection with cooldown so residual mic audio is ignored
            self.wake_word.unsuppress(cooldown=1.0)

    async def _speak_interruptible(self, text: str) -> bool:
        """
        Speak a sentence, allowing the user to barge in by talking over it.
        Returns True if playback was interrupted by user speech.
        """
        if not self.config.voice.barge_in:
            await self.tts.speak(text)
            return False

        from jarvis.voice.implementations import wait_for_speech

        cancel = asyncio.Event()
        speak_task = asyncio.create_task(self.tts.speak(text))
        monitor_task = asyncio.create_task(
            wait_for_speech(self.config.voice.input_device, cancel_event=cancel)
        )

        interrupted = False
        try:
            done, _ = await asyncio.wait(
                {speak_task, monitor_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor_task in done and not speak_task.done():
                try:
                    interrupted = monitor_task.result()
                except Exception as e:
                    logger.debug("Barge-in monitor failed: %s", e)
                if interrupted:
                    self.tts.abort()
        finally:
            cancel.set()
            results = await asyncio.gather(
                speak_task, monitor_task, return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.debug("Speak/monitor task error: %s", r)

        if interrupted:
            logger.info("⏹️  Barge-in — response playback cut short")
        return interrupted

    async def _confirm_tool(self, tool_name: str, args: dict[str, Any]) -> bool:
        """
        Ask the user to confirm a dangerous tool call (voice yes/no).
        Returns True only on an explicit yes.
        """
        from jarvis.voice.implementations import record_until_silence

        if not self.tts or not self.stt:
            logger.warning("Confirmation requested but voice stack unavailable — denying")
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
                input_device=self.config.voice.input_device,
            )
            text = (await self.stt.transcribe(audio)).strip().lower()
        except Exception as e:
            logger.error("Confirmation recording failed: %s — denying", e)
            return False

        confirmed = _is_explicit_yes(text)
        logger.info("Confirmation for %s: '%s' → %s", tool_name, text, confirmed)
        if not confirmed:
            await self._speak_safe("Cancelled.")
        return confirmed

    async def _confirm_text_tool(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Ask for an exact yes in interactive text mode; deny in pipelines."""
        if not sys.stdin.isatty():
            logger.warning("Text confirmation requires an interactive terminal; denied %s", tool_name)
            return False
        summary = ", ".join(f"{key}={value!r}" for key, value in args.items())
        answer = await asyncio.to_thread(input, f"Approve {tool_name}({summary})? Type yes: ")
        return answer.strip().casefold() == "yes"

    async def _play_beep(self) -> None:
        """Play a short beep to indicate wake word detected."""
        try:
            import numpy as np
            import sounddevice as sd
            # Short 200ms 880Hz beep
            sr = 24000
            t = np.linspace(0, 0.2, int(sr * 0.2), False)
            beep = (np.sin(2 * np.pi * 880 * t) * 0.3).astype(np.float32)
            # Fade in/out to avoid click
            fade = int(sr * 0.01)
            beep[:fade] *= np.linspace(0, 1, fade).astype(np.float32)
            beep[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: (sd.play(beep, sr), sd.wait()))
        except Exception:
            pass  # Non-critical

    async def _play_listening_beep(self) -> None:
        """Play a soft short beep to indicate ready for follow-up."""
        try:
            import numpy as np
            import sounddevice as sd
            sr = 24000
            t = np.linspace(0, 0.1, int(sr * 0.1), False)
            beep = (np.sin(2 * np.pi * 660 * t) * 0.15).astype(np.float32)
            fade = int(sr * 0.01)
            beep[:fade] *= np.linspace(0, 1, fade).astype(np.float32)
            beep[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: (sd.play(beep, sr), sd.wait()))
        except Exception:
            pass

    async def _record_and_transcribe(self) -> str | None:
        """
        Record audio and transcribe. Returns cleaned text or None if
        empty/hallucinated.
        """
        from jarvis.voice.implementations import record_until_silence

        t0 = time.monotonic()
        try:
            audio_bytes = await record_until_silence(
                sample_rate=16000,
                input_device=self.config.voice.input_device,
            )
        except Exception as e:
            logger.error("Recording failed: %s", e)
            return None
        t_record = time.monotonic() - t0

        t1 = time.monotonic()
        text = await self.stt.transcribe(audio_bytes)
        text = text.strip()
        t_stt = time.monotonic() - t1

        if self.config.system.latency_report:
            logger.info("⏱️  record=%.2fs stt=%.2fs", t_record, t_stt)

        if not text or _is_hallucination(text, audio_bytes) or _is_garbage(text):
            logger.info("(empty, hallucinated, or garbage transcription, ignoring)")
            return None

        return text

    async def _conversation_session(self) -> None:
        """
        Run an interactive conversation session.

        After each response, Jarvis immediately listens for a follow-up
        command (with a soft beep). If the user stays silent, it drops
        back to wake word mode. This gives a natural Iron Man-style flow:
        one wake word → multi-turn conversation → silence → idle.
        """
        MAX_SILENT_FOLLOWUPS = 2  # how many empty recordings before ending session
        silent_count = 0

        while self._running:
            # Listen for speech
            logger.info("🎤 Listening for command...")
            text = await self._record_and_transcribe()

            if text is None:
                silent_count += 1
                if silent_count >= MAX_SILENT_FOLLOWUPS:
                    logger.info("🔇 Session ended (silence), returning to wake word")
                    return
                # Play a subtle beep to show we're still listening
                await self._play_listening_beep()
                continue

            # Got speech — reset silence counter
            silent_count = 0
            logger.info("🧑 You: %s", text)

            # Strip punctuation for phrase matching
            text_clean = re.sub(r"[^\w\s']", "", text.lower()).strip()

            # Exit commands — shut down the app
            if text_clean in ("exit", "quit", "bye", "goodbye", "shut down", "stop",
                              "thats all", "that is all", "never mind", "go to sleep"):
                await self._speak_safe("Goodbye!")
                self._running = False
                return

            # Dismissal phrases — end the session but keep app running
            if text_clean in ("thanks", "thank you", "ok thanks", "thats it",
                              "nothing else", "im good", "all done", "done"):
                await self._speak_safe("Alright, I'll be here if you need me.")
                return

            # Process through agent (streaming: speaks sentences as they generate)
            try:
                response = await self._handle_user_message_stream(text)
            except Exception as e:
                logger.error("Agent error: %s", e)
                response = "Sorry, something went wrong while processing your request."
                self.wake_word.suppress()
                try:
                    await self.tts.speak(response)
                finally:
                    self.wake_word.unsuppress(cooldown=0.5)

            # Log final response
            logger.info("🤖 Jarvis: %s", response)

            # Soft beep to signal "I'm still listening"
            logger.debug("🔔 Playing follow-up beep...")
            await self._play_listening_beep()
            logger.debug("🔔 Beep done, looping back to listen")

    async def run(self) -> None:
        """Main voice loop: wake word → conversation session → repeat."""
        assert self.wake_word is not None
        assert self.stt is not None
        assert self.tts is not None

        logger.info("🎙️  Say '%s' to begin\n", self.config.voice.wake_word)

        async for detected in self.wake_word.listen():
            if not self._running:
                break

            if not detected:
                continue

            # Wake word heard — start a conversation session
            self.wake_word.suppress()
            await self._play_beep()

            await self._conversation_session()

            # Session ended — resume wake word detection
            if self._running:
                self.wake_word.unsuppress(cooldown=1.0)
                logger.info("🎙️ Listening for wake word '%s'...", self.config.voice.wake_word)

    async def shutdown(self) -> None:
        """Gracefully tear down all services."""
        self._running = False
        logger.info("🔄 Shutting down ...")

        pending = tuple(self._memory_tasks)
        if self._memory_lookup is not None:
            pending += (self._memory_lookup,)
        if pending:
            try:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True),
                                       self.config.memory.shutdown_timeout)
            except TimeoutError:
                logger.warning("Memory writes exceeded shutdown deadline")
        services = [shutdown_browser]
        services.extend(service.shutdown for service in
                        (self.wake_word, self.stt, self.tts, self.desktop, self.memory, self.llm)
                        if service is not None)
        for close in services:
            try:
                await close()
            except Exception:
                logger.exception("Service cleanup failed; continuing shutdown")

        logger.info("✅ Shutdown complete")

    def _save_conversation(self, text: str, response: str) -> None:
        if is_small_talk(text):
            return
        if self.memory is None or not self.config.memory.auto_save:
            return
        if len(self._memory_tasks) >= self.config.memory.pending_writes:
            logger.warning("Memory write queue full; skipping conversation persistence")
            return
        task = asyncio.create_task(self.memory.add_conversation(text, response))
        self._memory_tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self._memory_tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error("Conversation persistence failed: %s", done.exception())

        task.add_done_callback(completed)

    async def _get_memory_context(self, text: str) -> list[str]:
        if is_small_talk(text) or not self.config.memory.enabled or self.memory is None:
            logger.debug("Memory retrieval skipped for this turn")
            return []
        if self._memory_lookup is not None and not self._memory_lookup.done():
            logger.warning("Previous memory lookup still running; continuing without memory")
            return []
        task = asyncio.create_task(self.memory.get_context(text))
        self._memory_lookup = task
        # A synchronous embedding call in a worker cannot be forcibly stopped.
        # Retain it until completion so timeouts cannot build an unbounded queue.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        started = time.monotonic()
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.config.memory.retrieval_timeout)
        except TimeoutError:
            logger.warning("Memory lookup exceeded %.1fs; continuing without memory",
                           self.config.memory.retrieval_timeout)
            return []
        except Exception:
            logger.warning("Memory lookup failed; continuing without memory", exc_info=True)
            return []
        finally:
            logger.info("Memory lookup wait: %.2fs", time.monotonic() - started)
