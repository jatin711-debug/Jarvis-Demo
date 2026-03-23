"""
JARVIS Orchestrator — Voice-Only
=================================
Central coordinator — boots services, manages the voice event loop.
Wake word → Record → STT → Agent → TTS → Speaker
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from jarvis.core.events import EventBus
from jarvis.agent.llm_client import LLMClient
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.agent.react_agent import ReActAgent
from jarvis.memory.manager import MemoryManager
from jarvis.tools.system_tools import register_system_tools
from jarvis.tools.browser_tools import register_browser_tools, shutdown_browser
from jarvis.utils.logger import get_logger, setup_logging
from jarvis.utils.config import expand_path

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


def _is_hallucination(text: str, audio_bytes: bytes) -> bool:
    """Detect likely Whisper hallucinations on very short / quiet audio."""
    import numpy as np
    import io, soundfile as sf

    # Check if the text matches known hallucination patterns
    cleaned = text.strip().lower().rstrip(".!?,")
    if cleaned not in _HALLUCINATION_PHRASES:
        return False

    # Verify audio is actually very quiet (confirms hallucination)
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
        # If audio is quiet (<100 RMS) and text is a known hallucination → skip
        if rms < 100:
            logger.debug("Whisper hallucination filtered: '%s' (rms=%.0f)", text, rms)
            return True
    except Exception:
        pass
    return False

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
        self.event_bus = EventBus()
        self._running = False

        # Core services
        self.llm: LLMClient | None = None
        self.tools: ToolRegistry | None = None
        self.agent: ReActAgent | None = None
        self.memory: MemoryManager | None = None

        # Voice services
        self.stt: STTEngine | None = None
        self.tts: TTSEngine | None = None
        self.wake_word: OpenWakeWordEngine | None = None

        # Conversation state
        self._conversation_history: list[dict[str, str]] = []

        # (Echo cancellation is now handled via wake_word.suppress/unsuppress)

    async def startup(self) -> None:
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

        # 4. Tools
        self.tools = ToolRegistry()
        register_system_tools(self.tools, cfg.security)
        register_browser_tools(self.tools)

        # 5. Agent
        self.agent = ReActAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=cfg.llm.system_prompt,
        )
        logger.info("🤖 Agent ready with %d tools", len(self.tools))

        # 6. Memory
        self.memory = MemoryManager(cfg.memory, cfg.llm)
        await self.memory.startup()

        # 7. Voice stack
        await self._init_voice(cfg)

        self._running = True
        logger.info("✅ All services initialized")

        # Startup announcement
        if self.tts:
            await self.tts.speak("Jarvis online. How can I help?")

    async def _init_voice(self, cfg: JarvisConfig) -> None:
        """Initialize TTS, STT, and WakeWord engines."""
        from jarvis.voice.implementations import KokoroTTS, WhisperSTT, OpenWakeWordEngine

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

        # 1. Retrieve relevant memories
        context = await self.memory.get_context(text)
        if context:
            logger.debug("📝 Retrieved %d memories", len(context))

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
        asyncio.create_task(self.memory.add_conversation(text, response))

        return response

    async def _speak_safe(self, text: str) -> None:
        """Speak with wake word suppression to prevent echo triggers."""
        self.wake_word.suppress()
        try:
            await self.tts.speak(text)
        finally:
            # Resume detection with cooldown so residual mic audio is ignored
            self.wake_word.unsuppress(cooldown=1.0)

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

        try:
            audio_bytes = await record_until_silence(
                sample_rate=16000,
                input_device=self.config.voice.input_device,
            )
        except Exception as e:
            logger.error("Recording failed: %s", e)
            return None

        text = await self.stt.transcribe(audio_bytes)
        text = text.strip()

        if not text or _is_hallucination(text, audio_bytes):
            logger.info("(empty or hallucinated transcription, ignoring)")
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

            # Exit commands — shut down the app
            if text.lower() in ("exit", "quit", "bye", "goodbye", "shut down", "stop",
                                "that's all", "that is all", "never mind", "go to sleep"):
                await self._speak_safe("Goodbye!")
                self._running = False
                return

            # Dismissal phrases — end the session but keep app running
            if text.lower() in ("thanks", "thank you", "ok thanks", "that's it",
                                "nothing else", "i'm good", "all done", "done"):
                await self._speak_safe("Alright, I'll be here if you need me.")
                return

            # Process through agent
            try:
                response = await self._handle_user_message(text)
            except Exception as e:
                logger.error("Agent error: %s", e)
                response = "Sorry, something went wrong while processing your request."

            # Speak response, then immediately ready for follow-up
            logger.info("🤖 Jarvis: %s", response)
            self.wake_word.suppress()
            try:
                await self.tts.speak(response)
            finally:
                self.wake_word.unsuppress(cooldown=0.5)

            # Soft beep to signal "I'm still listening"
            await self._play_listening_beep()

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

        # Close Playwright browser if open
        await shutdown_browser()

        if self.wake_word:
            await self.wake_word.shutdown()
        if self.stt:
            await self.stt.shutdown()
        if self.tts:
            await self.tts.shutdown()
        if self.memory:
            await self.memory.shutdown()
        if self.llm:
            await self.llm.shutdown()

        logger.info("✅ Shutdown complete")
