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
from jarvis.utils.logger import get_logger, setup_logging
from jarvis.utils.config import expand_path

if TYPE_CHECKING:
    from jarvis.utils.config import JarvisConfig
    from jarvis.voice.engines import STTEngine, TTSEngine, WakeWordEngine

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
        self.wake_word: WakeWordEngine | None = None

        # Conversation state
        self._conversation_history: list[dict[str, str]] = []

        # Echo cancellation flag — mute wake word during TTS
        self._speaking = False

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

    async def run(self) -> None:
        """Main voice loop: wake word → record → STT → agent → TTS."""
        from jarvis.voice.implementations import record_until_silence

        assert self.wake_word is not None
        assert self.stt is not None
        assert self.tts is not None

        logger.info("🎙️  Say '%s' to begin\n", self.config.voice.wake_word)

        async for detected in self.wake_word.listen():
            if not self._running:
                break

            # Echo cancel: ignore wake word while Jarvis is speaking
            if not detected or self._speaking:
                continue

            # Wake word heard — beep to confirm
            await self._play_beep()
            logger.info("🎤 Listening for command...")

            # Record user speech
            try:
                audio_bytes = await record_until_silence(
                    sample_rate=16000,
                    input_device=self.config.voice.input_device,
                )
            except Exception as e:
                logger.error("Recording failed: %s", e)
                self._speaking = True
                await self.tts.speak("Sorry, I couldn't hear you. Please try again.")
                self._speaking = False
                continue

            # Transcribe
            text = await self.stt.transcribe(audio_bytes)
            text = text.strip()
            if not text:
                logger.info("(empty transcription, ignoring)")
                continue

            logger.info("🧑 You: %s", text)

            # Exit commands
            if text.lower() in ("exit", "quit", "bye", "goodbye", "shut down", "stop"):
                self._speaking = True
                await self.tts.speak("Goodbye!")
                self._speaking = False
                self._running = False
                break

            # Process through agent
            try:
                response = await self._handle_user_message(text)
            except Exception as e:
                logger.error("Agent error: %s", e)
                response = "Sorry, something went wrong while processing your request."

            # Speak the response (with echo cancellation)
            logger.info("🤖 Jarvis: %s", response)
            self._speaking = True
            await self.tts.speak(response)
            self._speaking = False

    async def shutdown(self) -> None:
        """Gracefully tear down all services."""
        self._running = False
        logger.info("🔄 Shutting down ...")

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
