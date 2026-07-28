"""Test Kokoro TTS engine."""
import asyncio
import os
import sys

sys.path.insert(0, ".")

import sounddevice as sd

from jarvis.utils.config import load_config


async def test_tts():
    config = load_config()

    print("1. Initializing Kokoro TTS...")

    try:
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError:
        print("❌ Kokoro or soundfile not installed.")
        return

    # Kokoro uses early-binding Pipelines.
    # English language code is 'a' (American) or 'b' (British).
    try:
        pipeline = KPipeline(lang_code='a')
        print("✅ Pipeline initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return

    text = "Hello there! I am Jarvis. My voice generation system is now fully operational."
    voice = config.voice.tts.voice  # default: af_heart
    print(f"2. Synthesizing speech with voice '{voice}'...")

    try:
        # returns a generator of (graphemes, phonemes, audio)
        generator = pipeline(
            text, voice=voice, speed=config.voice.tts.speed, split_pattern=r'\n+'
        )

        for i, (gs, ps, audio) in enumerate(generator):
            print(f"  Played segment {i+1}: {gs}")
            # audio is a numpy array of shape (N,)
            # Kokoro base sample rate is 24000
            samplerate = 24000
            sd.play(audio, samplerate)
            sd.wait()

            # Save to file
            os.makedirs(".jarvis_test", exist_ok=True)
            out_file = f".jarvis_test/tts_test_{i}.wav"
            sf.write(out_file, audio, samplerate)
            print(f"  ✅ Saved to {out_file}")

    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return

    print("\n✅ TTS test complete!")

if __name__ == "__main__":
    asyncio.run(test_tts())
