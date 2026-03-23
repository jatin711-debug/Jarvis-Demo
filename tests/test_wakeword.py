"""Test OpenWakeWord engine with ONNX inference."""
import asyncio
import sys
sys.path.insert(0, ".")

from jarvis.utils.config import load_config


async def test_wakeword():
    config = load_config()
    target_ww = config.voice.wake_word  # default: 'hey_jarvis'
    
    print(f"1. Initializing OpenWakeWord for '{target_ww}'...")
    
    try:
        from openwakeword.model import Model
        import pyaudio
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return
        
    try:
        # Use ONNX inference — works natively on Windows without tflite-runtime
        owwModel = Model(
            wakeword_models=[target_ww],
            inference_framework="onnx"
        )
        print(f"✅ OpenWakeWord model loaded successfully (ONNX).")
        print(f"   Loaded models: {list(owwModel.models.keys())}")
    except Exception as e:
        print(f"❌ Failed to load OpenWakeWord model: {e}")
        return

    print("\n2. Checking PyAudio device availability...")
    try:
        audio = pyaudio.PyAudio()
        info = audio.get_default_input_device_info()
        print(f"  Default input device: {info.get('name')}")
        print(f"  Sample rate: {int(info.get('defaultSampleRate', 0))} Hz")
        print(f"  Input channels: {info.get('maxInputChannels', 0)}")
        audio.terminate()
        print("✅ PyAudio initialized — microphone detected.")
    except Exception as e:
        print(f"  ⚠️ PyAudio issue (expected if no mic): {e}")

    # Quick functional test: feed silence and verify no false-positive
    print("\n3. Running false-positive test (feeding 1s of silence)...")
    import numpy as np
    silence = np.zeros(16000, dtype=np.int16)
    scores = owwModel.predict(silence)
    max_score = max(scores.values()) if scores else 0
    print(f"  Scores on silence: {scores}")
    print(f"  Max score: {max_score:.4f} (should be ~0)")
    if max_score < 0.1:
        print("✅ No false positive on silence.")
    else:
        print("⚠️ Unexpected high score on silence!")

    print("\n✅ Wake word test complete! System is ready to listen.")

if __name__ == "__main__":
    asyncio.run(test_wakeword())
