"""Voice engines, recording helpers and playback against fake audio hardware.

Nothing here touches a real device or model: ``pyaudio``, ``sounddevice``,
``openwakeword``, ``soxr`` and ``piper`` are replaced through ``sys.modules`` so the
suite also runs on a CI image that only has numpy and soundfile.
"""

import asyncio
import io
import logging
import sys
import threading
import types
from contextlib import aclosing

import numpy as np
import pytest
import soundfile as sf

import jarvis.voice.implementations as impl
from jarvis.utils.config import JarvisConfig
from jarvis.voice import cues

VOICE_LOGGER = "jarvis.voice.implementations"


def chunk(level: int, frames: int = 1024) -> np.ndarray:
    """A block whose RMS is exactly ``level`` (constant sample value)."""
    return np.full(frames, level, dtype=np.int16)


def wav_samples(wav: bytes) -> np.ndarray:
    samples, _ = sf.read(io.BytesIO(wav), dtype="int16")
    return samples


@pytest.fixture(autouse=True)
def fresh_rate_cache():
    impl._MIC_RATE_CACHE.clear()
    yield
    impl._MIC_RATE_CACHE.clear()


# ── Fake microphone (pyaudio) ───────────────────────────────


class FakeStream:
    def __init__(self, mic, kwargs):
        self.mic = mic
        self.kwargs = kwargs
        self.closed = False
        self.stopped = False
        self.in_read = False
        self.closed_while_reading = False

    def read(self, frames, exception_on_overflow=False):
        self.mic.note("read")
        item = self.mic.next_item()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item()
        return item.tobytes()

    def stop_stream(self):
        self.mic.note("stop_stream")
        self.stopped = True

    def close(self):
        self.mic.note("close")
        self.closed_while_reading = self.in_read
        self.closed = True


class FakePyAudio:
    def __init__(self, mic):
        self.mic = mic
        self.terminated = False
        mic.note("PyAudio")
        mic.instances.append(self)

    def get_default_input_device_info(self):
        self.mic.note("get_default_input_device_info")
        self.mic.rate_queries += 1
        return {"index": 0, "name": "Fake Mic", "defaultSampleRate": float(self.mic.rate)}

    def get_device_info_by_index(self, index):
        self.mic.note("get_device_info_by_index")
        return {"index": index, "name": f"Fake Mic {index}", "maxInputChannels": 1,
                "defaultSampleRate": float(self.mic.rate)}

    def get_device_count(self):
        self.mic.note("get_device_count")
        return 1

    def open(self, **kwargs):
        self.mic.note("open")
        if self.mic.open_errors:
            raise self.mic.open_errors.pop(0)
        stream = FakeStream(self.mic, kwargs)
        self.mic.streams.append(stream)
        return stream

    def terminate(self):
        self.mic.note("terminate")
        self.terminated = True


class FakeMic:
    """A scripted microphone: items are int16 blocks, exceptions to raise, or callables."""

    def __init__(self, script=(), rate=16000):
        self.script = list(script)
        self.rate = rate
        self.open_errors: list[BaseException] = []
        self.instances: list[FakePyAudio] = []
        self.streams: list[FakeStream] = []
        self.threads: dict[str, set[int]] = {}
        self.rate_queries = 0

    def note(self, call):
        self.threads.setdefault(call, set()).add(threading.get_ident())

    def next_item(self):
        if not self.script:
            raise AssertionError("mic script exhausted: the code read more than the test scripted")
        return self.script.pop(0)

    def install(self, monkeypatch):
        module = types.ModuleType("pyaudio")
        module.paInt16 = 8
        module.PyAudio = lambda: FakePyAudio(self)
        monkeypatch.setitem(sys.modules, "pyaudio", module)
        return self

    @property
    def stream(self):
        return self.streams[-1]

    def assert_released(self):
        assert self.streams and all(s.closed for s in self.streams)
        assert self.instances and all(pa.terminated for pa in self.instances)

    def assert_off_the_event_loop(self):
        assert self.threads, "no hardware call happened"
        for call, idents in self.threads.items():
            assert threading.get_ident() not in idents, f"{call} ran on the event loop thread"


@pytest.fixture
def mic(monkeypatch):
    return FakeMic().install(monkeypatch)


def fake_soxr(monkeypatch, resample):
    module = types.ModuleType("soxr")
    module.resample = resample
    monkeypatch.setitem(sys.modules, "soxr", module)


# ── Fake speaker (sounddevice) ──────────────────────────────


class FakeOutputStream:
    def __init__(self, speaker, kwargs):
        self.speaker = speaker
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.writes: list[np.ndarray] = []

    def start(self):
        self.started = True

    def write(self, data):
        if self.stopped:
            self.speaker.rejected_writes += 1
            raise self.speaker.error("Stream is stopped")
        if self.speaker.fail_on(self):
            raise self.speaker.error("device lost")
        self.writes.append(np.array(data))
        if self.speaker.after_write:
            self.speaker.after_write(self)

    def abort(self):
        self.stopped = True

    def close(self):
        self.closed = True

    @property
    def samples(self) -> int:
        return sum(len(w) for w in self.writes)


class FakeSpeaker:
    def __init__(self):
        self.streams: list[FakeOutputStream] = []
        self.fail_on = lambda stream: False
        self.after_write = None
        self.rejected_writes = 0  # writes the device refused because the stream was stopped

        class PortAudioError(Exception):
            pass

        self.error = PortAudioError

    def install(self, monkeypatch):
        module = types.ModuleType("sounddevice")
        module.PortAudioError = self.error
        module.OutputStream = self._open
        monkeypatch.setitem(sys.modules, "sounddevice", module)
        return self

    def _open(self, **kwargs):
        stream = FakeOutputStream(self, kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def speaker(monkeypatch):
    return FakeSpeaker().install(monkeypatch)


class FakeClock:
    """Replaces the player's clock and blocking wait, so drain() costs no real time."""

    def __init__(self):
        self.now = 1000.0
        self.naps: list[float] = []

    def __call__(self):
        return self.now

    def nap(self, seconds):
        self.naps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(impl, "_clock", fake)
    monkeypatch.setattr(impl, "_nap", fake.nap)
    return fake


@pytest.fixture
def sleeps(monkeypatch):
    waited: list[float] = []

    async def fake_sleep(seconds):
        waited.append(seconds)

    monkeypatch.setattr(impl, "_sleep", fake_sleep)
    return waited


def seconds_of(sample_rate: int, seconds: float) -> np.ndarray:
    return np.zeros(int(sample_rate * seconds), dtype=np.float32)


# ── Resampling ──────────────────────────────────────────────


class TestResample:
    def test_same_rate_is_untouched(self):
        audio = np.arange(5, dtype=np.int16)
        assert impl._resample(audio, 16000, 16000) is audio

    def test_overshoot_is_clipped_not_wrapped(self, monkeypatch):
        fake_soxr(monkeypatch, lambda audio, a, b: np.array([40000.0, -40000.0, 12.7], np.float32))
        out = impl._resample(np.zeros(3, np.int16), 8000, 16000)
        assert out.dtype == np.int16
        assert out.tolist() == [32767, -32768, 12]

    def test_scipy_fallback_is_clipped_too(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "soxr", None)  # import soxr -> ImportError
        signal = types.ModuleType("scipy.signal")
        signal.resample_poly = lambda audio, up, down: np.array([1e6, -1e6, 3.0])
        scipy = types.ModuleType("scipy")
        scipy.signal = signal
        monkeypatch.setitem(sys.modules, "scipy", scipy)
        monkeypatch.setitem(sys.modules, "scipy.signal", signal)
        out = impl._resample(np.zeros(3, np.int16), 44100, 16000)
        assert out.tolist() == [32767, -32768, 3]


# ── Native rate lookup ──────────────────────────────────────


class TestNativeRate:
    def test_queried_once_then_cached(self, mic):
        assert impl._get_mic_native_rate(None) == 16000
        assert impl._get_mic_native_rate(None) == 16000
        assert mic.rate_queries == 1

    def test_refresh_asks_the_device_again(self, mic):
        impl._get_mic_native_rate(None)
        mic.rate = 48000
        assert impl._get_mic_native_rate(None, refresh=True) == 48000
        assert impl._get_mic_native_rate(None) == 48000
        assert mic.rate_queries == 2

    def test_devices_are_cached_separately(self, mic):
        impl._get_mic_native_rate(None)
        impl._get_mic_native_rate(3)
        assert 3 in impl._MIC_RATE_CACHE and None in impl._MIC_RATE_CACHE

    async def test_async_lookup_runs_off_the_loop_and_only_once(self, mic):
        assert await impl._native_rate(None) == 16000
        assert await impl._native_rate(None) == 16000
        assert mic.rate_queries == 1
        mic.assert_off_the_event_loop()


# ── Gapless player ──────────────────────────────────────────


class TestGaplessPlayer:
    def test_audio_is_written_in_short_slices(self, speaker):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.3), 16000)
        (stream,) = speaker.streams
        assert [len(w) for w in stream.writes] == [1280, 1280, 1280, 960]
        assert stream.started and stream.kwargs["latency"] == "high"

    def test_play_after_abort_reopens_the_stream(self, speaker):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.3), 16000)
        first = speaker.streams[0]

        player.abort()
        assert first.stopped and not first.closed  # abort stops; it does not close

        player.reset_abort()
        player.play(seconds_of(16000, 0.3), 16000)  # a write to the stopped stream would raise

        assert len(speaker.streams) == 2
        assert first.closed
        assert speaker.streams[1].samples == 4800
        assert speaker.rejected_writes == 0  # recovered by design, not by retrying a failed write

    def test_abort_between_slices_cuts_a_long_chunk_short(self, speaker):
        player = impl._GaplessPlayer()
        speaker.after_write = lambda s: player.abort() if len(s.writes) == 5 else None
        player.play(seconds_of(16000, 3.0), 16000)  # 38 slices if nothing stopped it
        assert len(speaker.streams[0].writes) == 5

    def test_play_while_aborted_writes_nothing(self, speaker):
        player = impl._GaplessPlayer()
        player.abort()
        player.play(seconds_of(16000, 0.3), 16000)
        assert not speaker.streams or speaker.streams[0].samples == 0

    def test_a_write_cut_short_by_abort_is_not_a_device_fault(self, speaker, caplog):
        player = impl._GaplessPlayer()
        speaker.fail_on = lambda s: (player.abort(), True)[1]  # abort lands mid-write
        with caplog.at_level(logging.WARNING, logger=VOICE_LOGGER):
            player.play(seconds_of(16000, 0.3), 16000)  # returns quietly
        assert len(speaker.streams) == 1
        assert not caplog.records

    def test_rate_change_reopens_the_stream(self, speaker):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.1), 16000)
        player.play(seconds_of(16000, 0.1), 16000)
        assert len(speaker.streams) == 1
        player.play(seconds_of(24000, 0.1), 24000)
        assert len(speaker.streams) == 2
        assert speaker.streams[0].closed
        assert speaker.streams[1].kwargs["samplerate"] == 24000

    def test_failed_write_reopens_once_and_retries(self, speaker, caplog):
        player = impl._GaplessPlayer()
        speaker.fail_on = lambda s: s is speaker.streams[0]  # the first stream is dead
        with caplog.at_level(logging.WARNING, logger=VOICE_LOGGER):
            player.play(seconds_of(16000, 0.3), 16000)
        assert len(speaker.streams) == 2
        assert speaker.streams[0].closed and speaker.streams[0].samples == 0
        assert speaker.streams[1].samples == 4800  # nothing was lost
        assert any("reopening" in r.getMessage() for r in caplog.records)

    def test_second_failure_raises_and_the_next_play_recovers(self, speaker):
        player = impl._GaplessPlayer()
        speaker.fail_on = lambda s: True
        with pytest.raises(speaker.error):
            player.play(seconds_of(16000, 0.3), 16000)
        assert len(speaker.streams) == 2  # one reopen, no more

        speaker.fail_on = lambda s: False
        player.play(seconds_of(16000, 0.3), 16000)
        assert speaker.streams[-1].samples == 4800

    def test_other_errors_are_not_swallowed(self, speaker):
        player = impl._GaplessPlayer()
        speaker.fail_on = lambda s: (_ for _ in ()).throw(ValueError("bad shape"))
        with pytest.raises(ValueError):
            player.play(seconds_of(16000, 0.1), 16000)
        assert len(speaker.streams) == 1

    def test_close_is_idempotent_and_play_reopens(self, speaker):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.1), 16000)
        player.close()
        player.close()
        assert speaker.streams[0].closed
        player.play(seconds_of(16000, 0.1), 16000)
        assert len(speaker.streams) == 2

    def test_device_is_passed_to_the_stream(self, speaker):
        impl._GaplessPlayer(device=7).play(seconds_of(16000, 0.1), 16000)
        assert speaker.streams[0].kwargs["device"] == 7


class TestDrain:
    def test_waits_for_the_audio_that_is_still_queued(self, speaker, clock):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 1.0), 16000)
        player.drain()
        assert clock.now == pytest.approx(1001.0, abs=0.021)

    def test_queued_chunks_add_up(self, speaker, clock):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.5), 16000)
        player.play(seconds_of(16000, 0.5), 16000)
        player.drain()
        assert clock.now == pytest.approx(1001.0, abs=0.021)

    def test_audio_that_already_played_costs_nothing(self, speaker, clock):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.5), 16000)
        clock.now += 2.0  # the user kept talking; playback finished long ago
        player.drain()
        assert clock.naps == []

    def test_a_gap_before_the_next_chunk_is_not_counted_as_queued(self, speaker, clock):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 0.5), 16000)
        clock.now += 3.0  # synthesis was slow: the device went idle
        player.play(seconds_of(16000, 0.5), 16000)
        player.drain()
        assert clock.now == pytest.approx(1003.5, abs=0.021)  # waited ~0.5s, not 3.5s

    def test_the_wait_is_bounded(self, speaker, clock):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 30.0), 16000)
        player.drain()
        assert clock.now - 1000.0 == pytest.approx(impl._DRAIN_MAX_SECONDS, abs=0.021)
        clock.now = 1000.0
        player.drain(timeout=0.5)
        assert clock.now == pytest.approx(1000.5, abs=0.021)

    def test_abort_ends_the_wait(self, speaker, clock, monkeypatch):
        player = impl._GaplessPlayer()
        player.play(seconds_of(16000, 5.0), 16000)

        def nap(seconds):
            clock.nap(seconds)
            if len(clock.naps) == 3:
                player.abort()

        monkeypatch.setattr(impl, "_nap", nap)
        player.drain()
        assert len(clock.naps) == 3

    def test_nothing_to_wait_for_without_a_stream(self, clock):
        impl._GaplessPlayer().drain()
        assert clock.naps == []


# ── TTS engines on top of the player ────────────────────────


class TestKokoroPlayback:
    async def test_next_sentence_plays_after_a_barge_in(self, speaker, clock):
        tts = impl.KokoroTTS(JarvisConfig().voice)
        tts._pipeline = lambda text, **kw: iter([("g", "p", seconds_of(24000, 2.0))])
        speaker.after_write = lambda s: tts.abort() if len(s.writes) == 3 else None

        await tts.speak("first sentence")
        assert len(speaker.streams[0].writes) == 3  # 25 slices were available

        speaker.after_write = None
        await tts.speak("second sentence")
        assert len(speaker.streams) == 2
        assert speaker.streams[1].samples == 48000

        await tts.shutdown()
        assert all(s.closed for s in speaker.streams)


class FakePiperVoice:
    def __init__(self):
        self.calls = []
        self.config = types.SimpleNamespace(sample_rate=22050)

    def synthesize(self, text, **kwargs):
        self.calls.append(kwargs)
        pcm = (np.ones(2205, dtype=np.int16) * 1000).tobytes()
        return [types.SimpleNamespace(audio_int16_bytes=pcm)]


def fake_piper_config(monkeypatch):
    """A piper.config with SynthesisConfig, without the real package."""
    class SynthesisConfig:
        def __init__(self, length_scale=None):
            self.length_scale = length_scale

    config_module = types.ModuleType("piper.config")
    config_module.SynthesisConfig = SynthesisConfig
    package = types.ModuleType("piper")
    package.config = config_module
    monkeypatch.setitem(sys.modules, "piper", package)
    monkeypatch.setitem(sys.modules, "piper.config", config_module)


class TestPiperSpeed:
    def make(self, speed):
        config = JarvisConfig().voice
        config.tts.speed = speed
        tts = impl.PiperTTS(config)
        tts._voice = FakePiperVoice()
        return tts

    def test_speed_becomes_a_length_scale(self, monkeypatch):
        fake_piper_config(monkeypatch)
        options = self.make(1.25)._speed_options()
        assert options["syn_config"].length_scale == pytest.approx(0.8)

    def test_normal_speed_passes_nothing(self, monkeypatch):
        fake_piper_config(monkeypatch)
        assert self.make(1.0)._speed_options() == {}

    def test_old_piper_without_synthesis_config_is_survivable(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "piper.config", None)
        monkeypatch.setitem(sys.modules, "piper", types.ModuleType("piper"))
        with caplog.at_level(logging.WARNING, logger=VOICE_LOGGER):
            assert self.make(1.5)._speed_options() == {}
        assert any("speed ignored" in r.getMessage() for r in caplog.records)

    async def test_speak_and_synthesize_apply_the_option(self, monkeypatch, speaker, clock):
        fake_piper_config(monkeypatch)
        tts = self.make(2.0)
        tts._synth_kwargs = tts._speed_options()
        await tts.speak("hello")
        assert await tts.synthesize("hello")
        assert [c["syn_config"].length_scale for c in tts._voice.calls] == [0.5, 0.5]
        assert speaker.streams[0].samples == 2205

    async def test_default_speed_calls_synthesize_plainly(self, speaker, clock):
        tts = self.make(1.0)
        await tts.speak("hello")
        assert tts._voice.calls == [{}]


# ── Microphone stream wrapper ───────────────────────────────


class TestMicInput:
    async def test_open_failure_leaves_no_pyaudio_behind(self, mic):
        mic.open_errors = [OSError("no such device")]
        stream = impl._MicInput(None, 16000, 1024)
        with pytest.raises(OSError, match="no such device"):
            await stream.open()
        await stream.close()
        mic.assert_off_the_event_loop()
        assert mic.instances and all(pa.terminated for pa in mic.instances)

    async def test_close_waits_for_a_read_that_is_still_running(self, mic):
        started, gate = threading.Event(), threading.Event()

        def slow_read():
            mic.stream.in_read = True
            started.set()
            gate.wait(5)
            mic.stream.in_read = False
            return chunk(10)

        mic.script = [slow_read]
        stream = impl._MicInput(None, 16000, 1024)
        await stream.open()
        reader = asyncio.create_task(stream.read())
        await asyncio.to_thread(started.wait, 5)
        reader.cancel()  # the caller goes away while the read is blocked in its thread
        with pytest.raises(asyncio.CancelledError):
            await reader

        closing = asyncio.create_task(stream.close())
        await asyncio.sleep(0.05)
        assert not mic.stream.closed  # must not close under a running read
        gate.set()
        await closing
        assert mic.stream.closed and not mic.stream.closed_while_reading
        assert mic.instances[0].terminated

    async def test_close_gives_up_on_a_hung_read(self, mic, monkeypatch, caplog):
        monkeypatch.setattr(impl, "_SETTLE_SECONDS", 0.05)
        gate = threading.Event()
        mic.script = [lambda: (gate.wait(5), chunk(1))[1]]
        stream = impl._MicInput(None, 16000, 1024)
        await stream.open()
        reader = asyncio.create_task(stream.read())
        await asyncio.sleep(0.02)
        reader.cancel()
        with caplog.at_level(logging.WARNING, logger=VOICE_LOGGER):
            await stream.close()
        assert mic.stream.closed
        assert any("still blocked" in r.getMessage() for r in caplog.records)
        gate.set()

    async def test_close_twice_is_harmless(self, mic):
        stream = impl._MicInput(None, 16000, 1024)
        await stream.open()
        await stream.close()
        await stream.close()
        mic.assert_released()


# ── record_until_silence ────────────────────────────────────


class TestRecordUntilSilence:
    async def test_stops_after_silence_following_speech(self, mic):
        mic.script = [chunk(0)] * 10 + [chunk(1000)] * 5 + [chunk(0)] * 40
        wav = await impl.record_until_silence(silence_threshold=100.0)
        # 10 leading silent chunks never end it; 5 speech + 23 silent (1.5s) do.
        assert len(wav_samples(wav)) == (10 + 5 + 23) * 1024
        assert len(mic.script) == 40 - 23
        mic.assert_released()

    async def test_leading_silence_alone_does_not_stop_it(self, mic):
        mic.script = [chunk(0)] * 100
        wav = await impl.record_until_silence(silence_threshold=100.0, max_duration=1.0)
        assert len(wav_samples(wav)) == 15 * 1024  # max_duration, never "silence after speech"
        mic.assert_released()

    async def test_max_duration_bounds_the_recording_including_calibration(self, mic):
        mic.script = [chunk(5)] * 100
        wav = await impl.record_until_silence(max_duration=0.5)
        assert len(wav_samples(wav)) == 7 * 1024  # 4 calibration frames count toward the 7
        mic.assert_released()

    async def test_quiet_start_calibrates_low_and_keeps_every_frame(self, mic):
        mic.script = [chunk(10)] * 4 + [chunk(2000)] * 10 + [chunk(5)] * 40
        wav = await impl.record_until_silence()
        samples = wav_samples(wav)
        assert len(samples) == (4 + 10 + 23) * 1024
        assert samples[:1024].tolist() == chunk(10).tolist()  # the calibration frames are kept
        assert samples[4 * 1024] == 2000

    async def test_speech_from_the_first_chunk_is_detected_and_kept(self, mic):
        # After a cue or confirmation the user is already talking when recording starts.
        mic.script = [chunk(2000)] * 12 + [chunk(5)] * 40
        wav = await impl.record_until_silence()
        samples = wav_samples(wav)
        assert len(samples) == (12 + 23) * 1024  # ends on the silence, not at max_duration
        assert samples[0] == 2000  # the first words are not thrown away
        mic.assert_released()

    async def test_threshold_never_falls_below_the_floor(self, mic):
        # Dead-quiet calibration: threshold is the 30 floor, so level 40 counts as speech.
        mic.script = [chunk(0)] * 4 + [chunk(40)] * 3 + [chunk(0)] * 40
        wav = await impl.record_until_silence()
        assert len(wav_samples(wav)) == (4 + 3 + 23) * 1024

    async def test_a_single_dropout_frame_does_not_collapse_the_threshold(self, mic):
        # A digital-zero first buffer in a noisy room must not set threshold to the floor.
        mic.script = [chunk(0), chunk(300), chunk(300), chunk(300)] + [chunk(2000)] * 4
        mic.script += [chunk(250)] * 40  # room noise, below the calibrated threshold
        wav = await impl.record_until_silence()
        assert len(wav_samples(wav)) == (4 + 4 + 23) * 1024

    def test_calibration_is_bounded(self):
        assert impl._calibrated_threshold([0.0, 0.0, 0.0, 0.0])[0] == impl._THRESHOLD_FLOOR
        assert impl._calibrated_threshold([5000.0] * 4)[0] == impl._THRESHOLD_CEILING
        threshold, ambient = impl._calibrated_threshold([40.0, 40.0, 40.0, 900.0])
        assert ambient == pytest.approx(40.0, abs=5)
        assert threshold == pytest.approx(120.0, abs=15)

    async def test_stream_is_released_when_a_read_fails(self, mic):
        mic.script = [chunk(0), OSError("headset unplugged")]
        with pytest.raises(OSError, match="headset"):
            await impl.record_until_silence(silence_threshold=100.0)
        mic.assert_released()

    async def test_stream_is_released_when_calibration_fails(self, mic):
        mic.script = [chunk(0), OSError("headset unplugged")]
        with pytest.raises(OSError, match="headset"):
            await impl.record_until_silence()  # calibrates first
        mic.assert_released()

    async def test_nothing_leaks_when_the_device_will_not_open(self, mic):
        mic.open_errors = [OSError("device busy")]
        with pytest.raises(OSError, match="busy"):
            await impl.record_until_silence()
        assert not mic.streams
        assert mic.instances and all(pa.terminated for pa in mic.instances)

    async def test_open_failure_forgets_the_cached_rate(self, mic):
        await impl._native_rate(None)
        mic.open_errors = [OSError("device busy")]
        with pytest.raises(OSError):
            await impl.record_until_silence()
        assert None not in impl._MIC_RATE_CACHE  # the device may have changed

    async def test_blocking_audio_calls_stay_off_the_event_loop(self, mic):
        mic.script = [chunk(2000)] * 5 + [chunk(0)] * 40
        await impl.record_until_silence(silence_threshold=100.0)
        mic.assert_off_the_event_loop()

    async def test_rate_is_looked_up_once_across_recordings(self, mic):
        mic.script = [chunk(2000)] * 5 + [chunk(0)] * 30 + [chunk(2000)] * 5 + [chunk(0)] * 30
        await impl.record_until_silence(silence_threshold=100.0)
        await impl.record_until_silence(silence_threshold=100.0)
        assert mic.rate_queries == 1

    async def test_native_rate_audio_is_resampled_to_the_target(self, mic, monkeypatch):
        mic.rate = 32000
        fake_soxr(monkeypatch, lambda audio, a, b: audio[::2])  # 32 kHz -> 16 kHz
        mic.script = [chunk(2000, 2048)] * 5 + [chunk(0, 2048)] * 40
        wav = await impl.record_until_silence(silence_threshold=100.0)
        assert mic.stream.kwargs["rate"] == 32000
        assert mic.stream.kwargs["frames_per_buffer"] == 2048
        assert len(wav_samples(wav)) == (5 + 23) * 1024


# ── wait_for_speech (barge-in monitor) ──────────────────────


class TestWaitForSpeech:
    async def test_sustained_loud_audio_is_a_barge_in(self, mic):
        mic.script = [chunk(50)] * 7 + [chunk(2000)] * 5
        assert await impl.wait_for_speech() is True
        mic.assert_released()

    async def test_a_short_burst_is_not(self, mic):
        cancel = asyncio.Event()
        mic.script = ([chunk(50)] * 7 + [chunk(2000)] * 4 + [chunk(50)] * 3
                      + [lambda: (cancel.set(), chunk(50))[1]])
        assert await impl.wait_for_speech(cancel_event=cancel) is False
        mic.assert_released()

    async def test_cancellation_before_the_first_read_still_releases_the_mic(self, mic):
        cancel = asyncio.Event()
        cancel.set()
        assert await impl.wait_for_speech(cancel_event=cancel) is False
        mic.assert_released()

    async def test_max_duration_ends_the_monitor(self, mic):
        mic.script = [chunk(50)] * 10
        assert await impl.wait_for_speech(max_duration=0.2) is False  # 7 calibration + 3
        assert not mic.script
        mic.assert_released()

    async def test_stream_is_released_when_a_read_fails(self, mic):
        mic.script = [chunk(50), OSError("headset unplugged")]
        with pytest.raises(OSError):
            await impl.wait_for_speech()
        mic.assert_released()

    async def test_nothing_leaks_when_the_device_will_not_open(self, mic):
        mic.open_errors = [OSError("device busy")]
        with pytest.raises(OSError):
            await impl.wait_for_speech()
        assert not mic.streams
        assert all(pa.terminated for pa in mic.instances)
        assert None not in impl._MIC_RATE_CACHE

    async def test_each_sentence_reuses_the_cached_rate_and_stays_off_the_loop(self, mic):
        for _ in range(3):
            cancel = asyncio.Event()
            cancel.set()
            await impl.wait_for_speech(cancel_event=cancel)
        assert mic.rate_queries == 1
        mic.assert_off_the_event_loop()


# ── Wake word engine ────────────────────────────────────────


class FakeWakeModel:
    def __init__(self, scores, wake_word="hey_jarvis"):
        self.scores = list(scores)
        self.wake_word = wake_word
        self.resets = 0

    def predict(self, audio):
        return {self.wake_word: self.scores.pop(0)}

    def reset(self):
        self.resets += 1


def make_engine(scores=()):
    engine = impl.OpenWakeWordEngine(JarvisConfig().voice)
    engine._native_sr = 16000
    engine._model = FakeWakeModel(scores, engine.wake_word)
    return engine


async def collect_until_detection(engine, limit=50):
    events = []
    async with aclosing(engine.listen()) as stream:
        async for detected in stream:
            events.append(detected)
            if detected or len(events) >= limit:
                break
    return events


class TestWakeWordListen:
    async def test_detection_yields_true(self, mic, sleeps):
        mic.script = [chunk(10)] * 3
        engine = make_engine([0.0, 0.1, 0.9])
        assert await collect_until_detection(engine) == [False, False, True]
        assert sleeps == []
        mic.assert_released()
        assert engine._mic is None

    async def test_frame_size_follows_the_native_rate(self, mic, sleeps):
        mic.script = [chunk(10, 1280)]
        await collect_until_detection(make_engine([0.9]))
        assert mic.stream.kwargs["frames_per_buffer"] == 1280

    async def test_a_read_error_reopens_the_mic_instead_of_ending_the_loop(
            self, mic, sleeps, caplog):
        mic.script = [chunk(10), OSError("device unplugged"), chunk(10), chunk(10)]
        engine = make_engine([0.0, 0.0, 0.9])
        with caplog.at_level(logging.DEBUG, logger=VOICE_LOGGER):
            events = await collect_until_detection(engine)

        assert events == [False, False, False, True]  # the failed attempt yields False
        assert sleeps == [1.0]
        assert len(mic.streams) == 2
        assert mic.streams[0].closed  # the dead stream was released
        mic.assert_released()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "mic read failed" in warnings[0].getMessage()
        assert "device unplugged" in warnings[0].getMessage()
        assert any("mic recovered" in r.getMessage() for r in caplog.records)

    async def test_reopen_asks_the_device_for_its_rate_again(self, mic, sleeps, monkeypatch):
        fake_soxr(monkeypatch, lambda audio, a, b: audio[:1280])
        mic.script = [OSError("device unplugged"), chunk(10, 1920)]
        mic.rate = 24000  # a different headset came back
        engine = make_engine([0.9])
        await collect_until_detection(engine)
        assert engine._native_sr == 24000
        assert mic.stream.kwargs["frames_per_buffer"] == 1920

    async def test_open_failures_back_off_and_warn_once(self, mic, sleeps, caplog):
        mic.open_errors = [OSError("busy")] * 3
        mic.script = [chunk(10), chunk(10)]
        engine = make_engine([0.0, 0.9])
        with caplog.at_level(logging.DEBUG, logger=VOICE_LOGGER):
            events = await collect_until_detection(engine)

        assert events == [False, False, False, False, True]
        assert sleeps == [1.0, 2.0, 4.0]  # doubling
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # identical repeats drop to debug
        assert sum("failed again" in r.getMessage() for r in caplog.records) == 2
        mic.assert_released()

    async def test_delay_is_capped(self, mic, sleeps):
        mic.open_errors = [OSError("busy")] * 6
        mic.script = [chunk(10)]
        await collect_until_detection(make_engine([0.9]))
        assert sleeps == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]

    async def test_delay_resets_after_a_healthy_read(self, mic, sleeps):
        mic.open_errors = [OSError("busy")] * 2
        mic.script = [chunk(10), OSError("unplugged"), chunk(10)]
        await collect_until_detection(make_engine([0.0, 0.9]))
        assert sleeps == [1.0, 2.0, 1.0]

    async def test_suppressed_audio_is_drained_not_scored(self, mic, sleeps):
        mic.script = [chunk(10)] * 3
        engine = make_engine([0.9])  # a second predict() would raise IndexError
        engine._suppressed = True
        events = []
        async with aclosing(engine.listen()) as stream:
            async for detected in stream:
                events.append(detected)
                if len(events) == 2:
                    engine._suppressed = False
                if detected:
                    break
        assert events == [False, False, True]
        assert not mic.script

    async def test_startup_keeps_blocking_audio_calls_off_the_event_loop(
            self, mic, monkeypatch, caplog):
        mic.script = [chunk(0)] * 7  # a silent mic: also exercises the device listing
        created = []
        model_module = types.ModuleType("openwakeword.model")
        model_module.Model = lambda **kw: created.append(kw) or FakeWakeModel([])
        monkeypatch.setitem(sys.modules, "openwakeword", types.ModuleType("openwakeword"))
        monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

        engine = impl.OpenWakeWordEngine(JarvisConfig().voice)
        with caplog.at_level(logging.INFO, logger=VOICE_LOGGER):
            await engine.startup()

        assert engine._native_sr == 16000
        assert created and created[0]["inference_framework"] == "onnx"
        assert any("mic appears silent" in r.getMessage() for r in caplog.records)
        assert "get_device_count" in mic.threads
        mic.assert_off_the_event_loop()
        mic.assert_released()


class TestValidateMic:
    def test_loud_mic_passes_and_is_released(self, mic):
        mic.script = [chunk(500)] * 7
        assert impl._validate_mic(None, 16000) is True
        mic.assert_released()

    def test_a_failing_read_is_reported_and_still_releases(self, mic, caplog):
        mic.script = [chunk(500), OSError("gone")]
        with caplog.at_level(logging.WARNING, logger=VOICE_LOGGER):
            assert impl._validate_mic(None, 16000) is False
        assert any("mic validation failed" in r.getMessage() for r in caplog.records)
        mic.assert_released()


# ── Audio cues ──────────────────────────────────────────────


class TestCues:
    @pytest.fixture
    def played(self, monkeypatch):
        calls = []
        module = types.ModuleType("sounddevice")
        module.play = lambda data, samplerate, **kw: calls.append((data, samplerate, kw))
        module.wait = lambda: None
        monkeypatch.setitem(sys.modules, "sounddevice", module)
        return calls

    async def test_cues_use_the_configured_output_device(self, played):
        await cues.play_wake_cue(4)
        await cues.play_listening_cue(5)
        assert [kw for _, _, kw in played] == [{"device": 4}, {"device": 5}]

    async def test_default_device_is_none(self, played):
        await cues.play_wake_cue()
        assert played[0][2] == {"device": None}
        assert played[0][0].dtype == np.float32 and len(played[0][0]) > 0

    async def test_audio_failure_is_ignored(self, monkeypatch):
        module = types.ModuleType("sounddevice")

        def boom(*args, **kwargs):
            raise RuntimeError("no device")

        module.play = boom
        module.wait = lambda: None
        monkeypatch.setitem(sys.modules, "sounddevice", module)
        await cues.play_listening_cue(3)  # cosmetic: must not raise
