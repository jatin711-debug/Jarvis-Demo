"""Audio devices are chosen by name when asked, and fall back safely when absent."""

import logging

import pytest

from jarvis.utils.config import VoiceConfig
from jarvis.voice.devices import (
    default_hostapi,
    describe,
    resolve_device,
    resolve_voice_devices,
)


def dev(name, kind="input", hostapi=0):
    return {"name": name, "hostapi": hostapi,
            "max_input_channels": 2 if kind == "input" else 0,
            "max_output_channels": 2 if kind == "output" else 0}


# Mirrors a real Windows listing: the same device appears once per host API.
DEVICES = [
    dev("Microsoft Sound Mapper - Input"),                       # 0
    dev("Microphone Array (Realtek(R) Au"),                      # 1
    dev("Speakers (Realtek(R) Audio)", "output"),                # 2
    dev("JBL Tune 520BT Hands-Free", hostapi=0),                 # 3
    dev("JBL Tune 520BT Hands-Free", hostapi=2),                 # 4
    dev("JBL Tune 520BT Stereo", "output", hostapi=0),           # 5
]


class TestResolve:
    def test_none_means_system_default(self):
        assert resolve_device(None, "input", DEVICES) is None

    def test_a_valid_index_is_kept(self):
        assert resolve_device(1, "input", DEVICES) == 1

    def test_an_index_of_the_wrong_kind_falls_back_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="jarvis.voice.devices"):
            assert resolve_device(2, "input", DEVICES) is None
        assert "not an input device" in caplog.text

    def test_an_index_past_the_end_falls_back(self):
        assert resolve_device(99, "output", DEVICES) is None

    def test_a_name_survives_index_changes(self):
        shifted = [dev("Something new"), *DEVICES]
        assert resolve_device("jbl", "input", DEVICES) == 3
        assert resolve_device("jbl", "input", shifted) == 4

    def test_the_kind_is_respected_for_names(self):
        assert resolve_device("JBL", "output", DEVICES) == 5
        assert resolve_device("Realtek", "input", DEVICES) == 1

    def test_exact_names_beat_partial_matches(self):
        devices = [dev("USB Mic Pro"), dev("USB Mic")]
        assert resolve_device("usb mic", "input", devices) == 1

    def test_the_default_host_api_wins_then_the_lowest_index(self):
        assert resolve_device("JBL", "input", DEVICES, default_hostapi=2) == 4
        assert resolve_device("JBL", "input", DEVICES, default_hostapi=0) == 3
        assert resolve_device("JBL", "input", DEVICES, default_hostapi=None) == 3

    def test_a_missing_headset_falls_back_instead_of_picking_another_device(self, caplog):
        with caplog.at_level(logging.WARNING, logger="jarvis.voice.devices"):
            assert resolve_device("AirPods", "input", DEVICES) is None
        assert "matches no input device" in caplog.text

    def test_a_numeric_string_is_an_index(self):
        assert resolve_device("1", "input", DEVICES) == 1


def test_describe_lists_only_usable_devices_and_truncates():
    assert describe(DEVICES, "output") == "2=Speakers (Realtek(R) Audio), 5=JBL Tune 520BT Stereo"
    assert describe([], "input") == "none"
    many = [dev(f"mic{i}") for i in range(10)]
    assert describe(many, "input").endswith(", ...")


class TestConfig:
    @pytest.mark.parametrize("raw,expected", [(2, 2), ("JBL", "JBL"), (None, None)])
    def test_voice_devices_accept_an_index_or_a_name(self, raw, expected):
        cfg = VoiceConfig(input_device=raw, output_device=raw)
        assert cfg.input_device == expected and cfg.output_device == expected


class FakeSounddevice:
    def __init__(self, devices, default_input=0, fail=None):
        self._devices = devices
        self.default = type("Defaults", (), {"device": (default_input, 0)})()
        self._fail = fail

    def query_devices(self):
        if self._fail:
            raise self._fail
        return self._devices


class TestResolveVoiceDevices:
    def test_names_become_indexes_for_both_directions(self):
        voice = VoiceConfig(input_device="jbl", output_device="JBL Tune 520BT Stereo")
        resolved = resolve_voice_devices(voice, FakeSounddevice(DEVICES))
        assert (resolved.input_device, resolved.output_device) == (3, 5)
        assert voice.input_device == "jbl", "the configuration itself is not modified"

    def test_other_voice_settings_are_untouched(self):
        voice = VoiceConfig(input_device="jbl", wake_word="computer", sensitivity=0.7)
        resolved = resolve_voice_devices(voice, FakeSounddevice(DEVICES))
        assert (resolved.wake_word, resolved.sensitivity) == ("computer", 0.7)

    def test_the_default_hostapi_breaks_ties(self):
        voice = VoiceConfig(input_device="JBL")
        assert resolve_voice_devices(voice, FakeSounddevice(DEVICES, default_input=4)).input_device == 4

    def test_a_disconnected_headset_falls_back_to_the_default_device(self):
        resolved = resolve_voice_devices(VoiceConfig(input_device="AirPods"), FakeSounddevice(DEVICES))
        assert resolved.input_device is None

    def test_nothing_is_queried_when_nothing_is_configured(self):
        voice = VoiceConfig()
        assert resolve_voice_devices(voice, FakeSounddevice([], fail=OSError("never called"))) is voice

    def test_a_name_cannot_be_resolved_without_a_device_list(self):
        with pytest.raises(RuntimeError, match="by name"):
            resolve_voice_devices(VoiceConfig(input_device="JBL"),
                                  FakeSounddevice([], fail=OSError("PortAudio not initialized")))

    def test_an_index_is_kept_when_the_device_list_is_unavailable(self):
        voice = VoiceConfig(input_device=2)
        resolved = resolve_voice_devices(voice, FakeSounddevice([], fail=OSError("no portaudio")))
        assert resolved is voice


def test_default_hostapi_ignores_a_missing_or_negative_default():
    assert default_hostapi(FakeSounddevice(DEVICES, default_input=3), DEVICES) == 0
    assert default_hostapi(FakeSounddevice(DEVICES, default_input=4), DEVICES) == 2
    assert default_hostapi(FakeSounddevice(DEVICES, default_input=-1), DEVICES) is None
    assert default_hostapi(object(), DEVICES) is None
