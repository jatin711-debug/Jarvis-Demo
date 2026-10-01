"""Pick audio devices by name instead of by a fragile index.

Device indexes change whenever a headset (dis)connects or a driver reloads, so
``voice.input_device: 2`` can silently become a different microphone. A name
such as ``"JBL"`` keeps meaning the same device, and when it is absent Jarvis
falls back to the system default instead of opening the wrong one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any, Literal

from jarvis.utils.config import VoiceConfig
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

Kind = Literal["input", "output"]
DeviceSpec = int | str | None


def _channels(device: dict[str, Any], kind: Kind) -> int:
    return int(device.get(f"max_{kind}_channels", 0) or 0)


def resolve_device(spec: DeviceSpec, kind: Kind, devices: Sequence[dict[str, Any]],
                   default_hostapi: int | None = None) -> int | None:
    """Return the device index to use, or None for the system default.

    ``spec`` is an index, a name (case-insensitive substring, exact names win),
    or None. A spec that matches no usable device logs a warning and resolves
    to None. When one name exists under several host APIs (MME, WASAPI, ...),
    the default host API wins, then the lowest index.
    """
    if spec is None:
        return None
    if isinstance(spec, str) and spec.strip().isdecimal():
        spec = int(spec.strip())

    usable = [(i, d) for i, d in enumerate(devices) if _channels(d, kind) > 0]
    if isinstance(spec, int):
        if any(i == spec for i, _ in usable):
            return spec
        logger.warning("voice.%s_device index=%d is not an %s device; using the system default",
                       kind, spec, kind)
        return None

    wanted = spec.strip().casefold()
    exact = [(i, d) for i, d in usable if str(d.get("name", "")).casefold() == wanted]
    partial = [(i, d) for i, d in usable if wanted in str(d.get("name", "")).casefold()]
    matches = exact or partial
    if not matches:
        logger.warning("voice.%s_device name=%r matches no %s device; using the system default",
                       kind, spec, kind)
        return None
    matches.sort(key=lambda item: (item[1].get("hostapi") != default_hostapi, item[0]))
    index, device = matches[0]
    logger.info("voice.%s_device name=%r -> index=%d (%s)", kind, spec, index, device.get("name"))
    return index


def describe(devices: Sequence[dict[str, Any]], kind: Kind, limit: int = 6,
             hostapi: int | None = None) -> str:
    """A short "index=name" list of the usable devices, for messages.

    ``hostapi`` limits the list to one host API, since Windows lists the same
    device once per API (MME, DirectSound, WASAPI, ...).
    """
    usable = [f"{i}={dev.get('name')}" for i, dev in enumerate(devices)
              if _channels(dev, kind) > 0 and (hostapi is None or dev.get("hostapi") == hostapi)]
    if not usable:
        return "none"
    return ", ".join(usable[:limit]) + (", ..." if len(usable) > limit else "")


def default_hostapi(sd: Any, devices: Sequence[dict[str, Any]]) -> int | None:
    """The host API of the system default input device (None when it cannot be told)."""
    with contextlib.suppress(Exception):
        index = int(sd.default.device[0])
        if 0 <= index < len(devices):
            return devices[index].get("hostapi")
    return None


def _is_name(spec: DeviceSpec) -> bool:
    return isinstance(spec, str) and not spec.strip().isdecimal()


def resolve_voice_devices(voice: VoiceConfig, sd: Any = None) -> VoiceConfig:
    """``voice`` with device names turned into indexes (blocking; run it in a thread).

    Names are matched against the devices present right now; a name that matches
    nothing becomes the system default, with a warning. When the devices cannot be
    listed, numeric settings are left as they are, but a name cannot be resolved
    and is reported as a startup problem.
    """
    if voice.input_device is None and voice.output_device is None:
        return voice
    try:
        if sd is None:
            import sounddevice as sd
        devices = [dict(device) for device in sd.query_devices()]
    except Exception as exc:
        if _is_name(voice.input_device) or _is_name(voice.output_device):
            raise RuntimeError(
                f"cannot list audio devices to find voice.input_device/output_device by name "
                f"({type(exc).__name__}: {exc}); use a device index or check the audio drivers"
            ) from exc
        logger.warning("audio devices could not be listed; using the configured indexes (%s)", exc)
        return voice
    hostapi = default_hostapi(sd, devices)
    return voice.model_copy(update={
        "input_device": resolve_device(voice.input_device, "input", devices, hostapi),
        "output_device": resolve_device(voice.output_device, "output", devices, hostapi),
    })
