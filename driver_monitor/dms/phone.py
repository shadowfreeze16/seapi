"""DroidCam phone-camera setup helpers.

The Android DroidCam app exposes its camera as an MJPEG stream at
``http://PHONE_IP:PORT/video``.  This module prints the phone-side setup
steps, asks for the address, remembers it between runs, and builds the URL
used by :class:`dms.camera.NetworkSource`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .config import PROJECT_ROOT


DEFAULT_PHONE_IP = "10.107.190.123"
DEFAULT_PHONE_PORT = 4747
DEFAULT_PHONE_CONFIG = PROJECT_ROOT / "phone_camera.json"


@dataclass
class PhoneSettings:
    ip: str
    port: int


def normalize_phone_ip(value: str) -> str:
    """Accept an IPv4/IPv6 address, hostname, or pasted stream URL."""
    text = str(value).strip()
    if not text:
        raise ValueError("the IP address cannot be empty")
    if "://" in text:
        try:
            host = urlsplit(text).hostname
        except ValueError as exc:
            raise ValueError("the address is not a valid IP or hostname") from exc
        if not host:
            raise ValueError("the address is not a valid IP or hostname")
        return host
    text = text.split("/", 1)[0].strip()
    if not text:
        raise ValueError("the address is not a valid IP or hostname")
    # Allow "IP:port" even though the port is asked separately.
    if text.count(":") == 1:
        host, maybe_port = text.rsplit(":", 1)
        if maybe_port.isdigit():
            text = host
    return text


def validate_phone_port(value: int | str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("the port must be a number between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("the port must be between 1 and 65535")
    return port


def phone_stream_url(ip: str, port: int) -> str:
    host = normalize_phone_ip(ip)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{validate_phone_port(port)}/video"


def load_phone_settings(
    path: Path = DEFAULT_PHONE_CONFIG,
) -> PhoneSettings:
    ip = DEFAULT_PHONE_IP
    port = DEFAULT_PHONE_PORT
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return PhoneSettings(ip=ip, port=port)
    if isinstance(data, dict):
        if data.get("ip"):
            try:
                ip = normalize_phone_ip(data["ip"])
            except ValueError:
                pass
        if data.get("port") is not None:
            try:
                port = validate_phone_port(data["port"])
            except ValueError:
                pass
    return PhoneSettings(ip=ip, port=port)


def save_phone_settings(
    settings: PhoneSettings, path: Path = DEFAULT_PHONE_CONFIG
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"ip": settings.ip, "port": settings.port},
            indent=2,
        )
        + "\n"
    )


def show_phone_guide(
    settings: PhoneSettings, output: Callable[[str], None] = print
) -> None:
    output("")
    output("=" * 58)
    output("  PHONE CAMERA SETUP - DROIDCAM (ANDROID)")
    output("=" * 58)
    output("1. Install DroidCam on the phone:")
    output("     https://www.dev47apps.com   (Classic)")
    output("     https://droidcam.app       (new app)")
    output("2. Connect the phone and this Pi to the same Wi-Fi network.")
    output("3. Open DroidCam, grant Camera and Notifications permissions,")
    output("   and keep the live preview visible. Disable battery")
    output("   optimisation for DroidCam so it keeps streaming.")
    output("4. Note the Wi-Fi IP and port shown in the app (default 4747).")
    output("5. Enter the values below. Press Enter to keep the value in")
    output("   [brackets]. The address is saved in phone_camera.json.")
    output("")
    output(f"Stream URL: {phone_stream_url(settings.ip, settings.port)}")
    output("Only one client can use the phone stream at a time.")
    output("")


def _prompt_ip(
    default: str,
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
) -> str:
    while True:
        try:
            raw = input_fn(f"Phone IP [{default}]: ")
        except EOFError as exc:
            raise RuntimeError(
                "Phone setup needs a terminal. "
                "Use --phone-ip and --phone-port for non-interactive runs."
            ) from exc
        try:
            return normalize_phone_ip(raw or default)
        except ValueError as exc:
            output(f"  Invalid address: {exc}")


def _prompt_port(
    default: int,
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
) -> int:
    while True:
        try:
            raw = input_fn(f"Port [{default}]: ")
        except EOFError as exc:
            raise RuntimeError(
                "Phone setup needs a terminal. "
                "Use --phone-ip and --phone-port for non-interactive runs."
            ) from exc
        try:
            return validate_phone_port(raw or default)
        except ValueError as exc:
            output(f"  Invalid port: {exc}")


def setup_phone_camera(
    ip: str | None = None,
    port: int | None = None,
    config_path: Path = DEFAULT_PHONE_CONFIG,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> PhoneSettings:
    """Show the DroidCam guide, collect the address, and remember it."""
    defaults = load_phone_settings(config_path)
    chosen_ip = normalize_phone_ip(ip) if ip else None
    chosen_port = validate_phone_port(port) if port is not None else None

    show_phone_guide(
        PhoneSettings(chosen_ip or defaults.ip, chosen_port or defaults.port),
        output,
    )

    if chosen_ip is None:
        chosen_ip = _prompt_ip(defaults.ip, input_fn, output)
    if chosen_port is None:
        chosen_port = _prompt_port(defaults.port, input_fn, output)

    settings = PhoneSettings(ip=chosen_ip, port=chosen_port)
    try:
        save_phone_settings(settings, config_path)
    except OSError as exc:
        output(f"  Warning: could not save phone_camera.json: {exc}")
    output(f"Connecting to {phone_stream_url(settings.ip, settings.port)}")
    output("")
    return settings


__all__ = [
    "DEFAULT_PHONE_CONFIG",
    "DEFAULT_PHONE_IP",
    "DEFAULT_PHONE_PORT",
    "PhoneSettings",
    "load_phone_settings",
    "normalize_phone_ip",
    "phone_stream_url",
    "save_phone_settings",
    "setup_phone_camera",
    "show_phone_guide",
    "validate_phone_port",
]