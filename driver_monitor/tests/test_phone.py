"""DroidCam phone setup, address validation and persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dms.phone import (
    DEFAULT_PHONE_PORT,
    PhoneSettings,
    load_phone_settings,
    normalize_phone_ip,
    phone_stream_url,
    save_phone_settings,
    setup_phone_camera,
    validate_phone_port,
)


class PhoneSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "phone_camera.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_missing_config_uses_defaults(self) -> None:
        settings = load_phone_settings(self.path)
        self.assertEqual(settings.port, DEFAULT_PHONE_PORT)
        self.assertTrue(settings.ip)

    def test_round_trip(self) -> None:
        expected = PhoneSettings(ip="192.168.1.77", port=5050)
        save_phone_settings(expected, self.path)
        self.assertEqual(load_phone_settings(self.path), expected)

    def test_corrupt_config_uses_defaults(self) -> None:
        self.path.write_text("not json")
        settings = load_phone_settings(self.path)
        self.assertEqual(settings.port, DEFAULT_PHONE_PORT)
        self.assertTrue(settings.ip)

    def test_normalize_addresses(self) -> None:
        self.assertEqual(normalize_phone_ip(" 10.0.0.5 "), "10.0.0.5")
        self.assertEqual(normalize_phone_ip("10.0.0.5:4747"), "10.0.0.5")
        self.assertEqual(
            normalize_phone_ip("http://10.0.0.5:4747/video"), "10.0.0.5"
        )
        self.assertEqual(normalize_phone_ip("camera.local"), "camera.local")
        for bad in ("", "http://"):
            with self.assertRaises(ValueError):
                normalize_phone_ip(bad)

    def test_stream_url(self) -> None:
        self.assertEqual(
            phone_stream_url("10.0.0.5", 4747),
            "http://10.0.0.5:4747/video",
        )
        self.assertEqual(
            phone_stream_url("::1", 4747),
            "http://[::1]:4747/video",
        )

    def test_port_validation(self) -> None:
        self.assertEqual(validate_phone_port("4747"), 4747)
        for bad in ("abc", 0, 65536, None):
            with self.assertRaises(ValueError):
                validate_phone_port(bad)

    def test_prompt_defaults_and_saves(self) -> None:
        defaults = load_phone_settings(self.path)
        answers = iter(["", ""])
        output: list[str] = []
        settings = setup_phone_camera(
            config_path=self.path,
            input_fn=lambda _: next(answers),
            output=output.append,
        )
        self.assertEqual(settings, defaults)
        self.assertEqual(load_phone_settings(self.path), defaults)
        text = "\n".join(output)
        self.assertIn("PHONE CAMERA SETUP", text)
        self.assertIn("dev47apps.com", text)

    def test_prompt_retries_invalid_values(self) -> None:
        answers = iter(["http://", "10.0.0.9", "abc", "70000", "5050"])
        output: list[str] = []
        settings = setup_phone_camera(
            config_path=self.path,
            input_fn=lambda _: next(answers),
            output=output.append,
        )
        self.assertEqual(settings, PhoneSettings("10.0.0.9", 5050))
        text = "\n".join(output)
        self.assertIn("Invalid address", text)
        self.assertIn("Invalid port", text)

    def test_noninteractive_arguments_skip_prompts(self) -> None:
        def fail(_prompt: str) -> str:
            raise AssertionError("input() should not be called")

        output: list[str] = []
        settings = setup_phone_camera(
            ip="10.1.2.3",
            port=5000,
            config_path=self.path,
            input_fn=fail,
            output=output.append,
        )
        self.assertEqual(settings, PhoneSettings("10.1.2.3", 5000))
        self.assertEqual(load_phone_settings(self.path), settings)
        self.assertIn("Connecting to http://10.1.2.3:5000/video", output)

    def test_invalid_noninteractive_port(self) -> None:
        with self.assertRaises(ValueError):
            setup_phone_camera(
                ip="10.1.2.3",
                port=70000,
                config_path=self.path,
                output=lambda _: None,
            )


if __name__ == "__main__":
    unittest.main()