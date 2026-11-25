#!/usr/bin/env python3
"""Download and verify the pretrained ONNX models.

    python tools/download_models.py
    python tools/download_models.py --check
    python tools/download_models.py --extra
    python tools/download_models.py --force

Model files are not bundled with the source tree.  Every download is checked
against a pinned SHA-256 before it is moved into ``models/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "models"

MODELS: dict[str, dict[str, object]] = {
    "face_detection_yunet_2026may.onnx": {
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2026may.onnx"
        ),
        "sha256": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        "size": 229738,
        "label": "YuNet face detector (5 landmarks)",
    },
    "headpose_mobilenetv3_small.onnx": {
        "url": (
            "https://github.com/yakhyo/head-pose-estimation/releases/"
            "download/weights/mobilenetv3_small.onnx"
        ),
        "sha256": "e8ae4d932b3d13221638fc72e171603e020c6da28b770753f76146867f40e190",
        "size": 6111775,
        "label": "6DRepNet MobileNetV3-small head pose",
    },
    "eyes_ocec_s.onnx": {
        "url": (
            "https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_s.onnx"
        ),
        "sha256": "9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64",
        "size": 494914,
        "label": "OCEC-S eye open/closed classifier",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size": 38696353,
        "label": "SFace face-recognition embedding (driver identity)",
    },
}

EXTRA: dict[str, dict[str, object]] = {
    "headpose_mobilenetv2.onnx": {
        "url": (
            "https://github.com/yakhyo/head-pose-estimation/releases/"
            "download/weights/mobilenetv2.onnx"
        ),
        "sha256": "1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9",
        "size": 8905103,
        "label": "6DRepNet MobileNetV2 (slower reference model)",
    },
    "eyes_ocec_m.onnx": {
        "url": (
            "https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_m.onnx"
        ),
        "sha256": "810cf02f9c5a933fde688285aaf4548be0355092bcacbf3ee256ccfd01df6f6e",
        "size": 1693332,
        "label": "OCEC-M eye classifier (larger reference model)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, spec: dict[str, object]) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    expected = str(spec["sha256"])
    actual = sha256(path)
    if actual != expected:
        return False, f"sha256 mismatch ({actual[:12]}... != {expected[:12]}...)"
    return True, "ok"


def download(name: str, spec: dict[str, object], target_dir: Path, force: bool) -> bool:
    target = target_dir / name
    if target.exists() and not force:
        ok, message = verify(target, spec)
        if ok:
            print(f"  [ok]      {name} ({target.stat().st_size / 1e6:.1f} MB)")
            return True
        print(f"  [bad]     {name}: {message}")
        print("            re-download with --force")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [get]     {name}")
    print(f"            {spec['label']}")
    url = str(spec["url"])
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_dir, prefix=name + ".", suffix=".part", delete=False
        ) as tmp:
            tmp_name = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                done = 0
                last_tenth = -1
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    done += len(chunk)
                    if total:
                        tenth = int(done * 10 / total)
                        if tenth != last_tenth:
                            last_tenth = tenth
                            print(
                                f"            {done / 1e6:5.1f} / "
                                f"{total / 1e6:5.1f} MB",
                                end="\r",
                                flush=True,
                            )
                print()
        assert tmp_name is not None
        actual = sha256(tmp_name)
        expected = str(spec["sha256"])
        if actual != expected:
            print(f"  [FAIL]    sha256 mismatch for {name}")
            print(f"            expected {expected}")
            print(f"            actual   {actual}")
            tmp_name.unlink(missing_ok=True)
            return False
        os.replace(tmp_name, target)
        print(f"  [saved]   {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  [FAIL]    {name}: {exc}")
        if tmp_name is not None:
            tmp_name.unlink(missing_ok=True)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"destination directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify existing files, do not download",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="also download the optional MobileNetV2 / OCEC-M models",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download existing files"
    )
    args = parser.parse_args(argv)

    target_dir = args.model_dir
    wanted = dict(MODELS)
    if args.extra:
        wanted.update(EXTRA)

    print(f"Model directory: {target_dir}")
    if args.check:
        failed = 0
        for name, spec in wanted.items():
            ok, message = verify(target_dir / name, spec)
            print(f"  [{'ok' if ok else 'FAIL'}] {name}: {message}")
            failed += 0 if ok else 1
        return 0 if failed == 0 else 1

    results = [
        download(name, spec, target_dir, args.force)
        for name, spec in wanted.items()
    ]
    if not all(results):
        print("\nSome models could not be downloaded. Check your network and retry.")
        return 1
    print("\nAll models are present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())