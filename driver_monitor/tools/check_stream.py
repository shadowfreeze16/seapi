#!/usr/bin/env python3
"""Open a network camera stream, read a few frames, and save a preview.

Examples
--------
    python tools/check_stream.py http://192.168.1.50:4747/video
    python tools/check_stream.py rtsp://192.168.1.60:554/stream --frames 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dms.camera import NetworkSource, is_stream_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="HTTP(S) MJPEG, RTSP, RTMP or UDP stream URL")
    parser.add_argument("--frames", type=int, default=10, help="frames to read")
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("/tmp/dms_stream_preview.jpg"),
        help="where to save the last frame",
    )
    args = parser.parse_args(argv)

    if not is_stream_url(args.url):
        print(
            f"[ERROR] Not a network stream URL: {args.url}\n"
            "Use http://, https://, rtsp://, rtmp://, tcp:// or udp://.",
            file=sys.stderr,
        )
        return 2

    try:
        source = NetworkSource(args.url)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    print(f"Stream : {source.info.description}")
    frames = 0
    first = last = None
    image = None
    try:
        for _ in range(max(1, args.frames)):
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                time.sleep(0.2)
                continue
            if first is None:
                first = timestamp
            last = timestamp
            frames += 1
            image = frame
    finally:
        source.release()

    if image is None:
        print("[ERROR] The stream opened but delivered no frames.", file=sys.stderr)
        return 4

    elapsed = (last - first) if first is not None and last is not None else 0.0
    rate = (frames - 1) / elapsed if elapsed > 0 else 0.0
    height, width = image.shape[:2]
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.preview), image)

    print(f"Frames : {frames}/{max(1, args.frames)}")
    print(f"Size   : {width} x {height}")
    if rate > 0:
        print(f"Rate   : {rate:.1f} fps")
    print(f"Preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())