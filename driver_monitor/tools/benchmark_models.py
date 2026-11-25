#!/usr/bin/env python3
"""Benchmark the face, head-pose, eye and 3D-head stages on this machine.

    python tools/benchmark_models.py
    python tools/benchmark_models.py --image events/example.jpg --runs 30
    python tools/benchmark_models.py --json /tmp/bench.json

The script looks for a face in the supplied image (or scans the event photos
for one), then measures each model separately.  It is intentionally dependency
free apart from the project's own requirements.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from dms.config import DMSConfig  # noqa: E402
from dms.head3d import HeadRenderer  # noqa: E402
from dms.models import (  # noqa: E402
    EyeStateEstimator,
    FaceDetector,
    HeadPoseEstimator,
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return ordered[max(0, min(len(ordered) - 1, index))]


def _time_stage(fn, runs: int, warmup: int) -> tuple[list[float], object]:
    result = None
    for _ in range(warmup):
        result = fn()
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, result


def _head_benchmark(size: int):
    """Return a callable that renders a different pose on every call.

    The renderer caches identical poses, so benchmarking a constant angle
    would only measure a dictionary lookup.
    """
    renderer = HeadRenderer(size)
    counter = [0]

    def call():
        index = counter[0]
        counter[0] += 1
        yaw = float(index % 31) - 15.0
        pitch = float((index * 3) % 21) - 10.0
        roll = float((index * 5) % 13) - 6.0
        return renderer.render(
            yaw, pitch, roll, (1.0, 1.0), "NORMAL", "FORWARD"
        )

    return call


def _find_face_image(
    detector: FaceDetector, explicit: Path | None
) -> tuple[np.ndarray, np.ndarray, Path]:
    if explicit is not None:
        frame = cv2.imread(str(explicit))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {explicit}")
        faces = detector.detect(frame)
        if not faces:
            raise RuntimeError(f"No face detected in {explicit}")
        face = max(faces, key=lambda f: f.area)
        return frame, face.box, explicit

    candidates = sorted((PROJECT_ROOT / "events").glob("*.jpg"))
    for path in candidates:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        faces = detector.detect(frame)
        if faces:
            face = max(faces, key=lambda f: f.area)
            return frame, face.box, path
    raise RuntimeError(
        "No face found in events/*.jpg. Pass --image with a driver photo."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DMSConfig.model_dir)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    config = DMSConfig()
    config.model_dir = args.model_dir
    runs = max(1, args.runs)
    warmup = max(0, args.warmup)

    print(f"Benchmark: {runs} runs, {warmup} warmup frames")
    print(f"Models: {config.model_dir}")

    detector = FaceDetector(
        config.face_model_path,
        config.width,
        config.height,
        confidence=config.face_confidence,
        nms=config.face_nms,
        top_k=config.face_top_k,
        input_scale=config.face_input_scale,
    )
    pose_estimator = HeadPoseEstimator(
        config.headpose_model_path, threads=args.threads
    )
    eye_estimator = EyeStateEstimator(
        config.eye_model_path, threads=args.threads
    )

    frame, box, image_path = _find_face_image(detector, args.image)
    frame = cv2.resize(frame, (config.width, config.height))
    faces = detector.detect(frame)
    face = max(faces, key=lambda f: f.area) if faces else None
    if face is None:
        raise RuntimeError("Lost the face after resizing the image")

    print(f"Image: {image_path}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f"Face:  {face.width:.0f}x{face.height:.0f} px\n")

    results: dict[str, dict[str, float]] = {}

    def measure(name: str, fn, unit_note: str = "") -> object:
        samples, result = _time_stage(fn, runs, warmup)
        mean = sum(samples) / len(samples)
        results[name] = {
            "mean_ms": round(mean, 2),
            "p95_ms": round(_percentile(samples, 0.95), 2),
            "min_ms": round(min(samples), 2),
            "max_ms": round(max(samples), 2),
        }
        print(
            f"  {name:12s} mean {mean:6.2f} ms   "
            f"p95 {_percentile(samples, 0.95):6.2f} ms   "
            f"{1000.0 / max(mean, 1e-6):5.1f} fps  {unit_note}"
        )
        return result

    measure("face", lambda: detector.detect(frame), "YuNet")
    pose = measure(
        "headpose", lambda: pose_estimator.estimate(frame, face), "6DRepNet"
    )
    yaw = pose.yaw if pose is not None else 0.0
    measure(
        "eyes",
        lambda: eye_estimator.estimate(frame, face, yaw=yaw),
        "OCEC-S",
    )
    measure("head3d-420", _head_benchmark(420), "pure OpenCV")
    measure("head3d-320", _head_benchmark(320), "pure OpenCV")

    print()
    print("Model files")
    for name in (
        config.face_model,
        config.headpose_model,
        config.eye_model,
    ):
        path = config.model_dir / name
        if path.exists():
            print(f"  {name:42s} {path.stat().st_size / 1e6:6.2f} MB")
        else:
            print(f"  {name:42s} MISSING")

    payload = {
        "image": str(image_path),
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
        "runs": runs,
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")

    total = (
        results["face"]["mean_ms"]
        + results["headpose"]["mean_ms"]
        + results["eyes"]["mean_ms"]
        + results["head3d-320"]["mean_ms"]
    )
    print(
        f"\nCombined camera pipeline estimate: {total:.1f} ms/frame "
        f"(~{1000.0 / total:.1f} fps)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())