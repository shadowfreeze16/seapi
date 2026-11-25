"""Pretrained model wrappers.

The project deliberately uses three small, well-tested models instead of a
single large end-to-end network:

1. YuNet (OpenCV) for face detection and 5 landmarks.  ~225 KB, very fast.
2. 6DRepNet MobileNetV3-small for yaw / pitch / roll.  ~6 MB, ~22 ms on a Pi 4.
3. OCEC-S for eye open/closed state.  ~0.5 MB, ~1 ms.

Model files are never bundled with the source code; use
``python tools/download_models.py`` to fetch them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - handled with a clear message
    raise RuntimeError(
        "onnxruntime is required. Install it inside the project venv with:\n"
        "    pip install onnxruntime"
    ) from exc

from .geometry import rotation_matrix_to_euler


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Face:
    """A detected face.

    ``box`` is (x, y, w, h) in pixels.  ``landmarks`` is (5, 2) with the YuNet
    order: image-left eye, image-right eye, nose tip, image-left mouth corner,
    image-right mouth corner.  ``detected`` is False when the tracker is
    holding the last box through a brief detection dropout.
    """

    box: np.ndarray
    landmarks: np.ndarray
    score: float = 1.0
    detected: bool = True
    age: int = 0

    @property
    def x(self) -> float:
        return float(self.box[0])

    @property
    def y(self) -> float:
        return float(self.box[1])

    @property
    def width(self) -> float:
        return float(self.box[2])

    @property
    def height(self) -> float:
        return float(self.box[3])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def clipped(self, frame_width: int, frame_height: int) -> "Face":
        x0 = max(0, min(frame_width - 1, self.x))
        y0 = max(0, min(frame_height - 1, self.y))
        x1 = max(x0 + 1, min(frame_width, self.x + self.width))
        y1 = max(y0 + 1, min(frame_height, self.y + self.height))
        landmarks = self.landmarks.copy()
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, frame_width - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, frame_height - 1)
        return Face(
            box=np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32),
            landmarks=landmarks.astype(np.float32),
            score=self.score,
            detected=self.detected,
            age=self.age,
        )


@dataclass
class HeadPose:
    """Head pose in degrees plus the raw rotation matrix."""

    yaw: float
    pitch: float
    roll: float
    rotation: np.ndarray
    latency_ms: float = 0.0
    crop_box: tuple[int, int, int, int] | None = None


@dataclass
class EyeObservation:
    """Eye-open probabilities from OCEC.

    ``left_open`` refers to the image-left eye (the driver's right eye when the
    camera is not mirrored).  ``right_open`` is the image-right eye.
    """

    left_open: float
    right_open: float
    valid: bool
    boxes: list[tuple[int, int, int, int]]
    latency_ms: float = 0.0

    @property
    def average_open(self) -> float:
        return 0.5 * (self.left_open + self.right_open)

    @property
    def both_closed(self) -> bool:
        return self.valid and self.left_open < 0.5 and self.right_open < 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_crop(
    frame: np.ndarray, center_x: float, center_y: float, size: int
) -> np.ndarray:
    """Crop a square around a point, padding with edge pixels if necessary."""
    h, w = frame.shape[:2]
    size = max(2, int(size))
    x0 = int(round(center_x - size / 2.0))
    y0 = int(round(center_y - size / 2.0))
    x1, y1 = x0 + size, y0 + size

    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x1 - w)
    bottom = max(0, y1 - h)

    crop_x0 = max(0, x0)
    crop_y0 = max(0, y0)
    crop_x1 = min(w, x1)
    crop_y1 = min(h, y1)
    crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
    if crop.size == 0:
        raise ValueError("Empty face crop")

    if left or top or right or bottom:
        crop = cv2.copyMakeBorder(
            crop,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REPLICATE,
        )
    return crop


def _softmax_if_needed(values: np.ndarray) -> np.ndarray:
    """OCEC exports a sigmoid probability; some exports may emit logits."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    if float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
        return values
    # Stable sigmoid.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Face detection + tracking
# ---------------------------------------------------------------------------


class FaceDetector:
    """Thin wrapper around OpenCV's YuNet FaceDetectorYN.

    ``input_scale`` runs the network on a downscaled copy of the frame and
    scales the boxes and landmarks back up.  Face detection dominates the
    runtime on a Raspberry Pi, and half resolution is roughly four times
    faster while still giving accurate landmarks for a typical webcam.
    """

    def __init__(
        self,
        model_path: Path | str,
        width: int,
        height: int,
        confidence: float = 0.60,
        nms: float = 0.30,
        top_k: int = 5000,
        input_scale: float = 1.0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face model not found: {self.model_path}\n"
                "Run: python tools/download_models.py"
            )
        self.width = int(width)
        self.height = int(height)
        self.input_scale = float(np.clip(input_scale, 0.1, 1.0))
        self._input_size = (
            max(64, int(round(self.width * self.input_scale))),
            max(64, int(round(self.height * self.input_scale))),
        )
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            self._input_size,
            float(confidence),
            float(nms),
            int(top_k),
        )

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        detect_w = max(64, int(round(w * self.input_scale)))
        detect_h = max(64, int(round(h * self.input_scale)))
        if (detect_w, detect_h) != self._input_size:
            self._input_size = (detect_w, detect_h)
            self.detector.setInputSize(self._input_size)

        if (detect_w, detect_h) == (w, h):
            work = frame
        else:
            work = cv2.resize(
                frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA
            )

        _, raw = self.detector.detect(work)
        if raw is None or len(raw) == 0:
            return []

        scale_x = w / float(detect_w)
        scale_y = h / float(detect_h)
        faces: list[Face] = []
        for row in raw:
            x, y, bw, bh = [float(v) for v in row[:4]]
            if bw < 2 or bh < 2:
                continue
            landmarks = np.asarray(
                row[4:14], dtype=np.float32
            ).reshape(5, 2)
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y
            score = float(row[14]) if len(row) > 14 else 1.0
            faces.append(
                Face(
                    box=np.array(
                        [x * scale_x, y * scale_y, bw * scale_x, bh * scale_y],
                        dtype=np.float32,
                    ),
                    landmarks=landmarks,
                    score=score,
                )
            )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


class FaceTracker:
    """Selects the driver (largest face) and smooths the box / landmarks.

    A short hold-through-dropout keeps the eye crops and head-pose model stable
    when YuNet misses one or two frames.  The state machine still receives the
    real detection age, so a prolonged loss is reported as NO_FACE.
    """

    def __init__(self, smoothing: float = 0.55, max_missing: int = 3) -> None:
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.max_missing = max(0, int(max_missing))
        self.face: Face | None = None
        self.missing = 0

    def update(self, faces: list[Face], frame_width: int, frame_height: int) -> Face | None:
        if faces:
            raw = max(faces, key=lambda f: f.area).clipped(frame_width, frame_height)
            if self.face is None:
                tracked = raw
            else:
                prev = self.face
                # If the new face is far away, reset instead of dragging the box.
                distance = np.linalg.norm(raw.box[:2] - prev.box[:2])
                if distance > max(80.0, 0.8 * max(prev.width, prev.height)):
                    tracked = raw
                else:
                    a = self.smoothing  # weight of previous value
                    box = a * prev.box + (1.0 - a) * raw.box
                    landmarks = a * prev.landmarks + (1.0 - a) * raw.landmarks
                    tracked = Face(
                        box=box.astype(np.float32),
                        landmarks=landmarks.astype(np.float32),
                        score=raw.score,
                        detected=True,
                    )
            self.face = tracked
            self.missing = 0
            return tracked

        if self.face is not None and self.missing < self.max_missing:
            self.missing += 1
            held = Face(
                box=self.face.box.copy(),
                landmarks=self.face.landmarks.copy(),
                score=self.face.score,
                detected=False,
                age=self.missing,
            )
            self.face = held
            return held

        self.face = None
        self.missing += 1
        return None

    def reset(self) -> None:
        self.face = None
        self.missing = 0


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------


class HeadPoseEstimator:
    """6DRepNet ONNX head-pose estimator."""

    def __init__(
        self,
        model_path: Path | str,
        threads: int = 2,
        expand: float = 0.30,
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Head-pose model not found: {self.model_path}\n"
                "Run: python tools/download_models.py"
            )
        self.expand = float(expand)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros(
            (1, 3, self.input_size[1], self.input_size[0]), dtype=np.float32
        )
        try:
            self.session.run(self.output_names, {self.input_name: dummy})
        except Exception:
            # A warmup failure should not prevent the app from starting.
            pass

    def estimate(self, frame: np.ndarray, face: Face) -> HeadPose | None:
        h, w = frame.shape[:2]
        side = max(face.width, face.height) * (1.0 + self.expand)
        side = max(side, 48.0)
        cx, cy = face.center
        try:
            crop = _square_crop(frame, cx, cy, int(round(side)))
        except ValueError:
            return None

        started = time.perf_counter()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = np.transpose(tensor, (2, 0, 1))[None].astype(np.float32)

        try:
            outputs = self.session.run(
                self.output_names, {self.input_name: tensor}
            )
        except Exception:
            return None
        if not outputs:
            return None
        rotation = np.asarray(outputs[0], dtype=np.float32).reshape(3, 3)
        pitch, yaw, roll = rotation_matrix_to_euler(rotation)
        latency = (time.perf_counter() - started) * 1000.0

        # Crop box in original-frame coordinates (may extend outside the frame).
        half = side / 2.0
        cx0 = int(round(cx - half))
        cy0 = int(round(cy - half))
        crop_box = (cx0, cy0, int(round(side)), int(round(side)))
        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation=rotation,
            latency_ms=latency,
            crop_box=crop_box,
        )


# ---------------------------------------------------------------------------
# Eye state
# ---------------------------------------------------------------------------


class EyeStateEstimator:
    """OCEC eye-open/closed classifier."""

    def __init__(
        self,
        model_path: Path | str,
        threads: int = 1,
        crop_scale: float = 0.34,
        crop_aspect: float = 0.62,
        min_crop_px: int = 10,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Eye model not found: {self.model_path}\n"
                "Run: python tools/download_models.py"
            )
        self.crop_scale = float(crop_scale)
        self.crop_aspect = float(crop_aspect)
        self.min_crop_px = max(6, int(min_crop_px))

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        # Expected (N, 3, 24, 40).  Fall back to 24x40 if dynamic.
        self.input_h = int(shape[2]) if isinstance(shape[2], int) else 24
        self.input_w = int(shape[3]) if isinstance(shape[3], int) else 40
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros((2, 3, self.input_h, self.input_w), dtype=np.float32)
        try:
            self.session.run(self.output_names, {self.input_name: dummy})
        except Exception:
            # Some exports have a fixed batch of 1; the real call handles that.
            try:
                self.session.run(
                    self.output_names,
                    {self.input_name: dummy[:1]},
                )
            except Exception:
                pass

    def _crop_eye(
        self,
        frame: np.ndarray,
        center: np.ndarray,
        face: Face,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        eye_w = int(round(max(self.min_crop_px, face.width * self.crop_scale)))
        eye_h = int(round(max(self.min_crop_px, eye_w * self.crop_aspect)))
        cx, cy = float(center[0]), float(center[1])
        x0 = int(round(cx - eye_w / 2.0))
        y0 = int(round(cy - eye_h / 2.0))
        x1, y1 = x0 + eye_w, y0 + eye_h

        # If the crop is mostly outside the frame, report it as invalid.
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(w, x1), min(h, y1)
        if ix1 - ix0 < self.min_crop_px or iy1 - iy0 < self.min_crop_px:
            return None, (ix0, iy0, max(0, ix1 - ix0), max(0, iy1 - iy0))

        crop = frame[iy0:iy1, ix0:ix1]
        if crop.size == 0:
            return None, (ix0, iy0, 0, 0)
        # Pad if the eye is near an edge, so the classifier always sees a full
        # eye-shaped crop.
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(
                crop,
                pad_t,
                pad_b,
                pad_l,
                pad_r,
                borderType=cv2.BORDER_REPLICATE,
            )
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return None, (ix0, iy0, crop.shape[1], crop.shape[0])
        return crop, (ix0, iy0, crop.shape[1], crop.shape[0])

    @staticmethod
    def _preprocess(crop: np.ndarray, w: int, h: int) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))

    def estimate(
        self,
        frame: np.ndarray,
        face: Face,
        yaw: float | None = None,
        max_yaw: float = 60.0,
    ) -> EyeObservation:
        empty = EyeObservation(
            left_open=0.0,
            right_open=0.0,
            valid=False,
            boxes=[],
        )
        if face.width < 20 or face.height < 20:
            return empty
        if yaw is not None and abs(yaw) > max_yaw:
            return empty

        centers = face.landmarks[:2]
        crops: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        for center in centers:
            crop, box = self._crop_eye(frame, center, face)
            boxes.append(box)
            if crop is None:
                return EyeObservation(
                    left_open=0.0,
                    right_open=0.0,
                    valid=False,
                    boxes=boxes,
                )
            if float(crop.mean()) < 12.0:
                return EyeObservation(
                    left_open=0.0,
                    right_open=0.0,
                    valid=False,
                    boxes=boxes,
                )
            crops.append(self._preprocess(crop, self.input_w, self.input_h))

        batch = np.stack(crops, axis=0).astype(np.float32)
        started = time.perf_counter()
        try:
            outputs = self.session.run(self.output_names, {self.input_name: batch})
        except Exception:
            try:
                outputs = [
                    self.session.run(
                        self.output_names,
                        {self.input_name: batch[i : i + 1]},
                    )[0]
                    for i in range(batch.shape[0])
                ]
                probs = np.concatenate(
                    [np.asarray(o, dtype=np.float32).reshape(-1) for o in outputs],
                    axis=0,
                )
            except Exception:
                return EyeObservation(
                    left_open=0.0,
                    right_open=0.0,
                    valid=False,
                    boxes=boxes,
                )
        else:
            probs = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        probs = _softmax_if_needed(probs)
        if probs.size < 2:
            return EyeObservation(
                left_open=0.0,
                right_open=0.0,
                valid=False,
                boxes=boxes,
            )
        latency = (time.perf_counter() - started) * 1000.0
        left_val = float(np.clip(probs[0], 0.0, 1.0))
        right_val = float(np.clip(probs[1], 0.0, 1.0))

        # Require both eyes to agree; if one eye is > 0.80 and the other < 0.10,
        # it is landmark drift or lighting shadow, so mark invalid (do not count as closed).
        if (left_val > 0.80 and right_val < 0.10) or (right_val > 0.80 and left_val < 0.10):
            return EyeObservation(
                left_open=left_val,
                right_open=right_val,
                valid=False,
                boxes=boxes,
                latency_ms=latency,
            )

        return EyeObservation(
            left_open=left_val,
            right_open=right_val,
            valid=True,
            boxes=boxes,
            latency_ms=latency,
        )