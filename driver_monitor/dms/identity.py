"""Driver identity verification with the OpenCV SFace ONNX model.

SFace is a small face-recognition network.  This module keeps the model
handling, profile storage, matching and the three-check confirmation logic in
one place so the rest of the application only sees an ``IdentityResult``.

Only embeddings are stored, never enrolment images.  The profile file is
written with mode ``0600`` because a face embedding is biometric data.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - same message as models.py
    raise RuntimeError(
        "onnxruntime is required. Install it inside the project venv with:\n"
        "    pip install onnxruntime"
    ) from exc

from .config import DMSConfig
from .models import Face, HeadPose
from .eyes import EyeMetrics

# Identity states.
OFF = "OFF"
NO_FACE = "NO_FACE"
CHECKING = "CHECKING"
MATCH = "MATCH"
MISMATCH = "MISMATCH"
UNKNOWN = "UNKNOWN"

STABLE_STATES = {MATCH, MISMATCH}

# ArcFace/SFace 112x112 reference landmarks in YuNet order:
# image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
SFACE_REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

SFACE_INPUT_SIZE = (112, 112)
SFACE_MEAN = (127.5, 127.5, 127.5)
SFACE_SCALE = 1.0 / 128.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_embedding(values: np.ndarray) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros_like(vec)
    return (vec / norm).astype(np.float32)


@dataclass
class IdentityProfile:
    name: str
    embedding: np.ndarray
    created: str = ""
    model: str = ""
    samples: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "embedding": [round(float(v), 6) for v in self.embedding],
            "created": self.created,
            "model": self.model,
            "samples": int(self.samples),
        }


@dataclass
class IdentityResult:
    state: str = OFF
    name: str | None = None
    score: float | None = None
    best_name: str | None = None
    best_score: float | None = None
    checks: int = 0
    message: str = "IDENTITY OFF"
    valid: bool = False
    enforced: bool = False

    @property
    def mismatch(self) -> bool:
        return self.enforced and self.state == MISMATCH


class IdentityStore:
    """JSON-backed embedding store."""

    def __init__(self, path: Path | str, model: str = "") -> None:
        self.path = Path(path)
        self.model = model
        self.profiles: list[IdentityProfile] = []
        self.load()

    def load(self) -> list[IdentityProfile]:
        self.profiles = []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self.profiles
        if not isinstance(payload, dict):
            return self.profiles
        drivers = payload.get("drivers", [])
        if not isinstance(drivers, list):
            return self.profiles
        for entry in drivers:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            try:
                embedding = normalize_embedding(
                    np.asarray(entry.get("embedding", []), dtype=np.float32)
                )
            except (TypeError, ValueError):
                continue
            if not name or embedding.size == 0 or not np.all(
                np.isfinite(embedding)
            ):
                continue
            self.profiles.append(
                IdentityProfile(
                    name=name,
                    embedding=embedding,
                    created=str(entry.get("created", "")),
                    model=str(entry.get("model", "")),
                    samples=int(entry.get("samples", 0) or 0),
                )
            )
        return self.profiles

    def names(self) -> list[str]:
        return [profile.name for profile in self.profiles]

    def get(self, name: str) -> IdentityProfile | None:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        return None

    def save_profile(
        self,
        name: str,
        embedding: np.ndarray,
        samples: int = 0,
        replace: bool = False,
    ) -> IdentityProfile:
        name = str(name).strip()
        if not name:
            raise ValueError("the driver name cannot be empty")
        if self.get(name) is not None and not replace:
            raise ValueError(
                f"driver '{name}' is already enrolled; use --enroll-force"
            )
        profile = IdentityProfile(
            name=name,
            embedding=normalize_embedding(embedding),
            created=datetime.now().isoformat(timespec="seconds"),
            model=self.model,
            samples=int(samples),
        )
        if profile.embedding.size == 0:
            raise ValueError("the embedding is empty")
        self.profiles = [
            existing for existing in self.profiles if existing.name != name
        ]
        self.profiles.append(profile)
        self._write()
        return profile

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "model": self.model,
            "drivers": [profile.as_dict() for profile in self.profiles],
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass


class FaceEmbedder:
    """SFace ONNX wrapper: align the face and return an L2-normalised vector."""

    def __init__(
        self,
        model_path: Path | str,
        threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Identity model not found: {self.model_path}\n"
                "Run: python tools/download_models.py"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_outputs()[0].shape
        self.dimension = int(shape[1]) if len(shape) > 1 and isinstance(
            shape[1], int
        ) else 128

    def align(self, frame: np.ndarray, face: Face) -> np.ndarray:
        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        if landmarks.shape != (5, 2):
            raise ValueError("SFace needs five facial landmarks")
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            SFACE_REFERENCE,
            method=cv2.LMEDS,
        )
        if matrix is None:
            x = max(0, int(face.x))
            y = max(0, int(face.y))
            w = max(1, int(face.width))
            h = max(1, int(face.height))
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                raise ValueError("empty face crop")
            return cv2.resize(
                crop, SFACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR
            )
        return cv2.warpAffine(
            frame,
            matrix,
            SFACE_INPUT_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray | None:
        try:
            aligned = self.align(frame, face)
        except (ValueError, cv2.error):
            return None
        blob = cv2.dnn.blobFromImage(
            aligned,
            SFACE_SCALE,
            SFACE_INPUT_SIZE,
            SFACE_MEAN,
            swapRB=True,
            crop=False,
        )
        try:
            output = self.session.run(
                [self.output_name], {self.input_name: blob}
            )[0]
        except Exception:
            return None
        embedding = normalize_embedding(np.asarray(output))
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            return None
        return embedding


class IdentityVerifier:
    """Matches a face embedding against enrolled profiles with confirmation."""

    def __init__(
        self,
        config: DMSConfig,
        profiles: list[IdentityProfile],
        embedder: FaceEmbedder | None,
        authorized: list[str] | None = None,
    ) -> None:
        self.config = config
        self.profiles = list(profiles)
        self.embedder = embedder
        self.authorized = [str(name) for name in (authorized or [])]
        self.enforced = bool(self.authorized)
        self._stable = UNKNOWN if self.profiles else OFF
        self._name: str | None = None
        self._score: float | None = None
        self._stable_name: str | None = None
        self._stable_score: float | None = None
        self._best_name: str | None = None
        self._best_score: float | None = None
        self._match_streak = 0
        self._mismatch_streak = 0
        self._checks = 0
        self._valid = False

    @property
    def available(self) -> bool:
        return bool(self.profiles and self.embedder is not None)

    def update(
        self,
        frame: np.ndarray,
        face: Face | None,
        pose: HeadPose | None,
        now: float,
        frame_index: int = 0,
    ) -> IdentityResult:
        del now, frame_index  # kept for a uniform call signature
        if not self.available:
            return self._result(OFF, "IDENTITY OFF")
        if face is None or not face.detected:
            return self.no_face()

        if face.width < self.config.identity_min_face_px:
            return self._low_quality("IDENTITY: MOVE CLOSER")
        if pose is None:
            return self._low_quality("IDENTITY: FACE THE CAMERA")
        if (
            abs(pose.yaw) > 35.0
            or abs(pose.pitch) > 28.0
            or abs(pose.roll) > 30.0
        ):
            return self._low_quality("IDENTITY: LOOK STRAIGHT")

        embedding = self.embedder.embed(frame, face) if self.embedder else None
        if embedding is None:
            return self._low_quality("IDENTITY: HOLD STILL")

        best_profile = None
        best_score = -1.0
        for profile in self.profiles:
            score = cosine_similarity(embedding, profile.embedding)
            if score > best_score:
                best_profile = profile
                best_score = score
        self._checks += 1
        self._best_name = best_profile.name if best_profile else None
        self._best_score = float(best_score) if best_profile else None

        matched = (
            best_profile is not None
            and best_score >= self.config.identity_threshold
        )
        authorized_match = matched and (
            not self.enforced or best_profile.name in self.authorized
        )

        if authorized_match:
            self._match_streak += 1
            self._mismatch_streak = 0
            self._name = best_profile.name
            self._score = float(best_score)
            self._valid = True
            if self._match_streak >= self.config.identity_confirm_checks:
                self._stable = MATCH
                self._stable_name = best_profile.name
                self._stable_score = float(best_score)
                self._valid = True
                return self._result(
                    MATCH, f"DRIVER {self._stable_name} {self._stable_score:.2f}"
                )
            state = self._stable if self._stable in STABLE_STATES else CHECKING
            return self._result(
                state,
                f"IDENTITY CHECKING {self._match_streak}/"
                f"{self.config.identity_confirm_checks}",
            )

        self._mismatch_streak += 1
        self._match_streak = 0
        self._name = None
        self._score = float(best_score) if best_profile else None
        self._valid = True
        if self._mismatch_streak >= self.config.identity_confirm_checks:
            self._stable = MISMATCH
            self._stable_name = None
            self._stable_score = None
            return self._result(MISMATCH, "UNAUTHORIZED DRIVER")
        state = self._stable if self._stable in STABLE_STATES else CHECKING
        return self._result(
            state,
            f"IDENTITY MISMATCH {self._mismatch_streak}/"
            f"{self.config.identity_confirm_checks}",
        )

    def no_face(self) -> IdentityResult:
        self._match_streak = 0
        self._mismatch_streak = 0
        self._stable = NO_FACE
        self._stable_name = None
        self._stable_score = None
        self._valid = False
        self._name = None
        self._score = None
        return self._result(NO_FACE, "NO DRIVER FACE")

    def reset(self) -> None:
        self._stable = UNKNOWN if self.profiles else OFF
        self._name = None
        self._score = None
        self._stable_name = None
        self._stable_score = None
        self._best_name = None
        self._best_score = None
        self._match_streak = 0
        self._mismatch_streak = 0
        self._checks = 0
        self._valid = False

    def _low_quality(self, message: str) -> IdentityResult:
        state = self._stable if self._stable in STABLE_STATES else CHECKING
        return self._result(state, message, valid=False)

    def _result(
        self, state: str, message: str, valid: bool | None = None
    ) -> IdentityResult:
        return IdentityResult(
            state=state,
            name=self._stable_name if state == MATCH else None,
            score=self._stable_score if state == MATCH else None,
            best_name=self._best_name,
            best_score=self._best_score,
            checks=self._checks,
            message=message,
            valid=self._valid if valid is None else valid,
            enforced=self.enforced,
        )


@dataclass
class EnrollmentSession:
    """Collects good frontal frames and averages their embeddings."""

    name: str
    embedder: FaceEmbedder
    config: DMSConfig
    target_samples: int = 30
    embeddings: list[np.ndarray] = field(default_factory=list)
    last_sample: float = -1.0
    rejected: int = 0

    @property
    def count(self) -> int:
        return len(self.embeddings)

    @property
    def complete(self) -> bool:
        return self.count >= self.target_samples

    def add(
        self,
        frame: np.ndarray,
        face: Face | None,
        pose: HeadPose | None,
        eyes: EyeMetrics | None,
        now: float,
    ) -> bool:
        if self.complete:
            return False
        if now - self.last_sample < 0.08:
            return False
        if face is None or not face.detected:
            self.rejected += 1
            return False
        if face.width < self.config.setup_min_face_px:
            self.rejected += 1
            return False
        if pose is None:
            self.rejected += 1
            return False
        if (
            abs(pose.yaw) > self.config.setup_max_yaw
            or abs(pose.pitch) > self.config.setup_max_pitch
            or abs(pose.roll) > self.config.setup_max_roll
        ):
            self.rejected += 1
            return False
        if (
            eyes is not None
            and eyes.valid
            and (eyes.left_open + eyes.right_open) / 2.0 < 0.5
        ):
            self.rejected += 1
            return False
        embedding = self.embedder.embed(frame, face)
        if embedding is None:
            self.rejected += 1
            return False
        self.embeddings.append(embedding)
        self.last_sample = now
        return True

    def finish(self, store: IdentityStore, replace: bool = False) -> IdentityProfile:
        if not self.embeddings:
            raise ValueError("no usable enrolment frames were captured")
        average = normalize_embedding(
            np.mean(np.stack(self.embeddings, axis=0), axis=0)
        )
        return store.save_profile(
            self.name,
            average,
            samples=len(self.embeddings),
            replace=replace,
        )


def identity_quality_ok(
    config: DMSConfig,
    face: Face | None,
    pose: HeadPose | None,
    eyes: EyeMetrics | None = None,
) -> tuple[bool, str]:
    """Shared quality gate used by enrolment and tests."""
    if face is None or not face.detected:
        return False, "NO FACE"
    if face.width < config.setup_min_face_px:
        return False, "MOVE CLOSER"
    if pose is None:
        return False, "FACE THE CAMERA"
    if abs(pose.yaw) > config.setup_max_yaw:
        return False, "LOOK STRAIGHT"
    if abs(pose.pitch) > config.setup_max_pitch:
        return False, "LOOK STRAIGHT"
    if abs(pose.roll) > config.setup_max_roll:
        return False, "LEVEL YOUR HEAD"
    if (
        eyes is not None
        and eyes.valid
        and (eyes.left_open + eyes.right_open) / 2.0 < 0.5
    ):
        return False, "OPEN YOUR EYES"
    return True, "READY"


__all__ = [
    "CHECKING",
    "EnrollmentSession",
    "FaceEmbedder",
    "IdentityProfile",
    "IdentityResult",
    "IdentityStore",
    "IdentityVerifier",
    "MATCH",
    "MISMATCH",
    "NO_FACE",
    "OFF",
    "SFACE_REFERENCE",
    "UNKNOWN",
    "cosine_similarity",
    "identity_quality_ok",
    "normalize_embedding",
]
