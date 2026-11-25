"""SFace identity helpers, profile storage and confirmation logic."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dms.config import DMSConfig
from dms.eyes import EyeMetrics
from dms.identity import (
    CHECKING,
    MATCH,
    MISMATCH,
    EnrollmentSession,
    IdentityProfile,
    IdentityStore,
    IdentityVerifier,
    cosine_similarity,
    identity_quality_ok,
    normalize_embedding,
)
from dms.models import Face, HeadPose


def make_face(width: float = 120.0) -> Face:
    return Face(
        box=np.array([100, 80, width, width], dtype=np.float32),
        landmarks=np.array(
            [
                [110, 100],
                [180, 100],
                [145, 130],
                [115, 160],
                [175, 160],
            ],
            dtype=np.float32,
        ),
        score=0.98,
    )


def frontal_pose() -> HeadPose:
    return HeadPose(
        yaw=1.0,
        pitch=-1.0,
        roll=0.5,
        rotation=np.eye(3, dtype=np.float32),
    )


class FakeEmbedder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = normalize_embedding(vector)
        self.calls = 0

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        self.calls += 1
        return self.vector


class SimilarityTests(unittest.TestCase):
    def test_normalize_and_cosine(self) -> None:
        a = normalize_embedding(np.array([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(a, -a), -1.0, places=5)
        self.assertEqual(cosine_similarity(a, np.array([0.0, 0.0])), 0.0)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "driver_identity.json"
        self.store = IdentityStore(self.path, model="sface.onnx")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_save_load_and_permissions(self) -> None:
        profile = self.store.save_profile(
            "Alice", np.array([1.0, 2.0, 3.0]), samples=12
        )
        self.assertEqual(profile.name, "Alice")
        self.assertTrue(self.path.exists())
        mode = stat.S_IMODE(os.stat(self.path).st_mode)
        self.assertEqual(mode, 0o600)
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("embedding", text)
        self.assertNotIn("image", text)

        reloaded = IdentityStore(self.path, model="sface.onnx")
        self.assertEqual(reloaded.names(), ["Alice"])
        restored = reloaded.get("Alice")
        assert restored is not None
        self.assertAlmostEqual(
            cosine_similarity(restored.embedding, profile.embedding),
            1.0,
            places=5,
        )
        self.assertEqual(restored.samples, 12)

    def test_duplicate_requires_replace(self) -> None:
        self.store.save_profile("Alice", np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            self.store.save_profile("Alice", np.array([0.0, 1.0]))
        self.store.save_profile(
            "Alice", np.array([0.0, 1.0]), replace=True
        )
        self.assertEqual(len(self.store.names()), 1)

    def test_missing_or_bad_file_is_empty(self) -> None:
        self.assertEqual(IdentityStore(self.path).profiles, [])
        self.path.write_text("not json", encoding="utf-8")
        self.assertEqual(IdentityStore(self.path).profiles, [])


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.config.identity_confirm_checks = 3
        self.config.identity_every = 1
        self.alice = normalize_embedding(np.array([1.0, 0.0, 0.0]))
        self.bob = normalize_embedding(np.array([0.0, 1.0, 0.0]))
        self.profiles = [
            IdentityProfile("Alice", self.alice),
            IdentityProfile("Bob", self.bob),
        ]
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.face = make_face()
        self.pose = frontal_pose()

    def update(self, verifier: IdentityVerifier, count: int = 1):
        state = None
        for i in range(count):
            state = verifier.update(
                self.frame, self.face, self.pose, float(i), i
            )
        return state

    def test_match_requires_confirmation(self) -> None:
        verifier = IdentityVerifier(
            self.config, self.profiles, FakeEmbedder(self.alice), ["Alice"]
        )
        first = self.update(verifier)
        self.assertEqual(first.state, CHECKING)
        second = self.update(verifier)
        self.assertEqual(second.state, CHECKING)
        third = self.update(verifier)
        self.assertEqual(third.state, MATCH)
        self.assertEqual(third.name, "Alice")
        self.assertTrue(third.enforced)

    def test_mismatch_requires_confirmation(self) -> None:
        verifier = IdentityVerifier(
            self.config,
            self.profiles,
            FakeEmbedder(normalize_embedding(np.array([0.0, 0.0, 1.0]))),
            ["Alice"],
        )
        self.assertEqual(self.update(verifier).state, CHECKING)
        self.assertEqual(self.update(verifier).state, CHECKING)
        result = self.update(verifier)
        self.assertEqual(result.state, MISMATCH)
        self.assertTrue(result.mismatch)

    def test_authorized_driver_filter(self) -> None:
        verifier = IdentityVerifier(
            self.config, self.profiles, FakeEmbedder(self.bob), ["Alice"]
        )
        result = self.update(verifier, 3)
        self.assertEqual(result.state, MISMATCH)
        self.assertEqual(result.best_name, "Bob")

    def test_no_face_resets(self) -> None:
        verifier = IdentityVerifier(
            self.config, self.profiles, FakeEmbedder(self.alice), ["Alice"]
        )
        self.update(verifier, 3)
        result = verifier.no_face()
        self.assertEqual(result.state, "NO_FACE")
        self.assertEqual(result.name, None)

    def test_small_face_is_not_checked(self) -> None:
        verifier = IdentityVerifier(
            self.config, self.profiles, FakeEmbedder(self.alice), ["Alice"]
        )
        result = verifier.update(
            self.frame, make_face(40.0), self.pose, 0.0, 0
        )
        self.assertEqual(result.state, CHECKING)
        self.assertFalse(result.valid)


class EnrollmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = DMSConfig()
        self.config.identity_store = self.root / "driver_identity.json"
        self.config.enroll_samples = 3
        self.store = IdentityStore(self.config.identity_store)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_capture_and_finish(self) -> None:
        embedder = FakeEmbedder(np.array([1.0, 1.0, 0.0]))
        session = EnrollmentSession(
            "Alice", embedder, self.config, target_samples=3
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face = make_face()
        pose = frontal_pose()
        now = 0.0
        for _ in range(3):
            now += 0.1
            self.assertTrue(session.add(frame, face, pose, None, now))
        self.assertTrue(session.complete)
        profile = session.finish(self.store)
        self.assertEqual(profile.name, "Alice")
        self.assertEqual(profile.samples, 3)
        self.assertEqual(IdentityStore(self.config.identity_store).names(), ["Alice"])

    def test_quality_gate_uses_eye_metrics(self) -> None:
        config = DMSConfig()
        face = make_face()
        pose = frontal_pose()
        closed = EyeMetrics(valid=True, left_open=0.1, right_open=0.1)
        self.assertEqual(
            identity_quality_ok(config, face, pose, closed),
            (False, "OPEN YOUR EYES"),
        )
        open_eyes = EyeMetrics(valid=True, left_open=0.9, right_open=0.8)
        self.assertEqual(
            identity_quality_ok(config, face, pose, open_eyes),
            (True, "READY"),
        )

    def test_enrollment_rejects_closed_eyes(self) -> None:
        session = EnrollmentSession(
            "Alice", FakeEmbedder(np.array([1.0, 0.0])), self.config
        )
        closed = EyeMetrics(valid=True, left_open=0.1, right_open=0.1)
        self.assertFalse(
            session.add(
                np.zeros((480, 640, 3), dtype=np.uint8),
                make_face(),
                frontal_pose(),
                closed,
                1.0,
            )
        )
        self.assertEqual(session.count, 0)

    def test_rejects_bad_pose(self) -> None:
        session = EnrollmentSession(
            "Alice", FakeEmbedder(np.array([1.0, 0.0])), self.config
        )
        pose = HeadPose(
            yaw=40.0,
            pitch=0.0,
            roll=0.0,
            rotation=np.eye(3, dtype=np.float32),
        )
        self.assertFalse(
            session.add(
                np.zeros((480, 640, 3), dtype=np.uint8),
                make_face(),
                pose,
                None,
                1.0,
            )
        )
        self.assertEqual(session.count, 0)


if __name__ == "__main__":
    unittest.main()
