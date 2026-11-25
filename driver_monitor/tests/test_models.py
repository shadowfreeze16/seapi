"""Model wrappers, preprocessing and face tracking."""

from __future__ import annotations

import unittest

import numpy as np

from dms.config import DMSConfig
from dms.models import (
    EyeObservation,
    Face,
    FaceDetector,
    FaceTracker,
    _softmax_if_needed,
    _square_crop,
)


def make_face(
    x: float,
    y: float,
    w: float = 100.0,
    h: float = 120.0,
    detected: bool = True,
) -> Face:
    box = np.array([x, y, w, h], dtype=np.float32)
    landmarks = np.array(
        [
            [x + 25, y + 40],
            [x + 75, y + 40],
            [x + 50, y + 65],
            [x + 32, y + 95],
            [x + 68, y + 95],
        ],
        dtype=np.float32,
    )
    return Face(box=box, landmarks=landmarks, score=0.9, detected=detected)


class CropTests(unittest.TestCase):
    def test_square_crop_inside(self) -> None:
        frame = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
        crop = _square_crop(frame, 60, 50, 20)
        self.assertEqual(crop.shape, (20, 20, 3))
        self.assertTrue(np.array_equal(crop, frame[40:60, 50:70]))

    def test_square_crop_pads_edges(self) -> None:
        frame = np.full((50, 60, 3), 7, dtype=np.uint8)
        crop = _square_crop(frame, 0, 0, 20)
        self.assertEqual(crop.shape, (20, 20, 3))
        self.assertTrue(np.all(crop == 7))

    def test_square_crop_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            _square_crop(np.zeros((0, 0, 3), dtype=np.uint8), 0, 0, 5)


class SoftmaxTests(unittest.TestCase):
    def test_probabilities_pass_through(self) -> None:
        values = np.array([0.1, 0.8], dtype=np.float32)
        np.testing.assert_allclose(_softmax_if_needed(values), values)

    def test_logits_are_sigmoided(self) -> None:
        values = np.array([-1.0, 1.0], dtype=np.float32)
        out = _softmax_if_needed(values)
        np.testing.assert_allclose(out, [0.26894142, 0.7310586], atol=1e-6)

    def test_large_logits_are_stable(self) -> None:
        out = _softmax_if_needed(np.array([-1000.0, 1000.0], dtype=np.float32))
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)


class FaceDetectorScaleTests(unittest.TestCase):
    def test_boxes_and_landmarks_are_scaled_back(self) -> None:
        detector = object.__new__(FaceDetector)
        detector.width = 640
        detector.height = 480
        detector.input_scale = 0.5
        detector._input_size = (320, 240)

        expected_size = (240, 320)

        class Stub:
            @staticmethod
            def detect(frame):
                assert frame.shape[:2] == expected_size
                row = np.array(
                    [
                        10.0,
                        20.0,
                        30.0,
                        40.0,
                        12.0,
                        25.0,
                        35.0,
                        25.0,
                        22.0,
                        32.0,
                        15.0,
                        38.0,
                        30.0,
                        38.0,
                        0.9,
                    ],
                    dtype=np.float32,
                )
                return 0, row[None]

        detector.detector = Stub()
        faces = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(faces), 1)
        np.testing.assert_allclose(faces[0].box, [20, 40, 60, 80])
        np.testing.assert_allclose(
            faces[0].landmarks[0], [24.0, 50.0]
        )
        np.testing.assert_allclose(
            faces[0].landmarks[1], [70.0, 50.0]
        )
        self.assertAlmostEqual(faces[0].score, 0.9)


class FaceTests(unittest.TestCase):
    def test_properties(self) -> None:
        face = make_face(10, 20, 100, 120)
        self.assertEqual(face.x, 10)
        self.assertEqual(face.y, 20)
        self.assertEqual(face.width, 100)
        self.assertEqual(face.height, 120)
        self.assertEqual(face.area, 12000)
        self.assertEqual(face.center, (60, 80))

    def test_clipped(self) -> None:
        face = make_face(-10, -20, 100, 120)
        clipped = face.clipped(200, 200)
        self.assertGreaterEqual(clipped.x, 0)
        self.assertGreaterEqual(clipped.y, 0)
        self.assertLessEqual(clipped.x + clipped.width, 200)
        self.assertLessEqual(clipped.y + clipped.height, 200)
        self.assertTrue(np.all(clipped.landmarks[:, 0] >= 0))
        self.assertTrue(np.all(clipped.landmarks[:, 1] >= 0))


class FaceTrackerTests(unittest.TestCase):
    def test_selects_largest_and_initializes(self) -> None:
        tracker = FaceTracker(smoothing=0.5, max_missing=2)
        small = make_face(0, 0, 50, 50)
        large = make_face(100, 100, 120, 120)
        face = tracker.update([small, large], 640, 480)
        self.assertIsNotNone(face)
        np.testing.assert_allclose(face.box, large.box)

    def test_smooths_box(self) -> None:
        tracker = FaceTracker(smoothing=0.5, max_missing=2)
        tracker.update([make_face(0, 0)], 640, 480)
        face = tracker.update([make_face(40, 0)], 640, 480)
        self.assertAlmostEqual(float(face.box[0]), 20.0, places=5)
        self.assertTrue(face.detected)

    def test_jump_resets_box(self) -> None:
        tracker = FaceTracker(smoothing=0.9, max_missing=2)
        tracker.update([make_face(0, 0)], 640, 480)
        face = tracker.update([make_face(400, 0)], 640, 480)
        self.assertAlmostEqual(float(face.box[0]), 400.0, places=5)

    def test_holds_through_dropout(self) -> None:
        tracker = FaceTracker(smoothing=0.0, max_missing=2)
        tracker.update([make_face(10, 10)], 640, 480)
        held = tracker.update([], 640, 480)
        self.assertIsNotNone(held)
        self.assertFalse(held.detected)
        self.assertEqual(held.age, 1)
        held = tracker.update([], 640, 480)
        self.assertEqual(held.age, 2)
        self.assertIsNone(tracker.update([], 640, 480))

    def test_reset(self) -> None:
        tracker = FaceTracker()
        tracker.update([make_face(0, 0)], 640, 480)
        tracker.reset()
        self.assertIsNone(tracker.face)
        self.assertEqual(tracker.missing, 0)


class EyeObservationTests(unittest.TestCase):
    def test_average_and_both_closed(self) -> None:
        observation = EyeObservation(0.2, 0.4, True, [])
        self.assertAlmostEqual(observation.average_open, 0.3)
        self.assertTrue(observation.both_closed)
        self.assertFalse(EyeObservation(0.2, 0.8, True, []).both_closed)
        self.assertFalse(EyeObservation(0.0, 0.0, False, []).both_closed)


class ModelFileTests(unittest.TestCase):
    """Loads the real models when they are available."""

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        from dms.models import FaceDetector

        cls.config = DMSConfig()
        if not all(
            path.exists()
            for path in (
                cls.config.face_model_path,
                cls.config.headpose_model_path,
                cls.config.eye_model_path,
            )
        ):
            raise unittest.SkipTest("ONNX models are not downloaded")
        cls.detector = FaceDetector(
            cls.config.face_model_path,
            cls.config.width,
            cls.config.height,
            confidence=0.5,
            nms=cls.config.face_nms,
            top_k=cls.config.face_top_k,
            input_scale=cls.config.face_input_scale,
        )
        images = sorted((Path(__file__).resolve().parent.parent / "events").glob("*.jpg"))
        cls.image = None
        for path in images:
            import cv2

            frame = cv2.imread(str(path))
            if frame is None:
                continue
            faces = cls.detector.detect(frame)
            if faces:
                cls.image = frame
                cls.face = max(faces, key=lambda f: f.area)
                break
        if cls.image is None:
            raise unittest.SkipTest("no event photo with a detectable face")

    def test_face_detector(self) -> None:
        faces = self.detector.detect(self.image)
        self.assertGreaterEqual(len(faces), 1)
        self.assertEqual(faces[0].landmarks.shape, (5, 2))

    def test_head_pose(self) -> None:
        from dms.models import HeadPoseEstimator

        estimator = HeadPoseEstimator(self.config.headpose_model_path)
        pose = estimator.estimate(self.image, self.face)
        self.assertIsNotNone(pose)
        self.assertTrue(np.isfinite([pose.yaw, pose.pitch, pose.roll]).all())
        self.assertEqual(pose.rotation.shape, (3, 3))
        self.assertAlmostEqual(float(np.linalg.det(pose.rotation)), 1.0, places=3)

    def test_eye_state(self) -> None:
        from dms.models import EyeStateEstimator

        estimator = EyeStateEstimator(self.config.eye_model_path)
        observation = estimator.estimate(self.image, self.face)
        self.assertTrue(observation.valid)
        self.assertGreaterEqual(observation.left_open, 0.0)
        self.assertLessEqual(observation.left_open, 1.0)
        self.assertGreaterEqual(observation.right_open, 0.0)
        self.assertLessEqual(observation.right_open, 1.0)
        self.assertEqual(len(observation.boxes), 2)


if __name__ == "__main__":
    unittest.main()