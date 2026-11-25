import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch

import backend.ingestion as ingestion_module
from backend.ingestion import (
    CHUNK_SIZE,
    IngestionConflict,
    IngestionRegistry,
    VideoIngestionService,
)


class FakeProcessor:
    def __init__(self, events=None, error=None):
        self.events = [] if events is None else events
        self.error = error
        self.calls = []

    def __call__(self, video_path, video_id):
        self.calls.append((video_path, video_id))
        if self.error:
            raise self.error
        return self.events


class FakeIndexer:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def __call__(self, events):
        self.calls.append(events)
        if self.error:
            raise self.error


class CountingMetric:
    def __init__(self):
        self.value = 0

    def inc(self, amount):
        self.value += amount


class BoundedReadStream(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError("Upload content must be read in bounded chunks.")
        return super().read(size)


class FailingReadStream:
    def __init__(self):
        self.read_count = 0

    def read(self, size):
        self.read_count += 1
        if self.read_count == 1:
            return b"partial upload"
        raise OSError("read failed")


class VideoIngestionServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "registry.sqlite3"
        self.video_dir = self.root / "videos"
        self.temp_dir = self.root / "uploads"
        self.temp_dir.mkdir()
        self.events = [{
            "video_id": "placeholder",
            "timestamp": 1.0,
            "frame_path": "unused.jpg",
            "display_time": "00:01",
            "event_probability": 1.0,
        }]
        self.processor = FakeProcessor(events=self.events)
        self.indexer = FakeIndexer()
        self.service = self.make_service(self.processor, self.indexer)

    def tearDown(self):
        self.temp.cleanup()

    def make_service(self, processor, indexer, db_path=None, registry=None):
        return VideoIngestionService(
            registry=registry or IngestionRegistry(db_path or self.db_path),
            process_video=processor,
            index_events=indexer,
            video_dir=self.video_dir,
            temp_dir=self.temp_dir,
        )

    def upload(self, content, filename="clip.mp4", stream_type=io.BytesIO):
        return self.service.ingest(stream_type(content), filename)

    def record(self, content):
        digest = hashlib.sha256(content).hexdigest()
        connection = sqlite3.connect(self.db_path)
        try:
            return connection.execute(
                "SELECT video_id, video_path, state, response_json, error "
                "FROM ingestions WHERE content_sha256 = ?",
                (digest,),
            ).fetchone()
        finally:
            connection.close()

    def test_same_bytes_and_filename_are_processed_and_indexed_once(self):
        content = b"identical video"
        first = self.upload(content)
        second = self.upload(content)

        self.assertEqual(second.response, first.response)
        self.assertTrue(first.newly_processed)
        self.assertFalse(second.newly_processed)
        self.assertEqual(len(self.processor.calls), 1)
        self.assertEqual(len(self.indexer.calls), 1)

    def test_reused_upload_does_not_increment_verified_events_metric_again(self):
        content = b"verified event"
        first = self.upload(content)
        second = self.upload(content)
        metric = CountingMetric()

        for outcome in (first, second):
            increment = outcome.verified_events_metric_increment
            if increment:
                metric.inc(increment)

        self.assertEqual(first.verified_events_metric_increment, 1)
        self.assertEqual(second.verified_events_metric_increment, 0)
        self.assertEqual(metric.value, 1)

    def test_same_bytes_with_different_filenames_reuses_original_video(self):
        content = b"identical video"
        first = self.upload(content, "first-name.mp4")
        second = self.upload(content, "different-name.mp4")

        self.assertEqual(second.response, first.response)
        self.assertIn("first-name.mp4", second.response["video_url"])
        self.assertEqual(
            set(second.response),
            {"status", "video_id", "video_url", "events_verified"},
        )
        self.assertEqual(len(self.processor.calls), 1)
        self.assertEqual(len(self.indexer.calls), 1)

    def test_different_bytes_with_same_filename_are_distinct(self):
        first = self.upload(b"video one", "same.mp4")
        second = self.upload(b"video two", "same.mp4")

        self.assertNotEqual(first.response["video_id"], second.response["video_id"])
        self.assertEqual(len(self.processor.calls), 2)
        self.assertEqual(len(self.indexer.calls), 2)

    def test_new_ids_are_full_uuids_and_video_paths_stay_in_storage(self):
        outcome = self.upload(b"new identifier", "../../outside.mp4")
        record = self.record(b"new identifier")
        stored_path = Path(record[1])

        self.assertEqual(len(outcome.response["video_id"]), 36)
        self.assertEqual(stored_path.parent, self.video_dir)
        self.assertTrue(stored_path.is_file())

    def test_existing_short_id_video_is_not_renamed_or_removed(self):
        old_video = self.video_dir / "1234abcd_legacy.mp4"
        old_video.parent.mkdir(parents=True)
        old_video.write_bytes(b"legacy")

        self.upload(b"new video")

        self.assertEqual(old_video.read_bytes(), b"legacy")

    def test_registry_reinitialization_preserves_duplicate_detection(self):
        content = b"persistent video"
        first = self.upload(content)
        second_processor = FakeProcessor(events=self.events)
        second_indexer = FakeIndexer()
        restarted_service = self.make_service(
            second_processor, second_indexer, db_path=self.db_path
        )

        second = restarted_service.ingest(io.BytesIO(content), "renamed.mp4")

        self.assertEqual(second.response, first.response)
        self.assertEqual(json.loads(self.record(content)[3]), first.response)
        self.assertFalse(second.newly_processed)
        self.assertEqual(second_processor.calls, [])
        self.assertEqual(second_indexer.calls, [])

    def test_relative_registry_path_is_resolved_and_survives_cwd_change(self):
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            try:
                os.chdir(first_dir)
                registry = IngestionRegistry("registry/state.sqlite3")
                expected_path = Path(first_dir) / "registry/state.sqlite3"
                self.assertEqual(registry.db_path, expected_path.resolve())

                os.chdir(second_dir)
                video_path = Path(second_dir) / "videos" / "stored.mp4"
                video_path.parent.mkdir()
                video_path.write_bytes(b"stored")
                response = {
                    "status": "abstained_no_motion",
                    "video_id": "legacy-compatible-id",
                    "video_url": "/videos/legacy-compatible-id_stored.mp4",
                    "events_verified": 0,
                }
                self.assertIsNone(registry.begin("hash", "new-id", video_path))
                registry.complete("hash", response)
                duplicate = registry.begin("hash", "other-id", video_path)
                path_exists = expected_path.is_file()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(duplicate, response)
        self.assertTrue(path_exists)

    def test_registry_connections_close_after_commit_and_rollback(self):
        registry = IngestionRegistry(self.db_path)
        committed = None
        with registry._transaction() as connection:
            committed = connection
            connection.execute(
                """
                INSERT INTO ingestions
                    (content_sha256, video_id, video_path, state)
                VALUES ('committed', 'id', '/tmp/video', 'processing')
                """
            )
        with self.assertRaises(sqlite3.ProgrammingError):
            committed.execute("SELECT 1")

        rolled_back = None
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with registry._transaction() as connection:
                rolled_back = connection
                connection.execute(
                    """
                    INSERT INTO ingestions
                        (content_sha256, video_id, video_path, state)
                    VALUES ('rolled-back', 'id', '/tmp/video', 'processing')
                    """
                )
                raise RuntimeError("rollback")
        with self.assertRaises(sqlite3.ProgrammingError):
            rolled_back.execute("SELECT 1")

        connection = sqlite3.connect(registry.db_path)
        try:
            rows = connection.execute(
                "SELECT content_sha256 FROM ingestions ORDER BY content_sha256"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(rows, [("committed",)])

    def test_completed_entry_cannot_be_overwritten(self):
        outcome = self.upload(b"completed entry")
        record = self.record(b"completed entry")

        with self.assertRaisesRegex(RuntimeError, "completed"):
            self.service.registry.complete(
                hashlib.sha256(b"completed entry").hexdigest(),
                {"status": "incorrect"},
            )
        self.assertEqual(
            self.service.registry.begin(
                hashlib.sha256(b"completed entry").hexdigest(),
                "replacement-id",
                record[1],
            ),
            outcome.response,
        )
        self.assertEqual(self.record(b"completed entry")[3], json.dumps(outcome.response))

    def test_processing_failure_is_not_completed_and_retry_is_blocked(self):
        content = b"processing failure"
        failing_processor = FakeProcessor(error=ValueError("decode failed"))
        failing_service = self.make_service(failing_processor, FakeIndexer())

        with self.assertRaisesRegex(ValueError, "decode failed"):
            failing_service.ingest(io.BytesIO(content), "bad.mp4")

        record = self.record(content)
        self.assertEqual(record[2], "failed")
        self.assertIsNone(record[3])
        self.assertEqual(list(self.temp_dir.iterdir()), [])
        with self.assertRaisesRegex(IngestionConflict, "did not complete"):
            self.service.ingest(io.BytesIO(content), "bad.mp4")
        self.assertEqual(len(failing_processor.calls), 1)
        self.assertEqual(self.processor.calls, [])

    def test_zero_event_video_is_a_completed_abstention(self):
        content = b"valid video with no motion"
        processor = FakeProcessor(events=[])
        indexer = FakeIndexer()
        service = self.make_service(processor, indexer)

        first = service.ingest(io.BytesIO(content), "still.mp4")
        second = service.ingest(io.BytesIO(content), "still.mp4")

        self.assertEqual(first.response["status"], "abstained_no_motion")
        self.assertEqual(first.response["events_verified"], 0)
        self.assertEqual(second.response, first.response)
        self.assertEqual(first.verified_events_metric_increment, 0)
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(indexer.calls, [])
        self.assertEqual(self.record(content)[2], "completed")

    def test_duplicate_temporary_upload_is_removed(self):
        content = b"cleanup duplicate"
        self.upload(content)
        self.upload(content)

        self.assertEqual(list(self.temp_dir.iterdir()), [])

    def test_upload_read_failure_removes_temporary_file(self):
        with self.assertRaisesRegex(OSError, "read failed"):
            self.service.ingest(FailingReadStream(), "read-error.mp4")

        self.assertEqual(list(self.temp_dir.iterdir()), [])

    def test_registry_failure_removes_temporary_file(self):
        class FailingRegistry:
            def begin(self, *args):
                raise RuntimeError("registry unavailable")

        service = self.make_service(
            FakeProcessor(events=self.events),
            FakeIndexer(),
            registry=FailingRegistry(),
        )
        with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
            service.ingest(io.BytesIO(b"registry failure"), "registry.mp4")

        self.assertEqual(list(self.temp_dir.iterdir()), [])

    def test_cleanup_error_is_logged_without_masking_primary_failure(self):
        processor = FakeProcessor(error=ValueError("processing failed"))
        service = self.make_service(processor, FakeIndexer())
        real_unlink = os.unlink
        attempted_paths = []

        def fail_unlink(path):
            attempted_paths.append(Path(path))
            raise PermissionError("private path detail")

        with patch.object(ingestion_module.os, "unlink", side_effect=fail_unlink):
            with self.assertLogs(ingestion_module.logger, level=logging.WARNING) as log:
                with self.assertRaisesRegex(ValueError, "processing failed"):
                    service.ingest(io.BytesIO(b"secret upload bytes"), "failure.mp4")

        self.assertTrue(attempted_paths)
        self.assertTrue(all(path.is_file() for path in attempted_paths))
        self.assertNotIn("private path detail", "\n".join(log.output))
        for path in attempted_paths:
            real_unlink(path)

    def test_partial_index_failure_blocks_retry(self):
        content = b"partial index failure"
        failing_indexer = FakeIndexer(error=RuntimeError("index write failed"))
        failing_processor = FakeProcessor(events=self.events)
        failing_service = self.make_service(failing_processor, failing_indexer)

        with self.assertRaisesRegex(RuntimeError, "index write failed"):
            failing_service.ingest(io.BytesIO(content), "partial.mp4")

        record = self.record(content)
        self.assertEqual(record[2], "failed")
        self.assertIsNone(record[3])
        self.assertTrue(Path(record[1]).is_file())
        self.assertEqual(list(self.temp_dir.iterdir()), [])
        with self.assertRaisesRegex(IngestionConflict, "partial index rows"):
            self.service.ingest(io.BytesIO(content), "partial.mp4")
        self.assertEqual(len(failing_processor.calls), 1)
        self.assertEqual(len(failing_indexer.calls), 1)
        self.assertEqual(self.processor.calls, [])

    def test_missing_completed_video_does_not_return_success(self):
        content = b"missing stored file"
        first = self.upload(content)
        stored_path = self.video_dir / Path(first.response["video_url"]).name
        stored_path.unlink()

        with self.assertRaisesRegex(IngestionConflict, "stored file is missing"):
            self.upload(content)
        self.assertEqual(len(self.processor.calls), 1)
        self.assertEqual(len(self.indexer.calls), 1)
        self.assertEqual(list(self.temp_dir.iterdir()), [])

    def test_hashing_reads_upload_in_bounded_chunks(self):
        content = b"x" * (CHUNK_SIZE * 2 + 17)
        stream = BoundedReadStream(content)

        self.service.ingest(stream, "large.mp4")

        self.assertGreater(len(stream.read_sizes), 2)
        self.assertTrue(all(size == CHUNK_SIZE for size in stream.read_sizes))


class FakeFrame:
    shape = (360, 640, 3)


class FakeCapture:
    def __init__(self, opened=True, frames=4, fps=30, read_error_after=None):
        self.opened = opened
        self.frames = frames
        self.fps = fps
        self.read_error_after = read_error_after
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == 1:
            return self.fps
        return self.frames

    def read(self):
        if self.read_error_after == self.read_count:
            raise RuntimeError("frame read failed")
        if self.read_count == self.frames:
            return False, None
        self.read_count += 1
        return True, FakeFrame()

    def release(self):
        self.released = True


class VideoPipelineValidityTest(unittest.TestCase):
    def load_pipeline(self, capture, temp_dir, resize_error=None):
        cv2 = types.ModuleType("cv2")
        cv2.CAP_PROP_FPS = 1
        cv2.CAP_PROP_FRAME_COUNT = 2
        cv2.INTER_NEAREST = 0
        cv2.VideoCapture = lambda path: capture

        def resize(frame, size, interpolation):
            if resize_error:
                raise resize_error
            return frame

        cv2.resize = resize
        numpy = types.ModuleType("numpy")
        with patch.dict("sys.modules", {"cv2": cv2, "numpy": numpy}):
            module_spec = importlib.util.spec_from_file_location(
                f"isolated_processing_{id(capture)}",
                Path(__file__).resolve().parents[1] / "processing.py",
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        return module.VideoPipeline(
            upload_dir=temp_dir / "videos",
            clips_dir=temp_dir / "clips",
        )

    def test_decodable_zero_motion_short_video_remains_valid_empty_result(self):
        with tempfile.TemporaryDirectory() as workdir:
            capture = FakeCapture(frames=4)
            pipeline = self.load_pipeline(capture, Path(workdir))

            events = pipeline.process_video("fake.mp4", "test-id")

        self.assertEqual(events, [])
        self.assertTrue(capture.released)

    def test_unopenable_video_releases_capture(self):
        with tempfile.TemporaryDirectory() as workdir:
            capture = FakeCapture(opened=False)
            pipeline = self.load_pipeline(capture, Path(workdir))

            with self.assertRaisesRegex(ValueError, "Unable to open video"):
                pipeline.process_video("invalid.mp4", "test-id")

        self.assertTrue(capture.released)

    def test_invalid_frame_rate_releases_capture(self):
        for fps in (0, float("nan"), float("inf")):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as workdir:
                capture = FakeCapture(fps=fps)
                pipeline = self.load_pipeline(capture, Path(workdir))

                with self.assertRaisesRegex(ValueError, "invalid frame rate"):
                    pipeline.process_video("invalid-fps.mp4", "test-id")

                self.assertTrue(capture.released)

    def test_frame_read_exception_releases_capture(self):
        with tempfile.TemporaryDirectory() as workdir:
            capture = FakeCapture(read_error_after=1)
            pipeline = self.load_pipeline(capture, Path(workdir))

            with self.assertRaisesRegex(RuntimeError, "frame read failed"):
                pipeline.process_video("read-error.mp4", "test-id")

        self.assertTrue(capture.released)

    def test_frame_processing_exception_releases_capture(self):
        with tempfile.TemporaryDirectory() as workdir:
            capture = FakeCapture(frames=1)
            pipeline = self.load_pipeline(
                capture, Path(workdir), resize_error=RuntimeError("resize failed")
            )

            with self.assertRaisesRegex(RuntimeError, "resize failed"):
                pipeline.process_video("resize-error.mp4", "test-id")

        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
