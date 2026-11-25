import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass


CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)


class IngestionConflict(Exception):
    pass


@dataclass(frozen=True)
class IngestionOutcome:
    response: dict
    newly_processed: bool

    @property
    def verified_events_metric_increment(self):
        if self.newly_processed and self.response["status"] == "success":
            return self.response["events_verified"]
        return 0


class IngestionRegistry:
    def __init__(self, db_path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestions (
                    content_sha256 TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    response_json TEXT,
                    error TEXT
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.db_path)

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def begin(self, content_sha256, video_id, video_path):
        with self._transaction() as connection:
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO ingestions
                    (content_sha256, video_id, video_path, state)
                VALUES (?, ?, ?, 'processing')
                """,
                (content_sha256, video_id, str(video_path)),
            )
            row = connection.execute(
                """
                SELECT video_path, state, response_json, error
                FROM ingestions WHERE content_sha256 = ?
                """,
                (content_sha256,),
            ).fetchone()

        stored_path, state, response_json, error = row
        if state == "completed":
            if not Path(stored_path).is_file():
                raise IngestionConflict(
                    "This video was previously ingested, but its stored file is missing."
                )
            return json.loads(response_json)
        if state == "processing" and inserted.rowcount == 1:
            return None
        if state == "processing":
            raise IngestionConflict(
                "An ingestion for this content is already in progress."
            )
        detail = f": {error}" if error else ""
        raise IngestionConflict(
            f"A previous ingestion for this content did not complete{detail}. "
            "Automatic retry is blocked because partial index rows may exist."
        )

    def complete(self, content_sha256, response):
        with self._transaction() as connection:
            result = connection.execute(
                """
                UPDATE ingestions
                SET state = 'completed', response_json = ?, error = NULL
                WHERE content_sha256 = ? AND state = 'processing'
                """,
                (json.dumps(response), content_sha256),
            )
            if result.rowcount != 1:
                raise RuntimeError("Could not mark ingestion as completed.")

    def fail(self, content_sha256, error):
        with self._transaction() as connection:
            result = connection.execute(
                """
                UPDATE ingestions
                SET state = 'failed', error = ?
                WHERE content_sha256 = ? AND state = 'processing'
                """,
                (str(error), content_sha256),
            )
            if result.rowcount != 1:
                raise RuntimeError("Could not mark ingestion as failed.")


class VideoIngestionService:
    def __init__(
        self,
        registry,
        process_video,
        index_events,
        video_dir="data/videos",
        temp_dir="/tmp",
    ):
        self.registry = registry
        self.process_video = process_video
        self.index_events = index_events
        self.video_dir = Path(video_dir).resolve()
        self.temp_dir = Path(temp_dir).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, upload, filename):
        temp_path = None
        content_hash = hashlib.sha256()
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="semantic-eye-",
                suffix=".upload",
                dir=self.temp_dir,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                while True:
                    chunk = upload.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    content_hash.update(chunk)
                    temp_file.write(chunk)

            digest = content_hash.hexdigest()
            safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1] or "video"
            video_id = str(uuid.uuid4())
            final_video_path = (
                self.video_dir / f"{video_id}_{safe_filename}"
            ).resolve()
            if not final_video_path.is_relative_to(self.video_dir):
                raise ValueError("Generated video path is outside the video directory.")

            existing_response = self.registry.begin(
                digest, video_id, final_video_path
            )
            if existing_response is not None:
                return IngestionOutcome(existing_response, newly_processed=False)

            try:
                events = self.process_video(str(temp_path), video_id)
                if events is None:
                    raise RuntimeError("Video processing returned no result.")

                self.video_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_path), str(final_video_path))
                temp_path = None

                status = "success" if events else "abstained_no_motion"
                if events:
                    self.index_events(events)

                response = {
                    "status": status,
                    "video_id": video_id,
                    "video_url": f"/videos/{video_id}_{safe_filename}",
                    "events_verified": len(events),
                }
                self.registry.complete(digest, response)
                return IngestionOutcome(response, newly_processed=True)
            except Exception as error:
                self.registry.fail(digest, error)
                raise
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Failed to remove a temporary upload file.")
