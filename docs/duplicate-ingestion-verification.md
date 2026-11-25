# Duplicate ingestion verification

Verification recorded on 2026-11-25. This report covers the isolated regression suite and the real-backend smoke test. The smoke test used only disposable storage; it did not mount or modify existing application data.

## Isolated regression tests

Command:

```text
python -m unittest discover -s backend/tests -v
```

Result: **Ran 24 tests in 0.136s — OK.** The tests use temporary directories, fake processors and indexers, and fake OpenCV captures; they do not initialize FastAPI or download models. Coverage includes duplicate reuse, persistent registry state, upload cleanup, failure blocking, zero-event handling, SQLite connection closure, video-capture release, and metric accounting.

## Real integration setup

The backend image was built from `backend/Dockerfile` and run as a standalone container, not through Docker Compose. The backend's `/app/data` directory, SQLite registry, upload staging directory, and model/Hugging Face/Torch caches were redirected to a fresh temporary host directory. No repository data or existing cache was mounted.

The video fixture was generated with OpenCV's `VideoWriter`: 320×240 MJPG, 24 frames per second, 72 frames total, with a moving synthetic texture. OpenCV decoded all 72 frames. The fixture's purpose was to produce motion events and exercise real frame decoding, embedding, and LanceDB indexing. It was not intended to represent real-world footage or measure search quality.

The backend loaded `sentence-transformers/clip-ViT-L-14` on CPU. The model revision reported by the runtime was `1b4b2e899178706d1b9905460ac21de1e0ba86a5`.

### Runtime and dependency versions

| Component | Version |
| --- | --- |
| Container base | `python:3.10-slim-bookworm` |
| CPython | 3.10.22 |
| FastAPI | 0.162.5 |
| Starlette | 1.7.0 |
| Uvicorn | 0.54.0 |
| python-multipart | 0.0.32 |
| OpenCV (`opencv-python`) | 5.1.0.95 (`cv2` 5.1.0) |
| PyTorch | 2.14.1 |
| NumPy | 2.2.6 |
| Pillow | 12.3.0 |
| SciPy | 1.15.3 |
| sentence-transformers | 6.3.0 |
| LanceDB | 0.41.0 |
| PyArrow | 25.0.1 |
| MLflow | 3.18.0 |
| prometheus-client | 0.26.0 |

## Upload and restart measurements

The measured run used a fresh registry and LanceDB table. Before upload, the index contained **0 rows** and `events_verified_total` was **0.0**.

### First upload

Uploading the fixture as `original-motion.avi` returned **HTTP 200**:

```json
{
  "status": "success",
  "video_id": "33ad80b5-0841-4227-b5ea-dea02cefbf58",
  "video_url": "/videos/33ad80b5-0841-4227-b5ea-dea02cefbf58_original-motion.avi",
  "events_verified": 7
}
```

Immediately afterward, LanceDB contained **7 rows** and `events_verified_total` was **7.0**.

### Identical bytes under a renamed file

Uploading the same bytes as `renamed-motion.avi` returned **HTTP 200** with the same video ID and the original playback URL. The response still reported `events_verified: 7`; this is the stored response metadata, not seven newly processed events. LanceDB remained at **7 rows**, the metric remained **7.0**, and the upload staging directory was empty.

### Backend restart

The standalone backend was restarted with the same disposable data and model-cache mounts. The LanceDB table still contained **7 rows**. The Prometheus counter is process-local, so it reset to **0.0** after restart. Re-uploading the identical fixture returned the same video ID and original URL; the row count remained **7** and the counter remained **0.0**.

An HTTP GET to the returned `/videos/...` URL after restart returned **200**, content type `video/x-msvideo`, and **585,626 bytes**. The downloaded response had the same SHA-256 as the uploaded fixture: `ea9c8f63b0c4495a763ad7272ac544bedd467f3f88b9b1212cc942aa30942389`.

## Failed and in-progress records

Synthetic `failed` and `processing` records were inserted only into the disposable SQLite registry. Uploads matching those records returned:

- Failed record: **HTTP 409**, `A previous ingestion for this content did not complete: synthetic isolated failure. Automatic retry is blocked because partial index rows may exist.`
- In-progress record: **HTTP 409**, `An ingestion for this content is already in progress.`

The final disposable dataset had 7 indexed rows, one completed ingestion, one synthetic failed record, and one synthetic processing record. The upload staging directory was empty. These fixtures were left in the disposable test data; no automatic recovery or cleanup was exercised.

## Limitations and review caveat

- The protection and restart checks establish **sequential** duplicate handling only. Concurrent-upload behavior was not tested.
- No search query, result-ranking, or retrieval-quality evaluation was performed.
- Playback was tested with an HTTP GET of the video file, not in a browser or through the frontend player.
- The fixture is synthetic moving texture; the test establishes that this input produced and indexed seven events, not that embeddings or search results are semantically useful.
- This smoke test did not simulate frame-write or partial-index failures. In the current code, `VideoPipeline` does not check the boolean result of `cv2.imwrite` ([`processing.py`](../backend/processing.py#L98)); the indexer skips missing frame paths and can return normally when none are valid ([`search_engine.py`](../backend/search_engine.py#L131)). Since the ingestion service treats a normal return from the indexer as success, a failed frame write could be recorded as completed without the expected rows. The successful seven-row smoke result does not cover this failure path.
