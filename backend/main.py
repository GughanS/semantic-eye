from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool
import os
import time
from processing import VideoPipeline
from search_engine import SearchEngine
from ingestion import IngestionConflict, IngestionRegistry, VideoIngestionService

# Observability
from prometheus_client import Counter, Histogram, generate_latest
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Eye: Event Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Duration", ["endpoint"])
EVENTS_VERIFIED = Counter("events_verified_total", "Total Verified Events")

# Initialize the new Architecture
pipeline = VideoPipeline() # Handles Optical Flow & Temporal Units
engine = SearchEngine()    # Handles YOLO Tagging & DB Search
ingestion_service = VideoIngestionService(
    registry=IngestionRegistry(
        os.getenv("SEMANTIC_EYE_REGISTRY_PATH", "data/video_ingestions.sqlite3")
    ),
    process_video=pipeline.process_video,
    index_events=engine.index_verified_events,
    video_dir="data/videos",
    temp_dir=os.getenv("SEMANTIC_EYE_UPLOAD_TEMP_DIR", "/tmp"),
)

app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")
app.mount("/clips", StaticFiles(directory="data/clips"), name="clips")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        outcome = await run_in_threadpool(
            ingestion_service.ingest, file.file, file.filename or "video"
        )
        verified_count = outcome.verified_events_metric_increment
        if verified_count:
            EVENTS_VERIFIED.inc(verified_count)
        
        REQUEST_COUNT.labels(method="POST", endpoint="/upload", http_status=200).inc()
        REQUEST_LATENCY.labels(endpoint="/upload").observe(time.time() - start_time)
        return outcome.response
    except IngestionConflict as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/upload", http_status=409).inc()
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/upload", http_status=500).inc()
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_video(query: str):
    start_time = time.time()
    try:
        # We can also track searches in MLflow
        if mlflow.active_run():
            mlflow.log_param("last_query", query)
            
        results = await run_in_threadpool(engine.search, query)
        
        # Modify paths to be accessible via URL
        for r in results:
            relative_path = r['frame_path'].replace("data/clips", "/clips")
            relative_path = relative_path.replace("\\", "/") # Fix Windows paths
            r['url'] = relative_path
            
        REQUEST_COUNT.labels(method="GET", endpoint="/search", http_status=200).inc()
        REQUEST_LATENCY.labels(endpoint="/search").observe(time.time() - start_time)
            
        return {
            "status": "success" if results else "abstained",
            "results": results
        }
    except Exception as e:
        REQUEST_COUNT.labels(method="GET", endpoint="/search", http_status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)