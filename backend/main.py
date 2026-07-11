from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool
import shutil
import uuid
import os
import time
from processing import VideoPipeline
from search_engine import SearchEngine

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

app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")
app.mount("/clips", StaticFiles(directory="data/clips"), name="clips")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    start_time = time.time()
    video_id = str(uuid.uuid4())[:8]
    # Save to /tmp inside the container to avoid Docker for Windows bind mount IO deadlocks
    file_location = f"/tmp/{video_id}_{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        logger.info(f"Pipeline Started: {video_id} - file saved to {file_location}")
        
        # 1. Run Pipeline (Extract & Motion Gate) in a background thread to prevent blocking Uvicorn
        verified_events = await run_in_threadpool(pipeline.process_video, file_location, video_id)
        
        # Move video from /tmp to static directory so frontend can play it
        final_video_path = f"data/videos/{video_id}_{file.filename}"
        os.makedirs("data/videos", exist_ok=True)
        shutil.move(file_location, final_video_path)
        
        # 2. Index only if we have events
        if verified_events:
            await run_in_threadpool(engine.index_verified_events, verified_events)
            status = "success"
            EVENTS_VERIFIED.inc(len(verified_events))
        else:
            status = "abstained_no_motion"
        
        REQUEST_COUNT.labels(method="POST", endpoint="/upload", http_status=200).inc()
        REQUEST_LATENCY.labels(endpoint="/upload").observe(time.time() - start_time)
        
        return {
            "status": status, 
            "video_id": video_id, 
            "video_url": f"/videos/{video_id}_{file.filename}",
            "events_verified": len(verified_events)
        }
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/upload", http_status=500).inc()
        print(f"Error: {e}")
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