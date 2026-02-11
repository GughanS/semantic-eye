from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
from processing import VideoPipeline
from search_engine import SearchEngine

app = FastAPI(title="Semantic Eye: Event Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the new Architecture
pipeline = VideoPipeline() # Handles Optical Flow & Temporal Units
engine = SearchEngine()    # Handles CLIP Ranking

app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")
app.mount("/clips", StaticFiles(directory="data/clips"), name="clips")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())[:8]
    file_location = f"{pipeline.upload_dir}/{video_id}_{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print(f"Pipeline Started: {video_id}")
        
        # 1. Run Pipeline (Extract & Motion Gate)
        verified_events = pipeline.process_video(file_location, video_id)
        
        # 2. Index only if we have events
        if verified_events:
            engine.index_verified_events(verified_events)
            status = "success"
        else:
            status = "abstained_no_motion"
        
        return {
            "status": status, 
            "video_id": video_id, 
            "video_url": f"/videos/{video_id}_{file.filename}",
            "events_verified": len(verified_events)
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_video(query: str):
    try:
        results = engine.search(query)
        
        # Modify paths to be accessible via URL
        # Note: We now serve from /clips, not /frames
        for r in results:
            relative_path = r['frame_path'].replace("data/clips", "/clips")
            relative_path = relative_path.replace("\\", "/") # Fix Windows paths
            r['url'] = relative_path
            
        return {
            "status": "success" if results else "abstained",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Using port 8002 as per your previous setup
    uvicorn.run(app, host="0.0.0.0", port=8002)