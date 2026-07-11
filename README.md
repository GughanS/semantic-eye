# **Semantic Eye: Video Search Architecture (Professional Edition)**

Semantic Eye is a multimodal AI system that allows users to search within video footage using natural language queries (e.g., "Find the moment a car appears" or "A person walking").

This project has been heavily refactored to align with **Product-Based Engineering Standards**, moving away from black-box LLMs (like CLIP) in favor of traditional Machine Learning (YOLOv8) with robust MLOps, CI/CD, and Observability.

## **Key Features**

**Verified Event Pipeline:** Combines Optical Flow (Motion Physics) with YOLOv8 (Object Detection) to eliminate false positives on static scenes.

**Search Engine:** Replaced LanceDB embeddings with relational keyword matching (PostgreSQL) against YOLO-extracted tags.

**MLOps Ready:** Includes MLflow integration for model metrics tracking, and Prometheus/Grafana for API observability.

**Containerized & Scalable:** Fully Dockerized architecture via Docker Compose and multi-stage builds. CI/CD pipeline included for GitHub Actions.

## **Tech Stack**

**Frontend:**
- Framework: React (Vite) served via Nginx
- Styling: CSS Modules (Monolith Theme)

**Backend:**
- API: FastAPI (Python)
- ML Model: Ultralytics YOLOv8 (Object Detection, NO LLMs)
- DB: PostgreSQL (SQLAlchemy)
- Video Processing: OpenCV (cv2) + NumPy
- Motion Logic: Dense Optical Flow (Farneback)

**Observability & MLOps:**
- MLflow: Tracks YOLO inference parameters and metrics
- Prometheus & Grafana: Tracks FastAPI endpoint health, latency, and throughput

-------------------

## **Project Structure**

```
/semantic-eye
├── backend/
│   ├── main.py            # FastAPI Entry Point (w/ Prometheus metrics)
│   ├── processing.py      # Motion Gate & Video Slicing
│   ├── search_engine.py   # YOLOv8 & PostgreSQL Logic
│   ├── requirements.txt   # Python Dependencies
│   └── Dockerfile         # Multi-stage Docker build
├── frontend/
│   ├── src/               # React UI Logic
│   └── Dockerfile         # Nginx production build
├── .github/workflows/
│   └── ci-cd.yml          # GitHub Actions Pipeline
├── docker-compose.yml     # Orchestration (App, DB, MLflow, Grafana, Prometheus)
└── prometheus.yml         # Prometheus scrape config
```

## **Installation & Setup**

### **Running with Docker Compose (Recommended)**

The fastest way to spin up the entire cluster (Frontend, Backend, Database, and Observability tools).

```bash
docker-compose up --build
```

**Services will be available at:**
- **Frontend Dashboard:** `http://localhost:5173`
- **Backend API:** `http://localhost:8002`
- **MLflow Tracking:** `http://localhost:5000`
- **Grafana Dashboards:** `http://localhost:3000`
- **Prometheus UI:** `http://localhost:9090`

### **Usage Guide**

**Upload:** Click "Import Footage" to select a video file (MP4/MKV).

**Process:** Click "Run Pipeline". 

The system will:
1. Slice video into temporal units.
2. Calculate Optical Flow energy.
3. Generate Object Detection Tags (via YOLOv8) for valid events.
4. Index metadata into PostgreSQL.

**Search:** Type a query like "Person" or "Car".

## **Architecture Details**

The "Verified Event" Protocol

**Temporal Slicing:** Video is divided into 16-frame overlap windows.

**Motion Gate:** Every window is checked for motion energy. Energy < 0.5 is discarded immediately.

**Semantic Tagging (YOLOv8):** Surviving windows are passed to YOLOv8. Detected object classes (e.g., "car", "person") are saved as tags to the database.

**Search Scoring:** Search relies on textual matching against tags, combined with motion confidence scoring, producing a highly verifiable, deterministic result without LLM hallucination.

## **License**

MIT License. Built for educational and research purposes.