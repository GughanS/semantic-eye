<div align="center">
  <h1>Semantic Eye</h1>
  <p><strong>Intelligent Video Search Architecture</strong></p>

  <p>
    <a href="https://github.com/GughanS/semantic-eye/actions"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/GughanS/semantic-eye/ci-cd.yml?style=for-the-badge&logo=github"></a>
    <a href="https://github.com/GughanS/semantic-eye/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/GughanS/semantic-eye?style=for-the-badge"></a>
    <a href="https://fastapi.tiangolo.com"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"></a>
    <a href="https://react.dev"><img alt="React" src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB"></a>
    <a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white"></a>
  </p>
</div>

---

## Overview

**Semantic Eye** is a multimodal AI system that allows users to search within video footage using natural language queries (e.g., _"Find the moment a car appears"_ or _"A person walking"_).

Built with **Product-Based Engineering Standards**, this architecture moves away from unpredictable black-box LLMs (like CLIP) in favor of highly deterministic traditional Machine Learning (YOLOv8). It comes fully equipped with robust **MLOps, CI/CD pipelines, and Observability**.

## Key Features

- **Verified Event Pipeline**: Combines **Dense Optical Flow** (Motion Physics) with **YOLOv8** (Object Detection) to virtually eliminate false positives in static scenes.
- **High-Speed Search Engine**: Leverages relational keyword matching in **PostgreSQL** against YOLO-extracted tags instead of expensive vector embeddings.
- **MLOps & Observability Ready**: Fully integrated with **MLflow** for model metric tracking, and **Prometheus/Grafana** for deep API observability.
- **Containerized & Scalable**: A complete multi-container Docker architecture via `docker-compose`. Includes multi-stage builds and automated **CI/CD via GitHub Actions**.

---

## Tech Stack

### Frontend
- **Framework**: React (Vite)
- **Deployment**: Served via Nginx
- **Styling**: CSS Modules (Monolith Theme)

### Backend
- **API**: FastAPI (Python)
- **ML Model**: Ultralytics YOLOv8 (Object Detection)
- **Database**: PostgreSQL (SQLAlchemy ORM)
- **Video Processing**: OpenCV (`cv2`) & NumPy
- **Motion Logic**: Dense Optical Flow (Farneback Algorithm)

### Observability & MLOps
- **MLflow**: Tracks YOLO inference parameters, speeds, and confidence metrics.
- **Prometheus & Grafana**: Monitors FastAPI endpoint health, latency, throughput, and system resources.

---

## Project Structure

```text
semantic-eye/
├── .github/workflows/   # CI/CD Pipelines (GitHub Actions)
├── backend/
│   ├── main.py          # FastAPI Entry Point (w/ Prometheus metrics)
│   ├── processing.py    # Motion Gate & Video Slicing Logic
│   ├── search_engine.py # YOLOv8 & PostgreSQL Integration
│   ├── requirements.txt # Python Dependencies
│   └── Dockerfile       # Multi-stage Backend Docker Build
├── frontend/
│   ├── src/             # React UI Components
│   └── Dockerfile       # Nginx Production Build
├── docker-compose.yml   # Orchestration (App, DB, MLflow, Grafana, Prometheus)
└── prometheus.yml       # Prometheus Scrape Configuration
```

---

## Installation & Setup

### Running with Docker Compose (Recommended)

The fastest and most reliable way to spin up the entire cluster (Frontend, Backend, Database, and Observability stack).

1. Clone the repository:
   ```bash
   git clone https://github.com/GughanS/semantic-eye.git
   cd semantic-eye
   ```

2. Start the cluster:
   ```bash
   docker-compose up --build
   ```

### Service Endpoints

Once the cluster is running, services will be accessible at:
- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs**: [http://localhost:8002/docs](http://localhost:8002/docs)
- **Grafana Dashboards**: [http://localhost:3000](http://localhost:3000)
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

---

## Usage Guide

1. **Upload Footage**: Open the Frontend Dashboard and click **"Import Footage"** to select a video file (`.mp4`, `.mkv`).
2. **Process**: Click **"Run Pipeline"**. The system automatically:
   - Slices the video into temporal units.
   - Calculates Optical Flow energy to isolate motion.
   - Generates Object Detection Tags (via YOLOv8) for valid events.
   - Indexes metadata into PostgreSQL.
3. **Search**: Type a natural language query (e.g., _"Person"_, _"Car"_) in the search bar and instantly view the corresponding video segments.

---

## Architecture Deep Dive: The "Verified Event" Protocol

The core philosophy of Semantic Eye is verifiable, deterministic video understanding:

1. **Temporal Slicing**: The video is divided into rolling 16-frame overlap windows to ensure smooth temporal continuity.
2. **Motion Gate (Optical Flow)**: Every window is checked for motion energy. Windows with an energy score `< 0.5` are discarded immediately, saving massive computational overhead on static frames.
3. **Semantic Tagging (YOLOv8)**: Surviving motion windows are passed to the YOLOv8 model. Detected object classes are saved directly as relational tags.
4. **Search Scoring**: Queries perform textual matching against these deterministic tags combined with motion confidence scoring, resulting in highly verifiable results without the risk of LLM hallucinations.

---

## License

This project is licensed under the **MIT License**. Built for advanced engineering and research purposes.