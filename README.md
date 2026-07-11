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

It combines the power of motion detection (Optical Flow) with the deep semantic understanding of **CLIP (ViT-L-14)** to provide highly accurate, verified video search. By filtering out static frames before they ever reach the AI model, the system drastically reduces computational overhead and avoids false positives.

## Key Features

- **Semantic Search Engine**: Utilizes **OpenAI's CLIP model (clip-ViT-L-14)** via `sentence-transformers` for zero-shot image-to-text semantic matching.
- **High-Speed Vector Database**: Uses **LanceDB** to store and query high-dimensional embeddings (768-dim) efficiently.
- **Motion Gating Protocol**: Calculates Dense Optical Flow to filter out static scenes, ensuring the AI only processes frames where actual events occur.
- **Advanced Filtering**: Implements dynamic percentile scoring and negative-query anchoring to dramatically reduce false positives.
- **Containerized & Scalable**: A complete multi-container Docker architecture via `docker-compose`. Includes multi-stage builds and automated CI/CD via GitHub Actions.

---

## Tech Stack

### Frontend
- **Framework**: React (Vite)
- **Deployment**: Served via Nginx
- **Styling**: CSS Modules

### Backend
- **API**: FastAPI (Python)
- **AI Model**: CLIP (ViT-L-14) via `sentence-transformers`
- **Vector DB**: LanceDB
- **Video Processing**: OpenCV (`cv2`), NumPy, Pillow
- **Motion Logic**: Dense Optical Flow (Farneback Algorithm)

### Observability & MLOps
- **MLflow**: Tracks inference metrics and logs.
- **Prometheus & Grafana**: Monitors FastAPI endpoint health, latency, throughput, and system resources.

---

## Project Structure

```text
semantic-eye/
├── .github/workflows/   # CI/CD Pipelines (GitHub Actions)
├── backend/
│   ├── main.py          # FastAPI Entry Point (w/ Prometheus metrics)
│   ├── processing.py    # Motion Gate & Video Slicing Logic
│   ├── search_engine.py # CLIP & LanceDB Integration
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

The fastest and most reliable way to spin up the entire cluster.

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
   - Encodes surviving frames into 768-dimensional vectors using CLIP.
   - Indexes embeddings and metadata into LanceDB.
3. **Search**: Type a natural language query (e.g., _"Person"_, _"Car"_) in the search bar. The engine encodes your text with CLIP and performs a cosine similarity search against the video frames.

---

## Architecture Deep Dive: The "Verified Event" Protocol

The core philosophy of Semantic Eye is robust, verifiable video understanding:

1. **Temporal Slicing**: The video is divided into rolling overlap windows to ensure smooth temporal continuity.
2. **Motion Gate (Optical Flow)**: Every window is checked for motion energy. Windows lacking motion are discarded immediately, saving massive computational overhead.
3. **Semantic Encoding (CLIP)**: Surviving motion windows are passed to CLIP, generating rich semantic vectors.
4. **Three-Gate Search Engine**:
   - **Gate 1**: Hard-rejects anything below an absolute similarity floor.
   - **Gate 2**: A dynamic percentile floor trims the weakest matches among genuine candidates.
   - **Gate 3**: A negative-query anchor rejects false positives that share incidental visual features but lack the subject (e.g., an empty road vs. a car crash).

---

## License

This project is licensed under the **MIT License**. Built for advanced engineering and research purposes.