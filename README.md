# **Semantic Eye: Video Search Architecture**

Semantic Eye is a multimodal AI system that allows users to search within video footage using natural language queries (e.g., "Find the moment the baby smiles" or "A red car crash").

Unlike standard vector search engines that hallucinate results, Semantic Eye implements a Verified Event Architecture. It uses a physics-based Motion Gate (Optical Flow) to verify event existence before using Semantic Ranking (CLIP) to identify the content.

## **Key Features**

**Verified Event Pipeline:** Combines Optical Flow (Motion Physics) with CLIP (Semantics) to eliminate false positives on static scenes.

**Zero-Shot Search:** No training required. Search for any object, action, or event immediately after upload.

**Abstention Logic:** The system is engineered to say "I don't know" (System Abstained) rather than returning low-confidence junk results.

**Vector Database:** Powered by LanceDB for ultra-fast, serverless embedding retrieval.

**Cinematic UI:** A "Monolith" dark-themed React dashboard optimized for visual clarity.

## **Tech Stack**

**Frontend:**

Framework: React (Vite)

Styling: CSS Modules (Monolith Theme)

Icons: Lucide React

----------
**Backend:**

API: FastAPI (Python)

AI Model: OpenAI CLIP (ViT-B/32)

Vector DB: LanceDB

Video Processing: OpenCV (cv2) + NumPy

Motion Logic: Dense Optical Flow (Farneback)

-------------------

## **Project Structure**

```
/semantic-eye
├── backend/
│   ├── main.py            # FastAPI Entry Point
│   ├── processing.py      # Motion Gate & Video Slicing
│   ├── search_engine.py   # CLIP & LanceDB Logic
│   └── requirements.txt   # Python Dependencies
├── frontend/
│   ├── src/
│   │   └── App.jsx        # React UI Logic
│   └── package.json       # Node Dependencies
└── data/                  # Local storage for videos/indexes
```

## **Installation & Setup**

**Prerequisites**

*Python 3.10+*

*Node.js 18+*

### **1. Backend Setup**

The backend handles the heavy AI lifting.
```
cd backend

# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (Torch, OpenCV, LanceDB, FastAPI)
pip install -r requirements.txt

# Start the Server
python main.py


The backend will run on http://localhost:8002
```

## **2. Frontend Setup**

The frontend provides the dashboard.
```
cd frontend

# Install libraries
npm install

# Start the Development Server
npm run dev


The frontend will run on http://localhost:5173
```

### **Usage Guide**

**Upload:** Click "Import Footage" to select a video file (MP4/MKV).

**Process:** Click "Run Pipeline". 

The system will:

Slice video into temporal units.

Calculate Optical Flow energy.

Generate CLIP embeddings for valid events.

Index data into LanceDB.

**Search:** Type a query like "Person running" or "Blue truck".

**Results:** High Match: Green badge (High Semantic + High Motion).

**System Abstained:** Red alert (Evidence insufficient).

## **Architecture Details**

The "Verified Event" Protocol

**Temporal Slicing:** Video is divided into 16-frame overlap windows.

**Motion Gate:** Every window is checked for motion energy. Energy < 0.05 is discarded immediately.

**Semantic Ranking:** Surviving windows are embedded via CLIP.

**Calibration:** 

Score > 0.28: High Confidence

Score < 0.18: Hard Rejection (Noise)

## **License**

MIT License. Built for educational and research purposes.