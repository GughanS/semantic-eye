import lancedb
from sentence_transformers import SentenceTransformer
from PIL import Image
import pyarrow as pa
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Constants
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

VECTOR_DIM      = 768
DEFAULT_DB_PATH = "data/lancedb"
TABLE_NAME      = "gods_eye_events"
MODEL_ID        = "clip-ViT-L-14"

# Semantic must dominate; motion is a tiebreaker only.
W_SEMANTIC       = 0.85
W_MOTION         = 0.15
FLOOR_PERCENTILE = 15

# ΓöÇΓöÇ CLIP Prompt Ensembling ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
QUERY_TEMPLATES = [
    "a photo of {}",
    "a video frame showing {}",
    "a surveillance camera image of {}",
    "an image captured on CCTV of {}",
    "a scene depicting {}",
]

# ΓöÇΓöÇ Absolute Similarity Floor ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Empirical ViT-B/32 cosine similarity ranges:
#   Unrelated content (e.g. "car crash" vs pedestrian street): ~0.21ΓÇô0.24
#   Loosely related:                                            ~0.24ΓÇô0.27
#   Genuinely matching:                                         ~0.27+
# The absolute floor is Gate 1: hard-reject anything in the noise band.
# The dynamic floor is Gate 2: among survivors, reject the weakest percentile.
# Gate 2 is computed on Gate 1 survivors only ΓÇö so it trims the bottom of
# *genuine* candidates rather than wasting percentile mass on noise.
ABSOLUTE_SEM_FLOOR = 0.28

# ΓöÇΓöÇ Motion Normalisation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# event_probability is raw motion magnitude, not a bounded probability.
# Clamp then re-scale to [0,1] before weighting.
MOTION_CLAMP_MAX = 3.0

# ΓöÇΓöÇ Negative-Query Anchor ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
NEGATIVE_ANCHOR_TEMPLATE = "a photo of a completely unrelated scene with no {}"


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# SearchEngine
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class SearchEngine:
    def __init__(
        self,
        db_path:            str   = DEFAULT_DB_PATH,
        cache_dir:          str   = None,
        sem_weight:         float = W_SEMANTIC,
        motion_weight:      float = W_MOTION,
        floor_percentile:   int   = FLOOR_PERCENTILE,
        absolute_sem_floor: float = ABSOLUTE_SEM_FLOOR,
    ):
        if not np.isclose(sem_weight + motion_weight, 1.0):
            raise ValueError("sem_weight + motion_weight must equal 1.0")

        self.sem_weight         = sem_weight
        self.motion_weight      = motion_weight
        self.floor_percentile   = floor_percentile
        self.absolute_sem_floor = absolute_sem_floor

        if cache_dir is None:
            self.cache_dir = os.getenv("MODEL_CACHE_DIR", "model_cache")
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Loading CLIP model ({MODEL_ID}) into {self.cache_dir} ΓÇª")
        self.model = SentenceTransformer(MODEL_ID, cache_folder=self.cache_dir)

        self.db = lancedb.connect(db_path)
        self._init_table()

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Table setup
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _init_table(self) -> None:
        schema = pa.schema([
            pa.field("vector",            pa.list_(pa.float32(), VECTOR_DIM)),
            pa.field("video_id",          pa.string()),
            pa.field("timestamp",         pa.float64()),
            pa.field("frame_path",        pa.string()),
            pa.field("display_time",      pa.string()),
            pa.field("event_probability", pa.float32()),
        ])
        try:
            self.table = self.db.open_table(TABLE_NAME)
            logger.info("Opened existing LanceDB table.")
        except Exception:
            logger.info("No existing table ΓÇô creating a new one.")
            self.table = self.db.create_table(TABLE_NAME, schema=schema, mode="overwrite")

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Encoding helpers
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _encode_text_ensembled(self, text: str) -> np.ndarray:
        prompts  = [t.format(text) for t in QUERY_TEMPLATES]
        vecs     = self.model.encode(prompts, normalize_embeddings=True)
        mean_vec = vecs.mean(axis=0)
        return mean_vec / np.linalg.norm(mean_vec)

    def _encode_negative(self, text: str) -> np.ndarray:
        prompt = NEGATIVE_ANCHOR_TEMPLATE.format(text)
        return self.model.encode(prompt, normalize_embeddings=True)

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Indexing
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def index_verified_events(self, events_metadata: list[dict]) -> None:
        valid_events = [e for e in events_metadata if Path(e["frame_path"]).exists()]
        skipped = len(events_metadata) - len(valid_events)
        if skipped:
            logger.warning(f"Skipped {skipped} event(s) ΓÇô frame path not found.")
        if not valid_events:
            logger.warning("No valid events to index.")
            return

        logger.info(f"Encoding {len(valid_events)} frames ΓÇª")
        images  = [Image.open(e["frame_path"]).convert("RGB") for e in valid_events]
        vectors = self.model.encode(
            images,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(images) > 10,
        )

        rows = [
            {
                "vector":            vec.tolist(),
                "video_id":          str(e["video_id"]),
                "timestamp":         float(e["timestamp"]),
                "frame_path":        str(e["frame_path"]),
                "display_time":      str(e["display_time"]),
                "event_probability": float(e["event_probability"]),
            }
            for e, vec in zip(valid_events, vectors)
        ]
        self.table.add(rows)
        logger.info(f"Indexed {len(rows)} events.")

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Scoring helpers
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _cosine_from_distance(self, distance: float) -> float:
        """LanceDB cosine metric stores (1 ΓêÆ cosine_sim) as _distance."""
        return 1.0 - max(0.0, min(2.0, float(distance)))

    def _fused_score(self, sem: float, motion_raw: float) -> float:
        sem_norm    = (sem + 1.0) / 2.0
        motion_norm = min(motion_raw, MOTION_CLAMP_MAX) / MOTION_CLAMP_MAX
        return self.sem_weight * sem_norm + self.motion_weight * motion_norm

    def _dynamic_floor(self, sem_scores: list[float]) -> float:
        """
        Percentile floor computed on the provided scores.
        Always called on Gate-1 survivors (absolute-floor passes) so the
        percentile is meaningful ΓÇö it cuts the weakest genuine candidates,
        not noise that Gate 1 already handles.
        """
        if not sem_scores:
            return -1.0
        return float(np.percentile(sem_scores, self.floor_percentile))

    def _is_false_positive(
        self,
        frame_vec: np.ndarray,
        pos_vec:   np.ndarray,
        neg_vec:   np.ndarray,
        margin:    float = 0.01,
    ) -> bool:
        return float(np.dot(frame_vec, neg_vec)) > float(np.dot(frame_vec, pos_vec)) + margin

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Search
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def search(
        self,
        text_query: str,
        limit:      int           = 5,
        fetch_k:    Optional[int] = None,
    ) -> list[dict]:
        """
        Three-gate filtering pipeline:

          Gate 1 ΓÇô Absolute floor  (ABSOLUTE_SEM_FLOOR = 0.27)
            Hard-rejects anything in the noise band (~0.21ΓÇô0.26).
            Fast: no vector arithmetic needed.

          Gate 2 ΓÇô Dynamic percentile floor  (p15 of Gate-1 survivors)
            Computed exclusively on candidates that passed Gate 1, so the
            percentile reflects the distribution of *genuine* candidates,
            not noise.  Trims the weakest 15% of real matches ΓÇö useful when
            many frames are genuinely relevant and we want only the strongest.

          Gate 3 ΓÇô Negative-query rejection
            Re-scores each survivor against a negative anchor.  Catches
            edge-case false positives that share incidental visual features
            with the query (e.g. an open road resembling a crash scene).
        """
        fetch_k = fetch_k or max(limit * 3, 20)
        logger.info(f'Searching: "{text_query}"')

        # Encode ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        pos_vec = self._encode_text_ensembled(text_query)
        neg_vec = self._encode_negative(text_query)

        # Retrieve ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        try:
            df = (
                self.table
                .search(pos_vec.tolist())
                .metric("cosine")
                .limit(fetch_k)
                .to_pandas()
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        if df.empty:
            logger.warning("DB returned no candidates.")
            return []

        # Pre-compute all semantic scores ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        all_sem = [
            self._cosine_from_distance(row.get("_distance", 1.0))
            for _, row in df.iterrows()
        ]

        # Gate 1: absolute floor ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        gate1_survivors = [
            (i, row) for (i, (_, row)) in enumerate(df.iterrows())
            if all_sem[i] >= self.absolute_sem_floor
        ]
        n_rejected_g1 = len(df) - len(gate1_survivors)
        if n_rejected_g1:
            logger.debug(f"  Gate 1 rejected {n_rejected_g1} candidate(s) below absolute floor ({self.absolute_sem_floor})")

        if not gate1_survivors:
            logger.info(
                f'  Γ¢ö Abstained: all candidates below absolute floor. '
                f'"{text_query}" does not appear to be present in this video.'
            )
            return []

        # Gate 2: dynamic percentile floor (on Gate-1 survivors only) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        survivor_sem    = [all_sem[i] for i, _ in gate1_survivors]
        dynamic_floor   = self._dynamic_floor(survivor_sem)
        gate2_survivors = [
            (i, row) for (i, row) in gate1_survivors
            if all_sem[i] >= dynamic_floor
        ]
        n_rejected_g2 = len(gate1_survivors) - len(gate2_survivors)
        if n_rejected_g2:
            logger.debug(f"  Gate 2 rejected {n_rejected_g2} candidate(s) below dynamic floor ({dynamic_floor:.4f})")

        # Gate 3: negative-query rejection + fused scoring ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        results = []
        for i, row in gate2_survivors:
            sem  = all_sem[i]
            phys = float(row["event_probability"])
            t    = row["display_time"]

            frame_vec = np.array(row["vector"], dtype=np.float32)
            frame_vec = frame_vec / (np.linalg.norm(frame_vec) + 1e-9)
            if self._is_false_positive(frame_vec, pos_vec, neg_vec):
                logger.debug(f"  Gate 3 rejected {t}  sem={sem:.4f} ΓÇô closer to negative anchor")
                continue

            fused = self._fused_score(sem, phys)
            logger.info(f"  ACCEPT {t}  sem={sem:.4f}  motion={phys:.2f}  fused={fused:.3f}")
            results.append({
                "video_id":     str(row["video_id"]),
                "timestamp":    float(row["timestamp"]),
                "frame_path":   str(row["frame_path"]),
                "display_time": t,
                "score":        round(fused * 100, 1),
                "sem_score":    round(sem, 4),
                "motion_conf":  round(phys, 4),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:limit]

        if not results:
            logger.info(
                f'  Γ¢ö Abstained: no indexed frames genuinely match '
                f'"{text_query}". Content may not exist in this video.'
            )
        return results
