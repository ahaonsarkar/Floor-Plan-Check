"""
Room type classifier for the Automated 2D Floor Planning Evaluation system.

This module builds a Scikit‑Learn pipeline that uses a RandomForest model to
predict the functional type of a room (e.g., bedroom, bathroom, kitchen, …)
based on geometric and contextual features.

Features currently supported:
    - area (pixel²) and normalized area (relative to image size)
    - perimeter
    - aspect ratio (width / height)
    - number of doors intersecting the room mask
    - number of windows intersecting the room mask
    - room centroid (x, y) normalised by image dimensions (provides location)
    - count of adjacent rooms (rooms that share a wall segment)

The module provides:
    * ``RoomClassifier`` – encapsulates model loading, feature extraction, and
      prediction.
    * ``train_and_save`` – training pipeline that fits a RandomForest on a
      supplied dataset and persists the model to disk.
    * ``extract_features`` – a pure function that can be reused for both the
      training pipeline and runtime predictions.

Typical workflow:

    >>> from room_classifier import RoomClassifier, train_and_save
    >>> # 1️⃣  Train a model (once) using a CSV or in‑memory data matrix
    >>> train_and_save(train_X, train_y, "models/room_classifier.joblib")
    >>> # 2️⃣  Load the trained model for inference
    >>> clf = RoomClassifier("models/room_classifier.joblib")
    >>> room_type = clf.predict_one(room_mask, doors, windows, image_shape, adjacency_map)

All code is written with type hints, logging, and explicit error handling to
meet production‑grade standards.
"""

from __future__ import annotations

import cv2
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging configuration – callers can adjust the level globally.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------
Coord = Tuple[int, int]
FeatureDict = Dict[str, float]
RoomMask = np.ndarray  # binary mask (0/255) of a single room
AdjacencyMap = Dict[int, List[int]]  # room_id → list of neighbouring room IDs

# ---------------------------------------------------------------------------
# Helper dataclass for a single room's contextual information
# ---------------------------------------------------------------------------
@dataclass
class RoomContext:
    mask: RoomMask
    doors: List[Coord]  # list of door centroids intersecting the room
    windows: List[Coord]  # list of window centroids intersecting the room
    image_shape: Tuple[int, int]  # (height, width) of the original binary image
    room_id: int
    adjacency: List[int]  # IDs of adjacent rooms

# ---------------------------------------------------------------------------
# Feature extraction logic – pure function so it can be unit‑tested easily.
# ---------------------------------------------------------------------------
def extract_features(context: RoomContext) -> FeatureDict:
    """Compute a feature vector for a single room.

    The following numeric features are produced:
        * ``area`` – raw pixel area of the room mask.
        * ``area_norm`` – area normalised by total image area.
        * ``perimeter`` – contour perimeter.
        * ``aspect_ratio`` – width / height of the bounding rectangle.
        * ``door_count`` – number of doors intersecting the room.
        * ``window_count`` – number of windows intersecting the room.
        * ``centroid_x`` / ``centroid_y`` – normalised room centroid.
        * ``adjacent_count`` – how many distinct rooms share a wall.

    Args:
        context: A ``RoomContext`` instance containing the binary mask and
                 auxiliary information required for feature calculation.

    Returns:
        A plain ``dict`` mapping feature names to ``float`` values.
    """
    if context.mask.ndim != 2:
        raise ValueError("Room mask must be a 2‑D array")

    img_h, img_w = context.image_shape
    total_img_area = img_h * img_w

    # --- Geometric properties ------------------------------------------------
    # Area (count of non‑zero pixels)
    area = int(np.count_nonzero(context.mask))
    area_norm = area / total_img_area if total_img_area else 0.0

    # Find the external contour of the room mask to compute perimeter & bbox
    contours, _ = cv2.findContours(
        context.mask.astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("No contour found for room mask (room_id=%s)" % context.room_id)
    contour = max(contours, key=cv2.contourArea)  # biggest external contour
    perimeter = cv2.arcLength(contour, closed=True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0.0

    # --- Door / window statistics --------------------------------------------
    door_count = len(context.doors)
    window_count = len(context.windows)

    # --- Spatial location ------------------------------------------------------
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0.0, 0.0
    centroid_x = cx / img_w if img_w else 0.0
    centroid_y = cy / img_h if img_h else 0.0

    # --- Adjacency ------------------------------------------------------------
    adjacent_count = len(set(context.adjacency))

    features: FeatureDict = {
        "area": float(area),
        "area_norm": float(area_norm),
        "perimeter": float(perimeter),
        "aspect_ratio": float(aspect_ratio),
        "door_count": float(door_count),
        "window_count": float(window_count),
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "adjacent_count": float(adjacent_count),
    }
    log.debug("Features for room %s: %s", context.room_id, features)
    return features

# ---------------------------------------------------------------------------
# Core classifier wrapper
# ---------------------------------------------------------------------------
class RoomClassifier:
    """Encapsulates a trained RandomForest model for room type prediction.

    The model is persisted with ``joblib``.  Upon instantiation the model is
    loaded from ``model_path``; an ``IOError`` is raised if the file does not
    exist or cannot be deserialized.
    """

    VALID_LABELS = {"Bedroom", "Bathroom", "Kitchen", "Living room", "Dining room"}

    def __init__(self, model_path: str | pathlib.Path):
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.pipeline: Pipeline = joblib.load(self.model_path)
        log.info("RoomClassifier loaded model from %s", self.model_path)

    # ---------------------------------------------------------------------
    # Public API – single‑room prediction
    # ---------------------------------------------------------------------
    def predict_one(
        self,
        mask: RoomMask,
        doors: List[Coord],
        windows: List[Coord],
        image_shape: Tuple[int, int],
        room_id: int = 0,
        adjacency: Optional[List[int]] = None,
    ) -> str:
        """Predict the type of a single room.

        Args:
            mask: Binary mask of the room (same shape as the original image).
            doors: List of door centroids intersecting the room.
            windows: List of window centroids intersecting the room.
            image_shape: (height, width) tuple of the original processed image.
            room_id: Optional identifier used only for logging.
            adjacency: Optional list of neighbouring room IDs.

        Returns:
            Predicted room type as a string.
        """
        ctx = RoomContext(
            mask=mask,
            doors=doors,
            windows=windows,
            image_shape=image_shape,
            room_id=room_id,
            adjacency=adjacency or [],
        )
        feats = extract_features(ctx)
        X = np.array([list(feats.values())])
        pred = self.pipeline.predict(X)[0]
        if pred not in self.VALID_LABELS:
            log.warning("Model predicted unknown label %s for room %s", pred, room_id)
        return str(pred)

    # ---------------------------------------------------------------------
    # Batch prediction – useful when many rooms have been segmented.
    # ---------------------------------------------------------------------
    def predict_batch(self, contexts: List[RoomContext]) -> List[str]:
        """Predict room types for a list of ``RoomContext`` objects.

        Returns a list of predictions in the same order as ``contexts``.
        """
        if not contexts:
            return []
        feature_rows = [list(extract_features(ctx).values()) for ctx in contexts]
        X = np.asarray(feature_rows, dtype=float)
        preds = self.pipeline.predict(X)
        return [str(p) for p in preds]

class RoomTypeClassifier:
    """Heuristic and ML-based room type classifier."""
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            try:
                import joblib
                self.model = joblib.load(model_path)
            except Exception as e:
                log.warning(f"Failed to load ML model: {e}. Using heuristic fallback.")

    def classify(self, room_masks: List[np.ndarray]) -> List[str]:
        """
        Classify rooms using percentile-rank heuristics.

        Key insight: absolute pixel areas are meaningless across different image
        resolutions. Instead we rank rooms by their relative size within the
        floor plan and use that rank + shape features to assign types.

        Rank buckets (approximate):
          - Top 30% area         -> Living room / Dining room
          - 20-70% area          -> Bedroom
          - Bottom 20% area      -> Bathroom / Kitchen
          Aspect ratio and compactness refine within each bucket.
        """
        if not room_masks:
            return []

        # --- Extract geometric features for every room --------------------
        areas, aspect_ratios, compactness_scores = [], [], []
        for mask in room_masks:
            cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                areas.append(0.0)
                aspect_ratios.append(1.0)
                compactness_scores.append(0.0)
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / bh if bh > 0 else 1.0
            perim = cv2.arcLength(cnt, True)
            # Compactness = 4pi*area / perimeter^2  (circle=1, elongated<1)
            comp = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0
            areas.append(area)
            aspect_ratios.append(ar)
            compactness_scores.append(comp)

        areas_arr = np.array(areas, dtype=float)
        n = len(areas_arr)

        # Percentile rank per room (0=smallest, 1=largest)
        ranks = np.zeros(n)
        sorted_idx = np.argsort(areas_arr)
        for rank_pos, room_idx in enumerate(sorted_idx):
            ranks[room_idx] = rank_pos / max(n - 1, 1)

        room_types = []
        for i in range(n):
            rank = ranks[i]
            ar = aspect_ratios[i]
            comp = compactness_scores[i]
            area = areas_arr[i]

            if area < 100:
                room_types.append("Bathroom")
                continue

            if rank >= 0.70:
                # Largest rooms: Living room or Dining room
                # Dining is typically squarish; living is wider
                if comp > 0.65 and 0.7 <= ar <= 1.5:
                    room_types.append("Dining room")
                else:
                    room_types.append("Living room")

            elif rank >= 0.35:
                # Mid-size rooms: Bedroom or Kitchen
                # Kitchen tends to be squarish and compact
                if comp > 0.70 and 0.75 <= ar <= 1.35:
                    room_types.append("Kitchen")
                else:
                    room_types.append("Bedroom")

            else:
                # Smallest rooms: Bathroom (very small/compact) or Kitchen
                if comp > 0.72 and 0.75 <= ar <= 1.35 and rank >= 0.15:
                    room_types.append("Kitchen")
                else:
                    room_types.append("Bathroom")

        return room_types


# ---------------------------------------------------------------------------
# Training utilities – intended for an offline data‑science step.
# ---------------------------------------------------------------------------
def _build_pipeline() -> Pipeline:
    """Construct a Scikit‑Learn pipeline for the RandomForest classifier.

    The pipeline consists of a ``StandardScaler`` (centering & scaling) followed
    by a ``RandomForestClassifier`` with sensible default hyper‑parameters.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_and_save(
    X: List[List[float]],
    y: List[str],
    model_path: str | pathlib.Path,
) -> None:
    """Fit a RandomForest model on ``X`` / ``y`` and persist it.

    Args:
        X: Feature matrix where each inner list follows the ordering of feature
           names returned by :func:`extract_features`.
        y: Target labels – must be one of the supported room types.
        model_path: Destination path for the serialized ``joblib`` pipeline.
    """
    if len(X) != len(y):
        raise ValueError("Feature matrix length and label list must match")
    if not X:
        raise ValueError("Empty training data provided")

    pipeline = _build_pipeline()
    log.info("Training RandomForest on %d samples", len(X))
    pipeline.fit(X, y)

    # Persist the pipeline
    model_path = pathlib.Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    log.info("Model persisted to %s", model_path)

# ---------------------------------------------------------------------------
# Utility for converting a list of ``RoomContext`` objects into a feature matrix
# ---------------------------------------------------------------------------
def contexts_to_feature_matrix(contexts: List[RoomContext]) -> Tuple[np.ndarray, List[str]]:
    """Transform a collection of ``RoomContext`` objects into X, y arrays.

    The function assumes each ``RoomContext`` already contains a ``label``
    attribute on the instance (added dynamically by the data‑generation script).
    It returns the matrix ``X`` and the corresponding label list ``y``.
    """
    X: List[List[float]] = []
    y: List[str] = []
    for ctx in contexts:
        # ``label`` is expected to be present for training data only.
        label = getattr(ctx, "label", None)
        if label is None:
            raise AttributeError("RoomContext missing 'label' attribute for training")
        feats = extract_features(ctx)
        X.append(list(feats.values()))
        y.append(str(label))
    return np.asarray(X, dtype=float), y

# ---------------------------------------------------------------------------
# Example CLI for quick local training – not executed in production imports.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import csv
    import sys

    parser = argparse.ArgumentParser(description="Train a room‑type classifier.")
    parser.add_argument(
        "--train-csv",
        type=pathlib.Path,
        required=True,
        help="Path to a CSV file containing training data. The first column must be 'label', followed by feature columns in the order produced by `extract_features`.",
    )
    parser.add_argument(
        "--model-output",
        type=pathlib.Path,
        default=pathlib.Path("models/room_classifier.joblib"),
        help="Destination for the trained model.",
    )
    args = parser.parse_args()

    if not args.train_csv.is_file():
        log.error("Training CSV not found: %s", args.train_csv)
        sys.exit(1)

    # Load CSV – expecting a header row
    with args.train_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        feature_names = [fn for fn in reader.fieldnames if fn != "label"]
        X: List[List[float]] = []
        y: List[str] = []
        for row in reader:
            y.append(row["label"])
            X.append([float(row[fn]) for fn in feature_names])

    train_and_save(X, y, args.model_output)
    log.info("Training complete. Model saved to %s", args.model_output)
