"""
Automated 2D Floor Planning Evaluation API
============================================

FastAPI application for processing floor plans and generating quality assessments.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from evaluation_engine import (
        evaluate_accessibility,
        evaluate_room_adjacency,
        evaluate_space_utilization,
    )
    from processing.door_window_detection import DoorWindowDetector
    from image_processor import ImageProcessor
    from processing.room_segmentation import RoomSegmenter
    from processing.wall_detection import WallDetector
    from room_classifier import RoomTypeClassifier
    from suggestion_engine import ImprovementRecommender
    from door_window_detector import detect_openings
except ImportError:
    from ..evaluation_engine import (
        evaluate_accessibility,
        evaluate_room_adjacency,
        evaluate_space_utilization,
    )
    from ..processing.door_window_detection import DoorWindowDetector
    from ..image_processor import ImageProcessor
    from ..processing.room_segmentation import RoomSegmenter
    from ..processing.wall_detection import WallDetector
    from ..room_classifier import RoomTypeClassifier
    from ..suggestion_engine import ImprovementRecommender
    from ..door_window_detector import detect_openings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize components
wall_detector = WallDetector()
room_segmenter = RoomSegmenter()
door_window_detector = DoorWindowDetector()
room_classifier = RoomTypeClassifier()
improvement_recommender = ImprovementRecommender()
image_processor = ImageProcessor()


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    error: str
    details: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class FloorPlanResult(BaseModel):
    """Standardized floor plan analysis result."""

    job_id: str
    status: str
    processing_time: float
    rooms: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[str]
    visualization: Dict[str, Any]


class JobStatus(BaseModel):
    """Job status response model."""

    job_id: str
    status: str
    progress: int
    message: str


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded file is an image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image file."
        )
    if file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 50MB."
        )


def get_gemini_api_key() -> str | None:
    """Load Gemini API key from environment variables or .env files."""
    import os
    for key_name in ["GEMINI_API_KEY", "VITE_API_KEY"]:
        val = os.environ.get(key_name)
        if val:
            return val
    # Search common .env locations relative to CWD
    for path in [".env", "../.env", "../../.env", "Backend/.env"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        for key_name in ["GEMINI_API_KEY", "VITE_API_KEY"]:
                            if line.startswith(f"{key_name}="):
                                return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
    return None


def _normalize_label(label: str) -> str | None:
    """Map any room label string to one of our canonical room type names."""
    if not label:
        return None
    l = label.lower().strip()
    # Order matters — check bath before bed to avoid 'bathroom' matching 'bed'
    if any(x in l for x in ["bath", "toilet", "wc", "powder", "restroom", "washroom", "ensuite", "lavatory"]):
        return "Bathroom"
    if any(x in l for x in ["kitchen", "kit.", "cook", "scullery", "pantry"]):
        return "Kitchen"
    if any(x in l for x in ["bed", "chamber", "master", "guest", " br ", "br."]):
        return "Bedroom"
    if any(x in l for x in ["living", "lounge", "family", "drawing", "sitting", "salon", "tv room", "reception", "parlour"]):
        return "Living room"
    if any(x in l for x in ["dining", "dinette", "eating", "din."]):
        return "Dining room"
    if any(x in l for x in ["hall", "corridor", "lobby", "entrance", "foyer", "passage"]):
        return "Living room"  # treat hallways/lobby as living-adjacent
    return None


def classify_rooms_with_gemini(
    image: np.ndarray,
    room_masks: List[np.ndarray],
    room_types: List[str],
) -> List[str]:
    """
    Use Gemini Vision to read every text label in the floor plan image.
    Strategy:
      1. Send the full image to Gemini and ask it to return ALL visible room
         text labels along with the approximate pixel center of each label.
      2. For every detected label, find which segmented room mask contains
         (or is nearest to) that pixel center.
      3. Override the heuristic room type with the OCR-derived label.
    Falls back gracefully to heuristic types if the API call fails.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.info("No Gemini API key found — using heuristic room classification only.")
        return room_types

    try:
        import google.generativeai as genai
        import json
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        genai.configure(api_key=api_key)

        img_h, img_w = image.shape[:2]

        # Encode to JPEG for the multimodal request
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            logger.warning("Could not encode image for Gemini; falling back to heuristics.")
            return room_types
        img_bytes = encoded.tobytes()

        prompt = (
            "You are an expert at reading architectural floor plan drawings.\n"
            f"This floor plan image is {img_w} pixels wide and {img_h} pixels tall.\n\n"
            "TASK: Find every room label (text) written anywhere in this floor plan — "
            "inside rooms, near walls, or as annotations. Common labels include: "
            "'Living Room', 'Bedroom', 'Master Bedroom', 'Bathroom', 'Kitchen', "
            "'Dining Room', 'Hallway', 'Study', 'Toilet', 'WC', 'Lounge', etc.\n\n"
            "For each label you find, return:\n"
            "  - label: the exact text you read\n"
            "  - cx: approximate x pixel coordinate of the CENTER of the label text\n"
            "  - cy: approximate y pixel coordinate of the CENTER of the label text\n\n"
            "IMPORTANT RULES:\n"
            "- Only return labels that are ACTUALLY visible as text in the image.\n"
            "- Do NOT guess or invent labels that are not written.\n"
            "- If there is NO text at all in the image, return an empty array [].\n"
            "- Coordinates must be within the image bounds (0 to width/height).\n\n"
            "Respond ONLY with a valid JSON array. Example:\n"
            "[\n"
            "  {\"label\": \"Master Bedroom\", \"cx\": 320, \"cy\": 150},\n"
            "  {\"label\": \"Bathroom\", \"cx\": 480, \"cy\": 90},\n"
            "  {\"label\": \"Kitchen\", \"cx\": 200, \"cy\": 300}\n"
            "]\n"
            "No markdown, no backticks, no extra text. Just the raw JSON array."
        )

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"temperature": 0.0},
        )
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_bytes},
            prompt,
        ])

        raw = response.text.strip()
        # Strip any markdown code fences Gemini might add
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]).strip()

        detections = json.loads(raw)  # List of {label, cx, cy}
        logger.info(f"Gemini OCR found {len(detections)} text labels in the floor plan.")

        if not detections:
            return room_types

        # Precompute binary masks and centroids for fast lookup
        binary_masks = [m.astype(np.uint8) for m in room_masks]
        centroids = []
        for mask in binary_masks:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                centroids.append((float(xs.mean()), float(ys.mean())))
            else:
                centroids.append((0.0, 0.0))

        updated_types = list(room_types)
        # Track confidence: distance of the winning detection for each room
        # Lower distance = higher confidence. Start at infinity (no winner yet).
        confidence = [float("inf")] * len(room_types)

        for det in detections:
            raw_label = str(det.get("label", "")).strip()
            cx = int(det.get("cx", -1))
            cy = int(det.get("cy", -1))

            canonical = _normalize_label(raw_label)
            if not canonical:
                logger.debug(f"Skipping unrecognised label: '{raw_label}'")
                continue

            # Clamp to image bounds
            cx = max(0, min(cx, img_w - 1))
            cy = max(0, min(cy, img_h - 1))

            # Multi-point sampling: test label center + 8 neighbours (±offset)
            # This handles Gemini returning coordinates slightly off-center
            offset = max(8, int(min(img_w, img_h) * 0.01))
            sample_points = [
                (cx, cy),
                (cx - offset, cy), (cx + offset, cy),
                (cx, cy - offset), (cx, cy + offset),
                (cx - offset, cy - offset), (cx + offset, cy - offset),
                (cx - offset, cy + offset), (cx + offset, cy + offset),
            ]
            sample_points = [
                (max(0, min(px, img_w - 1)), max(0, min(py, img_h - 1)))
                for px, py in sample_points
            ]

            # Step 1: Direct hit — any sample point inside a mask
            matched_room = None
            best_dist = float("inf")
            for px, py in sample_points:
                for r_idx, mask in enumerate(binary_masks):
                    if mask[py, px] > 0:
                        # Distance from label center to this room's centroid
                        rcx, rcy = centroids[r_idx]
                        dist = (rcx - cx) ** 2 + (rcy - cy) ** 2
                        if matched_room is None or dist < best_dist:
                            matched_room = r_idx
                            best_dist = dist

            # Step 2: Nearest centroid fallback if no direct hit
            if matched_room is None:
                for r_idx, (rcx, rcy) in enumerate(centroids):
                    dist = (rcx - cx) ** 2 + (rcy - cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        matched_room = r_idx

            # Step 3: Confidence locking — only overwrite if this match
            # is closer (more confident) than the existing winner
            if matched_room is not None and best_dist < confidence[matched_room]:
                logger.info(
                    f"Assigning room {matched_room} -> '{canonical}' "
                    f"(OCR: '{raw_label}' at ({cx},{cy}), dist={best_dist:.0f})"
                )
                updated_types[matched_room] = canonical
                confidence[matched_room] = best_dist

        logger.info(f"Final room types after Gemini OCR: {updated_types}")
        return updated_types

    except Exception as exc:
        logger.warning(f"Gemini OCR classification failed ({exc}); falling back to heuristics.")
        return room_types


async def process_floor_plan(image_data: bytes) -> Dict[str, Any]:
    """
    Main processing pipeline for floor plan analysis.

    Args:
        image_data: Raw image bytes from upload

    Returns:
        Dictionary containing analysis results
    """
    start_time = time.time()
    job_id = f"job_{int(start_time * 1000)}"

    try:
        # Step 1: Decode image
        logger.info(f"[{job_id}] Decoding image...")
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        # Step 2: Preprocess image
        logger.info(f"[{job_id}] Preprocessing image...")
        preprocessed = image_processor.preprocess(image)

        # Step 3: Detect walls
        logger.info(f"[{job_id}] Detecting walls...")
        walls = wall_detector.detect(preprocessed)

        # Step 4: Segment rooms
        logger.info(f"[{job_id}] Segmenting rooms...")
        room_masks, room_contours = room_segmenter.segment(preprocessed)

        # Step 5: Detect doors and windows
        logger.info(f"[{job_id}] Detecting doors and windows...")
        try:
            # We calculate scale to adjust gap size dynamically
            img_h, img_w = preprocessed.shape[:2]
            scale = (img_h * img_w) / 250000.0
            
            openings = detect_openings(
                preprocessed,
                min_gap_area=max(10.0, 40.0 * scale),
                min_gap_perimeter_ratio=0.1,
                door_min_height=int(max(8, 20 * scale)),
                door_max_aspect=0.8,
                window_max_area_ratio=0.03,
            )
            raw_doors = openings.get("doors", [])
            raw_windows = openings.get("windows", [])
            
            doors = []
            for d_idx, door_poly in enumerate(raw_doors):
                if len(door_poly) > 0:
                    xs = [pt[0] for pt in door_poly]
                    ys = [pt[1] for pt in door_poly]
                    cx = int(sum(xs) / len(xs))
                    cy = int(sum(ys) / len(ys))
                    w_box = max(xs) - min(xs)
                    h_box = max(ys) - min(ys)
                    
                    # Associate door with the closest room
                    closest_room_id = 0
                    min_dist = float('inf')
                    for r_idx, r_mask in enumerate(room_masks):
                        r_ys, r_xs = np.where(r_mask > 0)
                        if len(r_xs) > 0:
                            dist = np.min((r_xs - cx)**2 + (r_ys - cy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_room_id = r_idx
                                
                    doors.append({
                        'id': d_idx,
                        'position': [cx, cy],
                        'type': 'interior_door',
                        'size': {'width': w_box, 'height': h_box},
                        'room_id': closest_room_id
                    })
                    
            windows = []
            for w_idx, win_poly in enumerate(raw_windows):
                if len(win_poly) > 0:
                    xs = [pt[0] for pt in win_poly]
                    ys = [pt[1] for pt in win_poly]
                    cx = int(sum(xs) / len(xs))
                    cy = int(sum(ys) / len(ys))
                    w_box = max(xs) - min(xs)
                    h_box = max(ys) - min(ys)
                    
                    # Associate window with the closest room
                    closest_room_id = 0
                    min_dist = float('inf')
                    for r_idx, r_mask in enumerate(room_masks):
                        r_ys, r_xs = np.where(r_mask > 0)
                        if len(r_xs) > 0:
                            dist = np.min((r_xs - cx)**2 + (r_ys - cy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_room_id = r_idx
                                
                    windows.append({
                        'id': w_idx,
                        'position': [cx, cy],
                        'type': 'window',
                        'size': {'width': w_box, 'height': h_box},
                        'room_id': closest_room_id
                    })
        except Exception as e:
            logger.warning(f"Real door/window detection failed, using fallback: {e}")
            doors, windows = [], []

        # Step 6: Classify rooms
        logger.info(f"[{job_id}] Classifying rooms...")
        room_types = room_classifier.classify(room_masks)
        room_types = classify_rooms_with_gemini(image, room_masks, room_types)

        # Step 7: Build adjacency information
        logger.info(f"[{job_id}] Building adjacency graph...")
        adjacency = build_adjacency(room_masks, walls)

        # Step 8: Evaluate scores
        logger.info(f"[{job_id}] Evaluating scores...")

        # Space utilization
        space_result = evaluate_space_utilization(
            binary_floor=preprocessed,
            room_masks=room_masks,
        )

        # Room adjacency
        adjacency_result = evaluate_room_adjacency(
            room_types=room_types,
            adjacency=adjacency,
        )

        # Accessibility
        accessibility_result = evaluate_accessibility(
            room_masks=room_masks,
            adjacency=adjacency,
        )

        # Calculate overall score
        overall_score = calculate_overall_score(
            space_result, adjacency_result, accessibility_result
        )

        # Step 9: Generate suggestions
        logger.info(f"[{job_id}] Generating improvement suggestions...")
        suggestions = improvement_recommender.generate(
            room_types=room_types,
            adjacency=adjacency,
            space_score=space_result["space_score"],
            adjacency_score=adjacency_result["adjacency_score"],
            accessibility_score=accessibility_result["accessibility_score"],
        )

        # Step 10: Prepare visualization data
        visualization = prepare_visualization(
            image=image,
            room_masks=room_masks,
            walls=walls,
            doors=doors,
            windows=windows,
        )

        processing_time = time.time() - start_time

        return {
            "job_id": job_id,
            "status": "completed",
            "processing_time": processing_time,
            "rooms": format_rooms(room_masks, room_types, room_contours),
            "metrics": {
                "space_utilization": space_result,
                "adjacency": adjacency_result,
                "accessibility": accessibility_result,
                "overall_quality": overall_score,
            },
            "suggestions": suggestions,
            "visualization": visualization,
        }

    except Exception as e:
        logger.error(f"[{job_id}] Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


def build_adjacency(room_masks: List[np.ndarray], walls: List[Dict]) -> Dict[int, List[int]]:
    """Build room adjacency graph based on shared walls."""
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(room_masks))}

    # Simple adjacency based on proximity
    for i in range(len(room_masks)):
        for j in range(i + 1, len(room_masks)):
            # Check if rooms share a wall or are close
            if rooms_adjacent(room_masks[i], room_masks[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def rooms_adjacent(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Check if two room masks are adjacent (share a boundary)."""
    # Dilate both masks slightly
    kernel = np.ones((5, 5), np.uint8)
    dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
    dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1)

    # Check for overlap in dilated masks
    overlap = np.logical_and(dilated1, dilated2)
    return np.count_nonzero(overlap) > 0


def calculate_overall_score(
    space_result: Dict,
    adjacency_result: Dict,
    accessibility_result: Dict,
) -> int:
    """Calculate composite quality score."""
    space_score = space_result.get("space_score", 0)
    adj_score = adjacency_result.get("adjacency_score", 0)
    access_score = accessibility_result.get("accessibility_score", 0)

    # Weighted average
    return int(
        space_score * 0.35 +
        adj_score * 0.35 +
        access_score * 0.30
    )


def format_rooms(
    room_masks: List[np.ndarray],
    room_types: List[str],
    contours: List[np.ndarray],
) -> List[Dict[str, Any]]:
    """Format room data for response."""
    rooms = []
    for i, (mask, room_type, contour) in enumerate(zip(room_masks, room_types, contours)):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rooms.append({
            "id": i,
            "type": room_type,
            "area": int(area),
            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        })
    return rooms


def prepare_visualization(
    image: np.ndarray,
    room_masks: List[np.ndarray],
    walls: List[Dict],
    doors: List[Dict],
    windows: List[Dict],
) -> Dict[str, Any]:
    """Prepare visualization data for frontend."""
    # Create colored room overlay
    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    for i, mask in enumerate(room_masks):
        color = colors[i % len(colors)]
        overlay[mask > 0] = color

    return {
        "image_shape": {"height": h, "width": w},
        "rooms_count": len(room_masks),
        "doors_count": len(doors),
        "windows_count": len(windows),
    }


@router.post("/upload-floor-plan", response_model=FloorPlanResult)
async def upload_floor_plan(file: UploadFile = File(...)):
    """
    Upload and analyze a floor plan image.

    This endpoint accepts an image file (PNG/JPG), processes it through
    the complete analysis pipeline, and returns comprehensive results.

    Args:
        file: Uploaded floor plan image

    Returns:
        FloorPlanResult with rooms, metrics, and suggestions
    """
    logger.info(f"Received upload request for file: {file.filename}")

    # Validate file
    validate_image_file(file)

    # Read image data
    image_data = await file.read()

    # Process floor plan
    result = await process_floor_plan(image_data)

    return result


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


# Export router
__all__ = ["router", "FloorPlanResult", "ErrorResponse", "JobStatus"]