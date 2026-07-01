"""
Room detection module for floor plan analysis.

Detects enclosed spaces (rooms) from a pre‑processed binary floor‑plan image,
identifies their boundaries, and returns labeled coordinates in JSON format.

Typical usage:
    from room_detector import detect_rooms, rooms_to_json
    processed = preprocess_image(...)            # from image_processor
    rooms = detect_rooms(processed)
    json_output = rooms_to_json(rooms)
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np

# ---------------------------------------------------------------------------
# Logging configuration (debug output goes to stderr; can be redirected)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------
Coordinate = Tuple[int, int]
Polygon = List[Coordinate]
RoomLabels = Dict[str, Polygon]


# ---------------------------------------------------------------------------
# Core detection pipeline
# ---------------------------------------------------------------------------
def _find_external_contours(
    binary_image: np.ndarray,
    min_area: float = 150.0,
    min_perimeter_ratio: float = 0.3,
) -> List[np.ndarray]:
    """
    Find external contours in a binary image and filter them by area and
    perimeter ratio (to roughly approximate rectangular rooms).

    Args:
        binary_image: Binary image where room boundaries are white (255)
                     on a black background.
        min_area: Minimum contour area (in pixel²) to be considered.
        min_perimeter_ratio: Minimum ratio of contour perimeter to sqrt(area).
                             Helps reject elongated noise.

    Returns:
        List of filtered contours (each contour is an Nx1x2 array of points).
    """
    log.debug("Finding external contours (min_area=%.0f, min_perimeter_ratio=%.2f)",
              min_area, min_perimeter_ratio)

    contours, hierarchy = cv2.findContours(
        binary_image,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            log.debug("Rejecting contour (area too small): %.0f", area)
            continue

        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        perimeter_to_sqrt_area = perimeter / (area ** 0.5)

        if perimeter_to_sqrt_area < min_perimeter_ratio:
            log.debug("Rejecting contour (perimeter ratio too low): %.2f", perimeter_to_sqrt_area)
            continue

        filtered.append(cnt)

    log.info("Detected %d potential room contours", len(filtered))
    return filtered


def _approx_polygon(cnt: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    """
    Approximate a contour to a polygon using the Douglas‑Peucker algorithm.

    Args:
        cnt: Contour array (Nx1x2).
        epsilon_factor: Approximation accuracy; multiplied by arc length.

    Returns:
        Approximated polygon (Nx1x2) with fewer vertices.
    """
    epsilon = epsilon_factor * cv2.arcLength(cnt, closed=True)
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
    return approx


def _polygon_to_coordinates(poly: np.ndarray) -> Polygon:
    """
    Convert a contour polygon to a plain list of (x, y) coordinate tuples.

    Args:
        poly: Polygon returned by `approxPolyDP`.

    Returns:
        List of (x, y) tuples.
    """
    return [(int(x), int(y)) for x, y in poly.squeeze()]


def _label_polygons(
    polygons: List[np.ndarray],
) -> RoomLabels:
    """
    Assign a unique label (e.g., "room_1", "room_2") to each polygon and
    convert it to coordinates.

    Args:
        polygons: List of polygon arrays.

    Returns:
        Mapping from label string to coordinate list.
    """
    room_dict: RoomLabels = {}
    for idx, poly in enumerate(polygons, start=1):
        label = f"room_{idx}"
        room_dict[label] = _polygon_to_coordinates(poly)
        log.debug("Labeled %s with %d points", label, len(poly))
    return room_dict


def detect_rooms(
    binary_image: np.ndarray,
    *,
    min_area: float = 150.0,
    min_perimeter_ratio: float = 0.3,
    epsilon_factor: float = 0.01,
) -> RoomLabels:
    """
    Detect individual rooms from a pre‑processed binary floor‑plan image.

    The function performs:

    1. External contour extraction with area/perimeter filtering.
    2. Polygon approximation to obtain relatively rectangular shapes.
    3. Labeling each detected room and returning its vertex coordinates.

    Args:
        binary_image: Binary image (single channel) where walls are white.
        min_area: Minimum contour area (pixel²) to consider a room.
        min_perimeter_ratio: Minimum perimeter/sqrt(area) ratio to reject elongated noise.
        epsilon_factor: Factor for polygon approximation (controls vertex count).

    Returns:
        Dict mapping room identifiers (e.g., "room_1") to a list of (x, y)
        coordinates representing the polygon of that room.

    Raises:
        ValueError: If ``binary_image`` is not a 2‑D array.
    """
    if len(binary_image.shape) != 2:
        raise ValueError("binary_image must be a 2‑D grayscale image")

    log.info("Starting room detection with min_area=%.0f, min_perimeter_ratio=%.2f",
             min_area, min_perimeter_ratio)

    # 1. Find and filter contours
    raw_contours = _find_external_contours(
        binary_image, min_area=min_area, min_perimeter_ratio=min_perimeter_ratio
    )

    if not raw_contours:
        log.warning("No room contours passed the filtering criteria.")
        return {}

    # 2. Approximate each contour to a polygon
    approximated = [_approx_polygon(cnt, epsilon_factor=epsilon_factor) for cnt in raw_contours]

    # 3. Convert polygons to coordinates and label them
    rooms = _label_polygons(approximated)

    log.info("Room detection completed: %d rooms found", len(rooms))
    return rooms


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------
def rooms_to_json(rooms: RoomLabels, *, indent: int = 2) -> str:
    """
    Serialize the room‑to‑coordinates mapping to a JSON string.

    Args:
        rooms: Mapping produced by :func:`detect_rooms`.
        indent: Number of spaces for pretty‑printing.

    Returns:
        JSON string representing the room coordinates.
    """
    log.debug("Serializing %d rooms to JSON", len(rooms))
    return json.dumps(rooms, indent=indent, ensure_ascii=False)


def save_rooms_to_file(
    rooms: RoomLabels,
    output_path: Union[str, Path],
    *,
    indent: int = 2,
) -> None:
    """
    Write the room coordinates to ``output_path`` as pretty‑printed JSON.

    Args:
        rooms: Mapping of room identifiers to coordinate lists.
        output_path: Destination file path.
        indent: JSON indentation level.

    Raises:
        OSError: If writing to the file fails.
    """
    json_str = rooms_to_json(rooms, indent=indent)
    path = Path(output_path)
    log.info("Saving rooms JSON to %s (%d bytes)", path, len(json_str))
    path.write_text(json_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# Example execution (run directly for quick debugging)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import numpy as np

    # Simple CLI sanity check: expects path to a pre‑processed binary image
    if len(sys.argv) != 2:
        print("Usage: python room_detector.py <path_to_binary_image>")
        sys.exit(1)

    bin_path = Path(sys.argv[1])
    if not bin_path.is_file():
        print(f"File not found: {bin_path}")
        sys.exit(1)

    try:
        img = cv2.imread(str(bin_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to load image")
        rooms = detect_rooms(img, min_area=200, min_perimeter_ratio=0.25)
        print("Detected rooms:")
        print(json.dumps(rooms, indent=2))
        # Optionally save JSON
        # save_rooms_to_file(rooms, "rooms.json")
    except Exception as exc:
        log.exception("Room detection failed")
        print(f"Error: {exc}")
        sys.exit(1)