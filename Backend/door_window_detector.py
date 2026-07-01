"""
Door and window detector for floor‑plan analysis.

Detects openings in wall structures that correspond to doors and windows
using OpenCV edge detection, contour analysis and simple geometric heuristics.

Typical usage
-------------
    from door_window_detector import detect_openings, openings_to_json
    # ``binary_image`` is the output of ``image_processor.preprocess_image``
    openings = detect_openings(binary_image)
    json_result = openings_to_json(openings)

The function returns a JSON‑serialisable mapping:
{
    "doors":   [[x1, y1], [x2, y2], ...],
    "windows": [[x1, y1], [x2, y2], ...]
}
where each coordinate list represents the vertices of a detected gap.
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Coordinate = Tuple[int, int]
Polygon = List[Coordinate]
OpeningList = List[Polygon]


def _invert_binary(image: np.ndarray) -> np.ndarray:
    """
    Invert a binary wall image (walls=white, background=black) so that gaps
    become white regions that are easier to extract as contours.

    Args:
        image: Single‑channel binary image (0‑255).

    Returns:
        Inverted image where foreground and background are swapped.
    """
    return cv2.bitwise_not(image)


def _find_gap_contours(
    inverted: np.ndarray,
    min_area: float = 50.0,
    min_perimeter_ratio: float = 0.2,
) -> List[np.ndarray]:
    """
    Extract external contours from an inverted binary image and filter them
    by area and perimeter‑to‑sqrt(area) ratio. The filtered contours are
    assumed to correspond to gaps (doors or windows).

    Args:
        inverted: Binary image where gaps appear as white shapes.
        min_area: Minimum contour area (pixel²) to keep.
        min_perimeter_ratio: Minimum perimeter / sqrt(area) ratio to keep.
                             Helps reject tiny noise blobs.

    Returns:
        List of filtered contours (each contour is an Nx1x2 array of points).
    """
    log.debug(
        "Finding gap contours (min_area=%.0f, min_perimeter_ratio=%.2f)",
        min_area,
        min_perimeter_ratio,
    )

    contours, hierarchy = cv2.findContours(
        inverted,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            log.debug("Rejecting gap (area too small): %.0f", area)
            continue

        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        perimeter_to_sqrt_area = perimeter / (area ** 0.5)

        if perimeter_to_sqrt_area < min_perimeter_ratio:
            log.debug(
                "Rejecting gap (perimeter ratio too low): %.2f", perimeter_to_sqrt_area
            )
            continue

        filtered.append(cnt)

    log.info("Detected %d potential gap contours", len(filtered))
    return filtered


def _classify_gap(
    cnt: np.ndarray,
    *,  # keyword‑only arguments for clarity
    door_min_height: int = 20,
    door_max_aspect: float = 0.6,
    window_max_area_ratio: float = 0.02,
    binary_shape: Tuple[int, int] = (0, 0),  # height, width of original image
) -> str:
    """
    Classify a gap contour as a door or a window based on geometric heuristics.

    Heuristics used:
        * **Door** – relatively tall/vertical gap or wide/horizontal gap (size > ``door_min_height``)
          and aspect ratio below ``door_max_aspect``.
        * **Window** – smaller gap whose area is below a small fraction of the
          image size (``window_max_area_ratio``) and whose shape is more square‑like.

    Args:
        cnt: Contour approximating a gap.
        door_min_height: Minimum length (in pixels) for a door opening.
        door_max_aspect: Maximum thickness/length ratio for a door.
        window_max_area_ratio: Maximum allowed area as a fraction of the total
                               image area for a window.
        binary_shape: (height, width) of the original binary image.

    Returns:
        The string ``"door"``, ``"window"``, or ``"other"``.
    """
    x, y, w, h = cv2.boundingRect(cnt)

    is_vertical_door = h >= door_min_height and (w / h <= door_max_aspect if h > 0 else False)
    is_horizontal_door = w >= door_min_height and (h / w <= door_max_aspect if w > 0 else False)

    if is_vertical_door or is_horizontal_door:
        log.debug("Gap classified as door (w=%d, h=%d)", w, h)
        return "door"

    # Compute area fraction of the whole image for window heuristic
    img_h, img_w = binary_shape
    img_area = img_h * img_w if img_h > 0 and img_w > 0 else 1
    area = cv2.contourArea(cnt)
    area_fraction = area / img_area if img_area > 0 else 0

    # Window heuristic: small area and roughly square aspect
    if area_fraction <= window_max_area_ratio:
        aspect = w / h if h != 0 else float('inf')
        if 0.5 <= aspect <= 2.0:  # fairly square
            log.debug(
                "Gap classified as window (area_frac=%.4f, aspect=%.2f)",
                area_fraction,
                aspect,
            )
            return "window"

    log.debug(
        "Gap classified as other (height=%d, aspect=%.2f, area_frac=%.4f)",
        h,
        w / h if h != 0 else float('inf'),
        area_fraction,
    )
    return "other"


def _extract_coordinates(cnt: np.ndarray) -> Polygon:
    """
    Convert a contour array to a plain list of (x, y) tuples.

    Args:
        cnt: Contour array (Nx1x2).

    Returns:
        List of (x, y) coordinate pairs.
    """
    return [(int(x), int(y)) for x, y in cnt.squeeze()]


def detect_openings(
    binary_wall_image: np.ndarray,
    *,
    min_gap_area: float = 50.0,
    min_gap_perimeter_ratio: float = 0.2,
    door_min_height: int = 120,
    door_max_aspect: float = 0.6,
    window_max_area_ratio: float = 0.02,
) -> Dict[str, OpeningList]:
    """
    Detect doors and windows from a pre‑processed binary wall image.

    The algorithm follows these steps:
        1. Invert the binary image so that gaps become bright shapes.
        2. Extract external contours of the inverted image.
        3. Filter contours by area and perimeter ratio.
        4. Classify each remaining contour as ``door``, ``window`` or ``other``
           using geometric heuristics (height, aspect ratio, area).
        5. Return a mapping with two keys: ``"doors"`` and ``"windows"``,
           each containing a list of coordinate polygons.

    Args:
        binary_wall_image: Binary image where walls are white (255) on a black
                           background. Produced by :func:`image_processor.preprocess_image`.
        min_gap_area: Minimum contour area (pixel²) to consider a gap.
        min_gap_perimeter_ratio: Minimum perimeter / sqrt(area) ratio to keep a gap.
        door_min_height: Minimum vertical size (pixels) for a gap to be considered a door.
        door_max_aspect: Maximum width/height ratio for a door (vertical orientation).
        window_max_area_ratio: Maximum fraction of total image area for a gap to be considered a window.

    Returns:
        Dictionary of the form ``{"doors": [...], "windows": [...]}`` where each list
        contains polygons (lists of ``[x, y]`` coordinates) representing the detected
        door and window openings.

    Raises:
        ValueError: If ``binary_wall_image`` is not a 2‑D NumPy array.
    """
    if len(binary_wall_image.shape) != 2:
        raise ValueError("binary_wall_image must be a 2‑D grayscale image")

    log.info("Starting door/window detection with min_gap_area=%.0f, min_gap_perimeter_ratio=%.2f",
             min_gap_area, min_gap_perimeter_ratio)

    # 1. Invert image – gaps become bright shapes
    inverted = _invert_binary(binary_wall_image)

    # 2. Find and filter gap contours
    gap_contours = _find_gap_contours(
        inverted,
        min_area=min_gap_area,
        min_perimeter_ratio=min_gap_perimeter_ratio,
    )

    doors: OpeningList = []
    windows: OpeningList = []

    # Pass image shape for window area fraction calculation
    binary_shape = binary_wall_image.shape  # (height, width)

    # 3. Classify each contour
    for cnt in gap_contours:
        gap_type = _classify_gap(
            cnt,
            door_min_height=door_min_height,
            door_max_aspect=door_max_aspect,
            window_max_area_ratio=window_max_area_ratio,
            binary_shape=binary_shape,
        )
        if gap_type == "door":
            doors.append(_extract_coordinates(cnt))
        elif gap_type == "window":
            windows.append(_extract_coordinates(cnt))
        # "other" gaps are ignored

    result: Dict[str, OpeningList] = {"doors": doors, "windows": windows}
    log.info("Detection completed: %d doors, %d windows", len(doors), len(windows))
    return result


def openings_to_json(openings: Dict[str, OpeningList], *, indent: int = 2) -> str:
    """
    Serialize the ``doors`` and ``windows`` mappings to a pretty‑printed JSON
    string.

    Args:
        openings: Dictionary produced by :func:`detect_openings`.
        indent: Number of spaces for JSON indentation.

    Returns:
        JSON string ready for transmission to the frontend or for storage.
    """
    log.debug("Serialising openings to JSON (doors=%d, windows=%d)", len(openings.get("doors", [])), len(openings.get("windows", [])))
    return json.dumps(openings, indent=indent, ensure_ascii=False)


def save_openings_to_file(
    openings: Dict[str, OpeningList],
    output_path: Union[str, Path],
    *,
    indent: int = 2,
) -> None:
    """
    Persist the opening detection results to ``output_path`` as formatted JSON.

    Args:
        openings: Dictionary produced by :func:`detect_openings`.
        output_path: Destination file path.
        indent: JSON indentation level.

    Raises:
        OSError: If writing to the file fails.
    """
    json_str = openings_to_json(openings, indent=indent)
    path = Path(output_path)
    log.info("Saving openings JSON to %s (%d bytes)", path, len(json_str))
    path.write_text(json_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# Example CLI usage (run ``python door_window_detector.py <path_to_binary>``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import numpy as np

    if len(sys.argv) != 2:
        print("Usage: python door_window_detector.py <path_to_processed_binary_image>")
        sys.exit(1)

    bin_path = Path(sys.argv[1])
    if not bin_path.is_file():
        print(f"File not found: {bin_path}")
        sys.exit(1)

    try:
        img = cv2.imread(str(bin_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to load image")

        openings = detect_openings(
            img,
            min_gap_area=80,
            min_gap_perimeter_ratio=0.15,
            door_min_height=150,
            door_max_aspect=0.5,
            window_max_area_ratio=0.015,
        )
        print("Detected openings:")
        print(json.dumps(openings, indent=2))

        # Optional: persist result
        # save_openings_to_file(openings, "openings.json")

    except Exception as exc:
        log.exception("Door/window detection failed")
        print(f"Error: {exc}")
        sys.exit(1)