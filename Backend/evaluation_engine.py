"""
Evaluation Engine – Space Utilization, Room Adjacency, & Movement Flow
===================================================================

This module now offers three independent evaluation utilities:

1. **Space Utilization** – ratio of usable floor area to total interior area.
2. **Room Adjacency** – graph‑based scoring of architectural adjacency rules.
3. **Movement Flow** – analysis of inter‑room accessibility using an A* path‑
   finding algorithm on a graph of rooms.  The result is an ``accessibility_score``
   (0‑100) where a higher score indicates more efficient movement.  In addition,
   the function returns a list of *inefficient* room pairs whose shortest‑path
   distance exceeds a configurable multiplier of the direct Euclidean distance.

All utilities expose a single public function each and share JSON serialisation
helpers and a small command‑line interface for ad‑hoc testing.
"""

from __future__ import annotations

import json
import logging
import math
import heapq
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import cv2
import numpy as np

# Optional – NetworkX provides convenient graph APIs but is not required.
try:
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False
    nx = None

# ---------------------------------------------------------------------------
# Logging – callers may configure the root logger; we default to INFO.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Basic image utilities (shared by space‑utilisation and other metrics)
# ---------------------------------------------------------------------------
def _pixel_count(mask: np.ndarray) -> int:
    """Count non‑zero pixels in a binary mask.

    Args:
        mask: 2‑D ``uint8`` array where foreground pixels are >0.
    Returns:
        Integer pixel count.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2‑D array")
    return int(np.count_nonzero(mask))

def _total_floor_area(binary_floor: np.ndarray) -> int:
    """Calculate interior floor area from a pre‑processed binary image.

    The pre‑processed image has walls drawn in white (255). The interior (usable)
    area is the inverse of that mask.
    """
    interior = cv2.bitwise_not(binary_floor)
    area = _pixel_count(interior)
    log.debug("Total interior floor area (pixels): %d", area)
    return area

def _rooms_area(room_masks: List[np.ndarray]) -> int:
    """Sum the pixel area of each room mask.

    Args:
        room_masks: List of binary masks – each mask must share the shape of the
                    original floor image.
    Returns:
        Total room area in pixel units.
    """
    total = 0
    for idx, mask in enumerate(room_masks, start=1):
        area = _pixel_count(mask)
        log.debug("Room %d area: %d pixels", idx, area)
        total += area
    log.info("Combined room area (usable before corridor subtraction): %d", total)
    return total

def _corridor_area(corridor_mask: Optional[np.ndarray]) -> int:
    """Return the area of the corridor mask, or ``0`` if none supplied."""
    if corridor_mask is None:
        return 0
    area = _pixel_count(corridor_mask)
    log.debug("Corridor area (pixels): %d", area)
    return area

def _compute_score(total: int, usable: int) -> int:
    """Calculate the integer space‑utilization score (0‑100)."""
    if total <= 0:
        raise ValueError("Total floor area must be a positive integer")
    score = round((usable / total) * 100)
    return max(0, min(100, int(score)))

# ---------------------------------------------------------------------------
# Public API – Space Utilization
# ---------------------------------------------------------------------------
def evaluate_space_utilization(
    binary_floor: np.ndarray,
    room_masks: List[np.ndarray],
    *,
    corridor_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute space‑utilization metrics for a floor plan.

    Returns a dictionary ready for JSON serialisation:
    {
        "space_score": <int 0‑100>,
        "usable_area": <pixel count>,
        "wasted_area": <pixel count>
    }
    """
    log.info("Starting space‑utilization evaluation …")

    if not room_masks:
        return {
            "space_score": 0,
            "usable_area": 0,
            "wasted_area": 0,
        }

    # Calculate usable room areas
    rooms_area = sum(int(np.count_nonzero(mask)) for mask in room_masks)
    corridor_area = _corridor_area(corridor_mask)
    usable_area = max(0, rooms_area - corridor_area)

    # Find the total footprint / boundary of the house to compute total interior area
    # Merge all room masks to find their combined shape
    union_mask = np.zeros_like(room_masks[0])
    for mask in room_masks:
        union_mask = cv2.bitwise_or(union_mask, mask)

    # Find contours of the union mask
    contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Create a convex hull around all room contours to get the total interior boundary
        all_pts = np.concatenate(contours)
        hull = cv2.convexHull(all_pts)
        hull_mask = np.zeros_like(union_mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        
        # The total interior area includes all space inside the boundary except thick wall structures (white walls)
        # So we count the non-wall pixels inside the hull mask
        non_wall_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(binary_floor))
        total_interior_area = int(np.count_nonzero(non_wall_mask))
    else:
        total_interior_area = _total_floor_area(binary_floor)

    if total_interior_area <= 0:
        total_interior_area = max(1, usable_area)

    # Wasted area is space inside the house that is not assigned to any room
    wasted_area = max(0, total_interior_area - usable_area)

    # Base score: ratio of usable room area to total interior area
    # Ideally, 70-95% of the house should be usable rooms (leaving 5-30% for walls & circulation/corridors)
    ratio = usable_area / total_interior_area
    
    if 0.70 <= ratio <= 0.95:
        base_score = 100.0
    elif ratio > 0.95:
        # Too little circulation space, indicating rooms are crammed / lack privacy
        base_score = 100.0 - (ratio - 0.95) * 100
    else:
        # Too much wasted/corridor space
        base_score = 100.0 - (0.70 - ratio) * 50

    # Apply penalties
    penalties = 0.0

    # 1. Room Aspect Ratio Penalty: only penalize if extremely elongated (AR > 3.0)
    for idx, mask in enumerate(room_masks):
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(cnt)
            if w > 0 and h > 0:
                ar = max(w, h) / min(w, h)
                if ar > 3.0:
                    # Penalize based on how narrow it is
                    penalties += min(5.0, (ar - 3.0) * 3.0)

    # 2. Size Imbalance Penalty: only penalize if extremely tiny (< 2%) or extremely dominating (> 85%)
    if len(room_masks) >= 2:
        areas = [int(np.count_nonzero(m)) for m in room_masks]
        total_rooms_area = sum(areas)
        if total_rooms_area > 0:
            min_ratio = min(areas) / total_rooms_area
            max_ratio = max(areas) / total_rooms_area
            if min_ratio < 0.02:
                penalties += 5.0  # tiny room penalty
            if max_ratio > 0.85:
                penalties += 5.0  # single dominating room penalty

    score = max(0, min(100, int(round(base_score - penalties))))

    result = {
        "space_score": score,
        "usable_area": usable_area,
        "wasted_area": wasted_area,
    }
    log.info(
        "Space Utilization – score=%d, usable=%d, wasted=%d, ratio=%.2f",
        score,
        usable_area,
        wasted_area,
        ratio,
    )
    return result

# ---------------------------------------------------------------------------
# Room Adjacency Evaluation (unchanged from previous implementation)
# ---------------------------------------------------------------------------
GOOD_ADJACENCIES: Set[Tuple[str, str]] = {
    ("Kitchen", "Dining room"),
    ("Dining room", "Kitchen"),
    ("Bedroom", "Bathroom"),
    ("Bathroom", "Bedroom"),
}
BAD_ADJACENCIES: Set[Tuple[str, str]] = {
    ("Bathroom", "Kitchen"),
    ("Kitchen", "Bathroom"),
    ("Bedroom", "Entrance"),
    ("Entrance", "Bedroom"),
}

GOOD_BONUS = 5   # points added per good adjacency
BAD_PENALTY = 10  # points subtracted per bad adjacency
BASE_SCORE = 100

def _build_room_graph(adjacency: Dict[int, List[int]]) -> "nx.Graph":
    """Create an undirected graph of room connectivity.

    Args:
        adjacency: Mapping from room identifier to a list of neighbouring room
                   identifiers.
    Returns:
        A NetworkX ``Graph`` (or a simple dict‑based fallback) containing the
        same edges.
    """
    if _HAS_NX:
        G = nx.Graph()
        for src, targets in adjacency.items():
            for tgt in targets:
                G.add_edge(src, tgt)
        return G
    else:  # pragma: no cover – fallback without NetworkX
        return adjacency  # type: ignore

def _score_adjacency_pair(type_a: str, type_b: str) -> int:
    """Return a numeric contribution for a single adjacency pair.

    Positive values reward good architectural relationships, negative values
    penalise undesirable ones.
    """
    if (type_a, type_b) in GOOD_ADJACENCIES:
        return GOOD_BONUS
    if (type_a, type_b) in BAD_ADJACENCIES:
        return -BAD_PENALTY
    return 0

def evaluate_room_adjacency(
    room_types: List[str],
    adjacency: Dict[int, List[int]],
) -> Dict[str, Any]:
    """Assess room adjacency quality and return a normalized score.

    Args:
        room_types: List of room type strings. The index of the list corresponds
                    to the integer room identifier used in ``adjacency`` (i.e.,
                    ``room_id == index``).  Types must match the canonical names
                    used in the heuristic tables (e.g., ``"Kitchen"``).
        adjacency: Mapping ``room_id -> list[neighbor_id]`` describing which
                    rooms share a wall/opening.

    Returns:
        ``{"adjacency_score": <int 0‑100>}``
    """
    log.info("Evaluating room adjacency …")

    if not room_types:
        return {"adjacency_score": 0}

    _ = _build_room_graph(adjacency)

    # We start with a base score of 90 (neutral score)
    base_score = 90.0
    total_delta = 0.0

    visited: Set[Tuple[int, int]] = set()
    for src, neighbours in adjacency.items():
        for tgt in neighbours:
            edge = tuple(sorted((src, tgt)))
            if edge in visited:
                continue
            visited.add(edge)
            try:
                type_src = room_types[src]
                type_tgt = room_types[tgt]
            except IndexError as exc:
                raise IndexError(
                    f"Room ID {src} or {tgt} out of range for provided room_types list"
                ) from exc
            
            # Custom Adjacency Rules:
            # Good pairings (bonus)
            if (type_src, type_tgt) in GOOD_ADJACENCIES or (type_tgt, type_src) in GOOD_ADJACENCIES:
                total_delta += 5.0
            # Bad pairings (penalty)
            elif (type_src, type_tgt) in BAD_ADJACENCIES or (type_tgt, type_src) in BAD_ADJACENCIES:
                total_delta -= 15.0
            # Bathroom adjacent to Dining Room (minor penalty)
            elif {type_src, type_tgt} == {"Bathroom", "Dining room"}:
                total_delta -= 5.0
            # Bedroom adjacent to Kitchen (minor penalty for noise/smell)
            elif {type_src, type_tgt} == {"Bedroom", "Kitchen"}:
                total_delta -= 5.0

    # Essential architectural expectations (penalize if they are missing)
    missing_penalties = 0.0
    
    # 1. Kitchen and Dining room should be adjacent if both exist
    if "Kitchen" in room_types and "Dining room" in room_types:
        kitchen_indices = [i for i, t in enumerate(room_types) if t == "Kitchen"]
        dining_indices = [i for i, t in enumerate(room_types) if t == "Dining room"]
        has_adj = False
        for k in kitchen_indices:
            for d in dining_indices:
                if d in adjacency.get(k, []) or k in adjacency.get(d, []):
                    has_adj = True
                    break
        if not has_adj:
            missing_penalties += 5.0

    # 2. Bedroom and Bathroom should be adjacent (at least one bedroom must have bathroom access)
    if "Bedroom" in room_types and "Bathroom" in room_types:
        bedroom_indices = [i for i, t in enumerate(room_types) if t == "Bedroom"]
        bathroom_indices = [i for i, t in enumerate(room_types) if t == "Bathroom"]
        has_adj = False
        for b in bedroom_indices:
            for bath in bathroom_indices:
                if bath in adjacency.get(b, []) or b in adjacency.get(bath, []):
                    has_adj = True
                    break
        if not has_adj:
            missing_penalties += 8.0

    # 3. Isolated rooms (no neighbors at all)
    for r_id in range(len(room_types)):
        if not adjacency.get(r_id, []):
            missing_penalties += 10.0

    raw_score = base_score + total_delta - missing_penalties
    adjacency_score = max(0, min(100, int(round(raw_score))))
    log.info("Adjacency evaluation completed – score %d (delta %.1f, missing %.1f)", adjacency_score, total_delta, missing_penalties)

    return {"adjacency_score": adjacency_score}

# ---------------------------------------------------------------------------
# Movement Flow / Accessibility Evaluation (A* on room graph)
# ---------------------------------------------------------------------------

def _room_centroids(room_masks: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Compute the centroid (x, y) of each binary room mask.

    Args:
        room_masks: List of binary masks (same dimensions as the floor image).
    Returns:
        List of (cx, cy) tuples – coordinates are in pixel units.
    """
    centroids: List[Tuple[float, float]] = []
    for idx, mask in enumerate(room_masks, start=1):
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            # Fallback: use mask's bounding box centre
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        centroids.append((cx, cy))
        log.debug("Room %d centroid: (%.2f, %.2f)", idx - 1, cx, cy)
    return centroids

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _build_weighted_graph(
    adjacency: Dict[int, List[int]],
    centroids: List[Tuple[float, float]],
) -> Dict[int, List[Tuple[int, float]]]:
    """Create a weighted adjacency list where edge weight = Euclidean distance.

    Returns a dict ``node -> list[(neighbor, weight)]``.
    """
    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(centroids))}
    for src, nbrs in adjacency.items():
        for tgt in nbrs:
            if tgt < 0 or tgt >= len(centroids):
                continue  # safeguard against bad indices
            weight = _euclidean(centroids[src], centroids[tgt])
            graph[src].append((tgt, weight))
    return graph

def _astar(
    graph: Dict[int, List[Tuple[int, float]]],
    start: int,
    goal: int,
    positions: List[Tuple[float, float]],
) -> float:
    """A* search on a weighted undirected graph.

    Returns the total cost of the shortest path.  If ``goal`` is unreachable,
    returns ``math.inf``.
    """
    if start == goal:
        return 0.0

    frontier: List[Tuple[float, int]] = []
    heapq.heappush(frontier, (0.0, start))
    cost_so_far: Dict[int, float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            return cost_so_far[current]
        for neighbor, weight in graph.get(current, []):
            new_cost = cost_so_far[current] + weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heuristic = _euclidean(positions[neighbor], positions[goal])
                priority = new_cost + heuristic
                heapq.heappush(frontier, (priority, neighbor))
    return math.inf

def evaluate_accessibility(
    room_masks: List[np.ndarray],
    adjacency: Dict[int, List[int]],
    *,
    inefficiency_factor: float = 1.5,
) -> Dict[str, Any]:
    """Assess overall movement flow using A* shortest‑path analysis.

    The algorithm computes the shortest path length between **every pair** of
    rooms on the connectivity graph.  The *accessibility score* is derived from
    the ratio of the ideal total distance (direct Euclidean distance between
    centroids) to the actual traversable total distance.  In addition, pairs whose
    actual path exceeds ``inefficiency_factor`` × the direct distance are flagged
    as *inefficient* and reported.

    Args:
        room_masks: List of binary masks for each detected room.
        adjacency: Mapping ``room_id -> list[neighbor_id]`` describing direct
                   connectivity (e.g., through doors).
        inefficiency_factor: Multiplier that decides when a path is considered
                             inefficient.  Default ``1.5`` (i.e., 50 % longer
                             than the straight line).

    Returns:
        ``{"accessibility_score": <int 0‑100>, "inefficient_paths": [...]}``
        where each entry in ``inefficient_paths`` is a dict with keys
        ``source``, ``target``, ``direct_distance``, and ``path_distance``.
    """
    log.info("Starting accessibility (movement flow) evaluation …")
    if not room_masks:
        return {"accessibility_score": 0, "inefficient_paths": []}
    if len(room_masks) == 1:
        return {"accessibility_score": 100, "inefficient_paths": []}

    centroids = _room_centroids(room_masks)
    graph = _build_weighted_graph(adjacency, centroids)

    n = len(room_masks)
    total_actual = 0.0
    total_direct = 0.0
    pairs = 0
    unreachable_pairs_count = 0
    inefficient_paths: List[Dict[str, Any]] = []
    
    for i in range(n):
        for j in range(i + 1, n):
            direct = _euclidean(centroids[i], centroids[j])
            path_len = _astar(graph, i, j, centroids)
            
            total_direct += direct
            pairs += 1
            
            if math.isinf(path_len):
                unreachable_pairs_count += 1
                # Penalty: assume the path is 2x the direct distance
                penalized_path_len = 2.0 * direct
                total_actual += penalized_path_len
                log.warning("Rooms %d and %d are not reachable", i, j)
            else:
                total_actual += path_len
                if path_len > inefficiency_factor * direct:
                    inefficient_paths.append(
                        {
                            "source": i,
                            "target": j,
                            "direct_distance": round(direct, 2),
                            "path_distance": round(path_len, 2),
                        }
                    )
                    log.debug(
                        "Inefficient path detected between %d and %d (direct %.2f, path %.2f)",
                        i,
                        j,
                        direct,
                        path_len,
                    )

    if pairs == 0 or total_actual == 0:
        base_score = 0
    else:
        ratio = total_direct / total_actual
        base_score = round(ratio * 100)

    # Disconnection penalty: 5 points per unreachable pair
    disconnection_penalty = unreachable_pairs_count * 5
    
    # Inefficient path penalty: 2 points per inefficient path
    inefficiency_penalty = len(inefficient_paths) * 2

    score = base_score - disconnection_penalty - inefficiency_penalty
    score = max(0, min(100, int(round(score))))
    
    log.info(
        "Accessibility evaluation completed – score %d (base %d, %d unreachable, %d inefficient pairs)",
        score,
        base_score,
        unreachable_pairs_count,
        len(inefficient_paths),
    )
    return {"accessibility_score": score, "inefficient_paths": inefficient_paths}

# ---------------------------------------------------------------------------
# Shared JSON helpers (used by all three evaluations)
# ---------------------------------------------------------------------------
def to_json(data: Dict[str, Any], *, indent: int = 2) -> str:
    """Serialise a dictionary to a pretty‑printed JSON string."""
    return json.dumps(data, indent=indent, ensure_ascii=False)

def save_to_file(
    data: Dict[str, Any],
    output_path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Write a result dictionary to ``output_path`` as JSON.

    Args:
        data: Result dictionary.
        output_path: Destination file path.
        indent: JSON indentation level (default 2).
    """
    path = Path(output_path)
    json_str = to_json(data, indent=indent)
    path.write_text(json_str, encoding="utf-8")
    log.info("Result saved to %s (%d bytes)", path, len(json_str))

# ---------------------------------------------------------------------------
# Simple CLI for quick interactive testing (not used by the API server).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import ast

    parser = argparse.ArgumentParser(description="Evaluate floor‑plan metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Space utilization sub‑command (unchanged)
    sp_parser = subparsers.add_parser("space", help="Space utilization score")
    sp_parser.add_argument("binary_floor", type=Path)
    sp_parser.add_argument("room_masks", type=Path, nargs="+")
    sp_parser.add_argument("--corridor", type=Path, default=None)
    sp_parser.add_argument("--output", type=Path, default=None)

    # Adjacency sub‑command (unchanged)
    adj_parser = subparsers.add_parser("adjacency", help="Room adjacency score")
    adj_parser.add_argument("--room-types", type=str, required=True, help="JSON list of room types")
    adj_parser.add_argument("--adjacency", type=str, required=True, help="JSON dict of adjacency list")
    adj_parser.add_argument("--output", type=Path, default=None)

    # Accessibility sub‑command
    acc_parser = subparsers.add_parser("accessibility", help="Movement flow / accessibility score")
    acc_parser.add_argument("room_masks", type=Path, nargs="+")
    acc_parser.add_argument("--adjacency", type=str, required=True, help="JSON dict of adjacency list")
    acc_parser.add_argument("--inefficiency-factor", type=float, default=1.5, help="Multiplier for flagging inefficient paths")
    acc_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "space":
        try:
            floor_img = cv2.imread(str(args.binary_floor), cv2.IMREAD_GRAYSCALE)
            if floor_img is None:
                raise ValueError("Failed to load binary floor image")
            room_imgs = []
            for rp in args.room_masks:
                img = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to load room mask: {rp}")
                room_imgs.append(img)
            corridor_img = None
            if args.corridor:
                corridor_img = cv2.imread(str(args.corridor), cv2.IMREAD_GRAYSCALE)
                if corridor_img is None:
                    raise ValueError("Failed to load corridor mask")
        except Exception as exc:
            log.exception("Error loading input files")
            sys.exit(1)
        result = evaluate_space_utilization(
            binary_floor=floor_img,
            room_masks=room_imgs,
            corridor_mask=corridor_img,
        )
        print(to_json(result))
        if args.output:
            save_to_file(result, args.output)

    elif args.command == "adjacency":
        try:
            room_types = ast.literal_eval(args.room_types)
            adjacency = ast.literal_eval(args.adjacency)
        except Exception as exc:
            log.error("Failed to parse JSON arguments: %s", exc)
            sys.exit(1)
        result = evaluate_room_adjacency(room_types, adjacency)
        print(to_json(result))
        if args.output:
            save_to_file(result, args.output)

    elif args.command == "accessibility":
        try:
            room_masks = []
            for rp in args.room_masks:
                img = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to load room mask: {rp}")
                room_masks.append(img)
            adjacency = ast.literal_eval(args.adjacency)
        except Exception as exc:
            log.exception("Error loading inputs for accessibility evaluation")
            sys.exit(1)
        result = evaluate_accessibility(
            room_masks,
            adjacency,
            inefficiency_factor=args.inefficiency_factor,
        )
        print(to_json(result))
        if args.output:
            save_to_file(result, args.output)