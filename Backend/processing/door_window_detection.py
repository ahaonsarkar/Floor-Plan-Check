"""Door and window detection module."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class DoorWindowDetector:
    """Detects doors and windows in floor plan images."""

    def __init__(self, template_path: str = None):
        self.template_path = template_path
        self.templates = self._load_templates()

    def _load_templates(self) -> List[np.ndarray]:
        """Load or create door/window templates."""
        templates = []

        # Create simple templates
        door_template = np.ones((10, 20), dtype=np.uint8) * 255
        templates.append(door_template)

        window_template = np.ones((5, 15), dtype=np.uint8) * 255
        templates.append(window_template)

        return templates

    def detect(self, image: np.ndarray, room_masks: List[np.ndarray]) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect doors and windows.

        Args:
            image: Original image
            room_masks: List of room masks

        Returns:
            Tuple of (doors, windows)
        """
        doors = []
        windows = []

        # Placeholder for actual detection logic
        for i, mask in enumerate(room_masks):
            if i == 0:  # Assuming first room has a door
                height, width = mask.shape
                doors.append({
                    'room_id': i,
                    'position': [width // 2, height // 2],
                    'type': 'interior_door',
                    'size': {'width': 30, 'height': 80}
                })

            if i == 1:  # Assuming second room has a window
                height, width = mask.shape
                windows.append({
                    'room_id': i,
                    'position': [width // 4, height // 4],
                    'type': 'window',
                    'size': {'width': 60, 'height': 40}
                })

        return doors, windows