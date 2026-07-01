"""Wall detection module using OpenCV."""

import cv2
import numpy as np
from typing import List, Dict, Any


class WallDetector:
    """Detects walls in floor plan images."""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect walls in a preprocessed image.

        Args:
            image: Grayscale preprocessed image

        Returns:
            List of wall segments with properties
        """
        # Binarize image
        _, binary = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        walls = []
        for i, contour in enumerate(contours):
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            walls.append({
                'id': i,
                'contour': contour.tolist(),
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'area': int(cv2.contourArea(contour))
            })

        return walls