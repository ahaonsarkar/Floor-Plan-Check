"""Room segmentation module."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class RoomSegmenter:
    """Segments rooms from floor plan images."""

    def __init__(self, min_area: int = 500):
        self.min_area = min_area

    def segment(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Segment rooms from an image.

        Args:
            image: Preprocessed image (walls are white 255, background is black 0)

        Returns:
            Tuple of (room_masks, contours)
        """
        # 1. Invert image so walls are black (0) and rooms/background are white (255)
        inverted = 255 - image
        _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

        # 2. Close outer wall gaps temporarily to prevent background floodfill leak inside the rooms
        # Using a 25x25 kernel is generally sufficient to close door/window gaps in outer walls
        h, w = binary.shape[:2]
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        binary_closed = cv2.erode(binary, kernel_close, iterations=1)

        # 3. Flood fill from the edges on binary_closed to isolate the outside background
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for x in range(w):
            if binary_closed[0, x] == 255:
                cv2.floodFill(binary_closed, mask, (x, 0), 0)
            if binary_closed[h - 1, x] == 255:
                cv2.floodFill(binary_closed, mask, (x, h - 1), 0)

        for y in range(h):
            if binary_closed[y, 0] == 255:
                cv2.floodFill(binary_closed, mask, (0, y), 0)
            if binary_closed[y, w - 1] == 255:
                cv2.floodFill(binary_closed, mask, (w - 1, y), 0)

        # 4. Use the isolated rooms mask to select room pixels from the original binary image
        binary_no_bg = np.zeros_like(binary)
        binary_no_bg[binary_closed == 255] = binary[binary_closed == 255]

        # 5. Calculate distance transform to dynamically choose erosion kernel size for separating rooms
        dist_transform = cv2.distanceTransform(binary_no_bg, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist <= 0:
            return [], []

        # Determine dynamic kernel size
        kernel_size = int(max_dist * 0.35)
        if kernel_size < 7:
            kernel_size = 7
        if kernel_size > 51:
            kernel_size = 51
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(binary_no_bg, kernel, iterations=1)

        # 6. Find contours of the separate rooms
        contours_eroded, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        room_masks = []
        room_contours = []

        for cnt in contours_eroded:
            if cv2.contourArea(cnt) < 100:
                continue

            # Create mask for this room (eroded size)
            room_mask_eroded = np.zeros((h, w), np.uint8)
            cv2.drawContours(room_mask_eroded, [cnt], -1, 255, -1)

            # Dilate back to original size
            room_mask_dilated = cv2.dilate(room_mask_eroded, kernel, iterations=1)

            # Constrain to the original binary_no_bg (no bleeding into walls)
            room_mask_final = cv2.bitwise_and(room_mask_dilated, binary_no_bg)

            # Check final area
            cnts_final, _ = cv2.findContours(room_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_final:
                continue

            contour_final = max(cnts_final, key=cv2.contourArea)
            if cv2.contourArea(contour_final) < self.min_area:
                continue

            room_masks.append(room_mask_final)
            room_contours.append(contour_final)

        return room_masks, room_contours