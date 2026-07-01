"""
Image processing module for floor plan analysis.

Handles loading, preprocessing, and basic transformations of floor plan images.
Provides functions for grayscale conversion, thresholding, denoising,
wall line detection, and contour extraction.

Typical workflow:
    image = load_image(path)
    processed = preprocess_image(image)
    contours = extract_contours(processed)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def load_image(image_path: Path) -> np.ndarray:
    """
    Load a floor plan image from disk.

    Args:
        image_path: Path to the image file (PNG/JPG)

    Returns:
        Image as numpy array in BGR format (as read by OpenCV).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is not a supported image format.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Attempt to read the image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image file: {image_path}. "
                         "Ensure it is a valid PNG or JPG.")
    return image


def validate_image_format(file_path: Path) -> bool:
    """
    Validate that the file is a supported image format.

    Args:
        file_path: Path to check

    Returns:
        True if the file exists and has a .png, .jpg, or .jpeg extension,
        False otherwise.
    """
    if not file_path.is_file():
        return False
    suffix = file_path.suffix.lower()
    return suffix in {'.png', '.jpg', '.jpeg'}


def preprocess_image(image: np.ndarray,
                     blur_ksize: int = 5,
                     threshold_block_size: int = 11,
                     threshold_c: int = 2) -> np.ndarray:
    """
    Preprocess the floor plan image for analysis.

    Steps:
        1. Convert to grayscale.
        2. Apply Gaussian blur to reduce noise.
        3. Apply adaptive thresholding to obtain a binary image.
        4. Perform morphological operations to enhance wall lines
           and remove small artifacts.

    Args:
        image: Input image array (BGR format).
        blur_ksize: Kernel size for Gaussian blur (must be odd and positive).
        threshold_block_size: Size of the pixel neighborhood for adaptive threshold
                              (must be odd and >= 3).
        threshold_c: Constant subtracted from the mean in adaptive threshold.

    Returns:
        A binary image (single channel) where walls are expected to be white
        (or black depending on implementation) on a contrasting background.

    Raises:
        ValueError: If kernel sizes are invalid.
    """
    if blur_ksize % 2 == 0 or blur_ksize < 1:
        raise ValueError("blur_ksize must be an odd positive integer")
    if threshold_block_size % 2 == 0 or threshold_block_size < 3:
        raise ValueError("threshold_block_size must be an odd integer >= 3")

    # 1. Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 3. Adaptive thresholding (binary inverse so walls become white on black background)
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=threshold_block_size,
        C=threshold_c
    )

    # 4. Morphological closing to connect broken wall segments
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional: Remove small noise contours via area filtering (done later in extract_contours)
    return morphed


def detect_wall_lines(binary_image: np.ndarray,
                      min_line_length: int = 30,
                      max_line_gap: int = 10) -> np.ndarray:
    """
    Detect wall-like lines in a binary image using the Probabilistic Hough Transform.

    Args:
        binary_image: Preprocessed binary image (single channel, 0-255).
        min_line_length: Minimum length of a line to be considered a wall (in pixels).
        max_line_gap: Maximum gap between line segments to treat them as a single line.

    Returns:
        An image (same size as input) with detected wall lines drawn in white on a black background.
    """
    # Ensure image is single channel 8-bit
    if len(binary_image.shape) != 2:
        raise ValueError("Input binary_image must be a single channel (grayscale) image")

    # Use HoughLinesP to detect lines
    lines = cv2.HoughLinesP(
        binary_image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    line_image = np.zeros_like(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), color=255, thickness=2)

    return line_image


def extract_contours(binary_image: np.ndarray,
                     min_area: float = 100.0) -> List[np.ndarray]:
    """
    Extract contours from a preprocessed binary image.

    Args:
        binary_image: Binary image (single channel) where walls/objects are white.
        min_area: Minimum contour area to be considered valid (filters noise).

    Returns:
        A list of contours, each contour being a Nx1x2 array of (x, y) points.
    """
    # Find contours (external only, as we expect walls to form closed boundaries)
    contours, hierarchy = cv2.findContours(
        binary_image,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area to remove small noise
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return filtered


def draw_contours_on_image(image: np.ndarray,
                           contours: List[np.ndarray],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
    """
    Draw detected contours on the original image for visualization.

    Args:
        image: Original BGR image.
        contours: List of contours returned by extract_contours.
        color: BGR color tuple for drawing contours.
        thickness: Line thickness for drawing.

    Returns:
        Image with contours drawn on it.
    """
    img_with_contours = image.copy()
    cv2.drawContours(img_with_contours, contours, -1, color, thickness)
    return img_with_contours


class ImageProcessor:
    """Wrapper class for floor plan image pre-processing."""

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image using preprocess_image."""
        return preprocess_image(image)


# Example usage (for quick testing when run as script)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not validate_image_format(img_path):
        print(f"Error: {img_path} is not a supported image format.")
        sys.exit(1)

    try:
        original = load_image(img_path)
        processor = ImageProcessor()
        processed = processor.preprocess(original)
        wall_lines = detect_wall_lines(processed)
        contours = extract_contours(processed)

        # For demonstration, show results (requires GUI environment)
        cv2.imshow("Original", original)
        cv2.imshow("Preprocessed", processed)
        cv2.imshow("Wall Lines", wall_lines)
        viz = draw_contours_on_image(original, contours)
        cv2.imshow("Contours", viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)