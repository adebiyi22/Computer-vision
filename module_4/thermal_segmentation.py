"""
==============================================================================
CSc 8830 - Computer Vision - Module 4 Assignment - Part 2
Human Boundary Segmentation from a Thermal (Infrared) Image
(Classical CV only, NO ML/DL)

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Finds the boundary/outline of a human in a thermal/infrared image using
    only classical OpenCV operations. Thermal cameras render body heat as
    high intensity values, so a warm human body typically appears as a
    bright blob against a cooler (darker) background - this is exploited
    directly via intensity thresholding rather than edge detection, which
    tends to be more robust on thermal images (they often have lower
    texture/contrast noise than RGB, but Canny-based edges can be noisy on
    low-contrast backgrounds).

PIPELINE:
    1. Convert to grayscale (thermal images are usually single-channel or
       a false-color palette applied to a single-channel temperature map;
       we work on the intensity/grayscale representation either way).
    2. Slight Gaussian blur to reduce sensor noise.
    3. Otsu's method automatically picks an intensity threshold that best
       separates two populations of pixels (here: warm foreground vs. cool
       background) by minimizing within-class variance - a classical,
       non-learned thresholding technique.
    4. Morphological opening (erode then dilate) removes small bright noise
       specks; morphological closing fills small holes inside the person's
       silhouette (e.g. from clothing that isn't skin-temperature).
    5. Connected-component analysis picks the largest bright blob as the
       person (assumes the human is the dominant warm object in frame).
    6. Extract the boundary contour of that component.

REQUIREMENTS:
    pip install opencv-python numpy

USAGE:
    python thermal_segmentation.py --image thermal_person.jpg \
        --out result.png --mask_out mask.png

    If your thermal image is a false-color palette (e.g. red/yellow/white
    for hot, blue/purple for cool) rather than true grayscale, pass
    --false_color to convert it to a single "heat" channel first (see
    to_heat_channel() below) before thresholding.

OUTPUT:
    Prints the number of foreground pixels and bounding box of the
    detected person, and saves the visualization + binary mask, same as
    rgb_segmentation.py.

NOTE ON COMPARISON WITH SAM2:
    See compare_with_sam2.py to compute IoU/Dice between this result and
    Meta's SAM2 model output on the same thermal image. SAM2 is used only
    as an external comparison baseline, not part of this implementation.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 4 Assignment
==============================================================================
"""

import argparse
import cv2
import numpy as np


def to_heat_channel(image_bgr):
    """
    Convert a false-color thermal palette image (e.g. a typical
    black-blue-purple-red-yellow-white "ironbow" palette) into a single
    grayscale "heat" channel, using the pixel's saturation/value profile
    as a rough proxy for temperature intensity. Most false-color palettes
    increase in both hue-progression and brightness with temperature, so
    using the max of the RGB channels approximates relative heat level
    reasonably well without requiring the exact palette definition.
    """
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    heat = np.maximum(np.maximum(r, g), b)
    return cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def segment_human_thermal(image_bgr, false_color=False, min_area_frac=0.01):
    """
    Classical CV pipeline to segment the dominant warm human blob in a
    thermal image. Returns a binary mask (uint8, 0/255) and the boundary
    contour.
    """
    if false_color:
        gray = to_heat_channel(image_bgr)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: mild blur to reduce sensor noise before thresholding.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Otsu's method picks the threshold that minimizes intra-class
    # intensity variance between the two dominant populations (warm
    # foreground vs. cool background) - fully automatic, no learned model.
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: morphological cleanup - remove small noise specks (opening),
    # then fill small internal holes (closing).
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, large_kernel, iterations=2)

    # Step 5: connected-component analysis - keep only the largest blob
    # above a minimum area fraction of the image (filters out small hot
    # noise blobs like reflective objects).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No foreground blob found after thresholding - "
                            "try --false_color if this is a color-palette "
                            "thermal image, or check image contrast.")

    h, w = gray.shape
    min_area = min_area_frac * h * w
    # label 0 is background; find the largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = np.argmax(areas) + 1  # +1 to account for skipping background
    if stats[best_idx, cv2.CC_STAT_AREA] < min_area:
        raise RuntimeError("Largest detected blob is smaller than the "
                            "minimum area threshold - check image quality "
                            "or lower --min_area_frac.")

    mask = np.where(labels == best_idx, 255, 0).astype(np.uint8)

    # Step 6: extract the boundary contour.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = max(contours, key=cv2.contourArea)
    rect = cv2.boundingRect(boundary)

    return mask, boundary, rect


def main():
    parser = argparse.ArgumentParser(
        description="Segment a human's boundary from a thermal image using "
                    "classical CV (Otsu thresholding + connected components), "
                    "no ML/DL.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="thermal_result.png")
    parser.add_argument("--mask_out", default="thermal_mask.png")
    parser.add_argument("--false_color", action="store_true",
                         help="Set this if the thermal image uses a color "
                              "palette (e.g. ironbow) rather than true grayscale.")
    parser.add_argument("--min_area_frac", type=float, default=0.01)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    mask, boundary, rect = segment_human_thermal(
        img, false_color=args.false_color, min_area_frac=args.min_area_frac)

    vis = img.copy()
    cv2.drawContours(vis, [boundary], -1, (0, 255, 0), 3)
    x, y, w, h = rect
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imwrite(args.out, vis)
    cv2.imwrite(args.mask_out, mask)

    fg_pixels = int(np.sum(mask == 255))
    print(f"Foreground pixels: {fg_pixels}")
    print(f"Bounding box (x,y,w,h): {rect}")
    print(f"Saved visualization to: {args.out}")
    print(f"Saved binary mask to: {args.mask_out}")


if __name__ == "__main__":
    main()
