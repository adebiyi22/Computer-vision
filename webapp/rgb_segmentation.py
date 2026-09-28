"""
==============================================================================
CSc 8830 - Computer Vision - Module 4 Assignment - Part 1
Human Boundary Segmentation from an RGB Image (Classical CV only, NO ML/DL)

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Finds the boundary/outline (silhouette) of a human in a regular color
    photo using only classical, non-learned OpenCV operations: edge
    detection, thresholding, morphology, connected components, and the
    GrabCut graph-cut refinement algorithm. GrabCut is a per-image
    energy-minimization method (it fits a Gaussian Mixture Model to the
    single input image at runtime) - it is NOT a trained/learned model and
    uses no external training data, so it complies with the assignment's
    "no deep learning / no machine learning" constraint.

PIPELINE:
    1. Convert to grayscale and reduce noise (bilateral filter - preserves
       edges while smoothing flat regions).
    2. Edge detection (Canny) to find strong intensity discontinuities.
    3. Morphological closing to bridge small gaps in the edge map so the
       silhouette forms a connected boundary.
    4. Find contours; assume the largest contour by area is the human
       (reasonable when the person is the dominant foreground subject).
    5. Use that contour's bounding box to initialize GrabCut, which refines
       the rough edge-based mask into a clean foreground/background
       segmentation using iterative graph-cut energy minimization.
    6. Extract the final boundary contour from the refined mask.

REQUIREMENTS:
    pip install opencv-python numpy

USAGE:
    python rgb_segmentation.py --image person.jpg --out result.png \
        --mask_out mask.png

    --image     path to an RGB photo containing one clearly foregrounded
                person (works best when the person isn't tiny in the frame
                and has some contrast against the background)
    --out       path to save the visualization (original + boundary overlay)
    --mask_out  path to save the binary segmentation mask (white=person)

OUTPUT:
    Prints the number of foreground pixels and the bounding box of the
    detected person, and saves:
      - <out>: original image with the detected boundary drawn in green
      - <mask_out>: binary mask (255=person, 0=background)

NOTE ON COMPARISON WITH SAM2:
    See compare_with_sam2.py in this folder to run Meta's SAM2 model on the
    same image and compute IoU/Dice overlap between this classical result
    and SAM2's segmentation. SAM2 itself is a deep-learning model and is
    used here ONLY as an external comparison baseline, not as part of this
    script's own implementation.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 4 Assignment
==============================================================================
"""

import argparse
import cv2
import numpy as np


def auto_canny(gray, sigma=0.33):
    """
    Automatically choose Canny thresholds based on the median pixel
    intensity of the image, a well-known robust heuristic (avoids manually
    tuned thresholds failing on images with different contrast/lighting).
    """
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def segment_human_rgb(image_bgr, canny_low=None, canny_high=None):
    """
    Classical CV pipeline to segment the dominant human silhouette in an
    RGB image. Returns a binary mask (uint8, 0/255) and the boundary
    contour (Nx1x2 array of points).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 1: edge-preserving smoothing to reduce noise without blurring
    # the silhouette boundary itself. A moderate sigmaColor is used so
    # genuine object boundaries (even moderate-contrast ones) are kept
    # while flat-region sensor/background noise is suppressed.
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=75)

    # Step 2: Canny edge detection. If explicit thresholds were passed use
    # them; otherwise auto-select thresholds from the image's own median
    # intensity, which is far more robust across varying lighting/contrast
    # than a single fixed threshold pair.
    if canny_low is not None and canny_high is not None:
        edges = cv2.Canny(smoothed, canny_low, canny_high)
    else:
        edges = auto_canny(smoothed)

    # Fallback: if smoothing suppressed the boundary gradient too much
    # (can happen on lower-contrast images), retry directly on the
    # unsmoothed grayscale image.
    if np.sum(edges > 0) < 200:
        edges = auto_canny(gray) if (canny_low is None or canny_high is None) \
            else cv2.Canny(gray, canny_low, canny_high)

    # Step 3: morphological closing (dilate then erode) to bridge small
    # gaps in the edge map so the person's outline becomes one connected
    # boundary rather than broken fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    closed = cv2.dilate(closed, kernel, iterations=1)

    # Step 4: find contours and take the largest by area as the initial
    # rough estimate of the person's location/extent.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found - try adjusting Canny thresholds "
                            "or check the image has sufficient contrast.")
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Pad the bounding box slightly so GrabCut has some background context
    # around the person to learn from.
    H, W = gray.shape
    pad_x, pad_y = int(0.08 * w), int(0.08 * h)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)
    rect = (x0, y0, x1 - x0, y1 - y0)

    # Step 5: refine with GrabCut. GrabCut fits foreground/background color
    # Gaussian Mixture Models from the pixels inside/outside the rectangle
    # and iteratively refines the segmentation via min-cut/max-flow graph
    # optimization - a classical (non-learned) energy-minimization method.
    mask = np.zeros(gray.shape, np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # GrabCut labels: 0=bg, 1=fg, 2=probable bg, 3=probable fg
    binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # Step 6: clean up small noise and extract the final boundary contour.
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    final_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    if not final_contours:
        raise RuntimeError("GrabCut refinement produced an empty mask.")
    boundary = max(final_contours, key=cv2.contourArea)

    return binary_mask, boundary, rect


def main():
    parser = argparse.ArgumentParser(
        description="Segment a human's boundary from an RGB photo using "
                    "classical CV (Canny + GrabCut), no ML/DL.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="rgb_result.png")
    parser.add_argument("--mask_out", default="rgb_mask.png")
    parser.add_argument("--canny_low", type=int, default=None,
                         help="Optional manual override; default is automatic "
                              "median-based threshold selection.")
    parser.add_argument("--canny_high", type=int, default=None)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    mask, boundary, rect = segment_human_rgb(img, args.canny_low, args.canny_high)

    vis = img.copy()
    cv2.drawContours(vis, [boundary], -1, (0, 255, 0), 3)
    x, y, w, h = rect
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imwrite(args.out, vis)
    cv2.imwrite(args.mask_out, mask)

    fg_pixels = int(np.sum(mask == 255))
    print(f"Foreground pixels: {fg_pixels}")
    print(f"Initial bounding box (x,y,w,h): {rect}")
    print(f"Saved visualization to: {args.out}")
    print(f"Saved binary mask to: {args.mask_out}")


if __name__ == "__main__":
    main()
