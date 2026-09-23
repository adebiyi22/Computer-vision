"""
==============================================================================
CSc 8830 - Computer Vision - Module 2 Assignment - Step 2
Real-World 2D Dimension Estimation via Perspective Projection

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Given a calibrated camera (intrinsic matrix K from Step 1), the known
    distance Z from the camera to a (roughly planar / fronto-parallel)
    object, and two pixel coordinates on that object (e.g. the two ends of
    an edge you want to measure), this script computes the real-world
    distance between those two points using the pinhole camera / perspective
    projection model.

THEORY (pinhole projection, inverted):
    Forward projection of a 3D point (X, Y, Z) to pixel (u, v):
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

    Inverting, given pixel (u, v) and known depth Z:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    So for two pixel points p1=(u1,v1), p2=(u2,v2) on an object at the same
    depth Z, their corresponding real-world (X, Y) coordinates can be
    recovered, and the real-world Euclidean distance between them is:
        D = sqrt((X2-X1)^2 + (Y2-Y1)^2)

    This assumes:
      (a) Both points lie at (approximately) the same depth Z from the
          camera (fronto-parallel object, or the object is thin/flat
          relative to Z).
      (b) Lens distortion has been corrected (we undistort pixel points
          using the distortion coefficients from calibration before
          applying the pinhole equations above).

REQUIREMENTS:
    pip install opencv-python numpy

USAGE (two ways to supply the pixel points):

  1) Interactive click mode (click two points on the displayed image):
      python perspective_dimension.py --calib calibration_result.npz \
                                       --image object.jpg \
                                       --distance 2500 \
                                       --interactive

  2) Direct coordinate mode (skip the GUI, pass pixel coords directly):
      python perspective_dimension.py --calib calibration_result.npz \
                                       --image object.jpg \
                                       --distance 2500 \
                                       --p1 120 340 --p2 560 340

  --distance is the real-world distance (in mm) from the camera to the
  object's plane, measured independently (e.g. tape measure / laser
  distance meter) at the time the photo was taken.

OUTPUT:
    Prints the undistorted pixel coordinates, the recovered real-world
    (X, Y) coordinates of each point (mm, relative to the camera's optical
    axis), and the real-world Euclidean distance between the two points
    (mm and cm).

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 2 Assignment
==============================================================================
"""

import argparse

import cv2
import numpy as np


def load_calibration(npz_path):
    """Load intrinsic matrix K and distortion coefficients from Step 1's output."""
    data = np.load(npz_path, allow_pickle=True)
    K = data["K"]
    dist = data["dist"]
    return K, dist


def undistort_pixel_points(points, K, dist):
    """
    Correct lens distortion for a set of pixel points, returning points as
    if they had been captured by an ideal distortion-free pinhole camera
    (still in pixel coordinates, using K).

    Args:
        points: Nx2 array of (u, v) pixel coordinates.
        K: 3x3 intrinsic matrix.
        dist: distortion coefficients.

    Returns:
        Nx2 array of undistorted (u, v) pixel coordinates.
    """
    pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    # P=K re-projects the undistorted normalized points back into pixel
    # coordinates using the same intrinsics, so the output stays in pixels.
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return undistorted.reshape(-1, 2)


def pixel_to_real_world(u, v, Z, K):
    """
    Convert a single (undistorted) pixel coordinate + known depth Z into
    real-world (X, Y) coordinates (in the same units as Z), relative to
    the camera's optical center, using the inverted pinhole equations.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y


def measure_distance(p1_px, p2_px, Z, K, dist):
    """
    Full pipeline: undistort two pixel points, back-project them to
    real-world (X, Y) at depth Z, and compute the Euclidean distance
    between them.

    Args:
        p1_px, p2_px: (u, v) pixel coordinate tuples.
        Z: known real-world distance from camera to the object plane (mm).
        K: 3x3 intrinsic matrix.
        dist: distortion coefficients.

    Returns:
        dict with undistorted pixel coords, real-world (X,Y) for each
        point, and the real-world distance in mm.
    """
    undistorted = undistort_pixel_points([p1_px, p2_px], K, dist)
    (u1, v1), (u2, v2) = undistorted

    X1, Y1 = pixel_to_real_world(u1, v1, Z, K)
    X2, Y2 = pixel_to_real_world(u2, v2, Z, K)

    distance_mm = float(np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2))

    return {
        "undistorted_p1": (u1, v1),
        "undistorted_p2": (u2, v2),
        "real_p1_mm": (X1, Y1),
        "real_p2_mm": (X2, Y2),
        "distance_mm": distance_mm,
    }


# --- Interactive click-based point selection -------------------------------

_click_points = []


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_click_points) < 2:
        _click_points.append((x, y))


def select_two_points_interactively(image_path):
    """Open a window and let the user click two points on the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    window_name = "Click 2 points to measure (press 'r' to reset, 'q' to quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)

    while True:
        vis = img.copy()
        for i, pt in enumerate(_click_points):
            cv2.circle(vis, pt, 6, (0, 0, 255), -1)
            cv2.putText(vis, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(_click_points) == 2:
            cv2.line(vis, _click_points[0], _click_points[1], (0, 255, 0), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            _click_points.clear()
        elif key == ord('q') or len(_click_points) == 2:
            break

    cv2.waitKey(500)
    cv2.destroyAllWindows()

    if len(_click_points) != 2:
        raise RuntimeError("Two points were not selected.")
    return _click_points[0], _click_points[1]


def main():
    parser = argparse.ArgumentParser(
        description="Compute real-world 2D distance between two image points "
                    "using perspective projection."
    )
    parser.add_argument("--calib", type=str, required=True,
                         help="Path to calibration_result.npz from Step 1.")
    parser.add_argument("--image", type=str, required=True,
                         help="Path to the image containing the object.")
    parser.add_argument("--distance", type=float, required=True,
                         help="Known real-world distance from camera to the "
                              "object plane, in millimeters.")
    parser.add_argument("--p1", type=float, nargs=2, metavar=("U1", "V1"),
                         help="Pixel coordinate of point 1 (skip --interactive).")
    parser.add_argument("--p2", type=float, nargs=2, metavar=("U2", "V2"),
                         help="Pixel coordinate of point 2 (skip --interactive).")
    parser.add_argument("--interactive", action="store_true",
                         help="Click the two points on the image instead of "
                              "passing --p1/--p2.")
    args = parser.parse_args()

    K, dist = load_calibration(args.calib)

    if args.interactive:
        p1, p2 = select_two_points_interactively(args.image)
    else:
        if args.p1 is None or args.p2 is None:
            raise ValueError("Provide --p1 and --p2, or use --interactive.")
        p1, p2 = tuple(args.p1), tuple(args.p2)

    result = measure_distance(p1, p2, args.distance, K, dist)

    print("=" * 70)
    print("REAL-WORLD DIMENSION MEASUREMENT")
    print("=" * 70)
    print(f"Known camera-to-object distance (Z): {args.distance:.2f} mm")
    print(f"Point 1 (pixel, undistorted):  {result['undistorted_p1']}")
    print(f"Point 2 (pixel, undistorted):  {result['undistorted_p2']}")
    print(f"Point 1 real-world (X, Y) mm:  {result['real_p1_mm']}")
    print(f"Point 2 real-world (X, Y) mm:  {result['real_p2_mm']}")
    print(f"\nReal-world distance: {result['distance_mm']:.2f} mm "
          f"({result['distance_mm']/10:.2f} cm)")


if __name__ == "__main__":
    main()
