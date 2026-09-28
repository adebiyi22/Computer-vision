"""
==============================================================================
CSc 8830 - Computer Vision - Module 2 Assignment - Step 1
Camera Calibration using OpenCV

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Calibrates a smartphone camera using a checkerboard (chessboard) pattern.
    Computes the camera intrinsic matrix (K), distortion coefficients,
    and per-image extrinsic parameters (rotation & translation vectors),
    then reports the mean reprojection error.

REQUIREMENTS:
    pip install opencv-python numpy

PREPARATION (before running):
    1. Print or display a chessboard pattern (default here: 9x6 internal
       corners, i.e. a 10x7 squares board). A standard printable pattern
       can be found at: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
    2. Measure the real-world size of ONE square on your printed pattern
       in millimeters (e.g. 25.0 mm) and set SQUARE_SIZE_MM below.
    3. Using your smartphone camera, take 15-20 photos of the checkerboard:
       - Vary the angle (tilt it), distance, and position in the frame.
       - Keep the entire checkerboard visible and in focus in every shot.
       - Keep the same camera/lens (no zoom changes) for all photos.
    4. Place all the photos into a folder, e.g. ./calib_images/
       (accepted formats: .jpg, .jpeg, .png)

USAGE:
    python camera_calibration.py --images_dir ./calib_images \
                                  --cols 9 --rows 6 \
                                  --square_size 25.0 \
                                  --out calibration_result.npz

OUTPUT:
    - Prints the camera intrinsic matrix K, distortion coefficients,
      per-image reprojection errors, and the overall mean reprojection error.
    - Saves K, distortion coefficients, rvecs, and tvecs to the specified
      .npz file (default: calibration_result.npz) for later use in Step 2.
    - Optionally saves annotated images with detected corners drawn on them
      into a subfolder "detected_corners/" inside images_dir, for visual
      verification / inclusion as evidence in your report.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 2 Assignment
==============================================================================
"""

import argparse
import glob
import os

import cv2
import numpy as np


def find_image_files(images_dir):
    """Collect all supported image files from the given directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)


def calibrate_camera(images_dir, pattern_cols, pattern_rows, square_size_mm,
                      save_corner_images=True):
    """
    Run OpenCV chessboard-based camera calibration.

    Args:
        images_dir: folder containing calibration checkerboard photos.
        pattern_cols: number of INTERNAL corners along the checkerboard width.
        pattern_rows: number of INTERNAL corners along the checkerboard height.
        square_size_mm: real-world edge length of one checkerboard square (mm).
        save_corner_images: if True, saves images with detected corners drawn,
            for visual verification.

    Returns:
        dict with calibration results (K, dist, rvecs, tvecs, per_image_errors,
        mean_error, image_size, used_images).
    """
    # Termination criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare the "object points": the known 3D coordinates of the
    # checkerboard corners in the checkerboard's own coordinate frame,
    # assuming Z = 0 (the board is planar). These are the same for every
    # image, scaled by the real-world square size.
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []  # 3D points in real-world space, one array per valid image
    imgpoints = []  # 2D points in image plane, one array per valid image
    used_images = []
    image_size = None

    image_files = find_image_files(images_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in directory: {images_dir}")

    corner_dir = os.path.join(images_dir, "detected_corners")
    if save_corner_images:
        os.makedirs(corner_dir, exist_ok=True)

    print(f"Found {len(image_files)} candidate images. Searching for "
          f"{pattern_cols}x{pattern_rows} checkerboard corners...\n")

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"  [SKIP] Could not read image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        # Locate the checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray, (pattern_cols, pattern_rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used_images.append(fname)
            print(f"  [OK]   {os.path.basename(fname)}")

            if save_corner_images:
                vis = img.copy()
                cv2.drawChessboardCorners(
                    vis, (pattern_cols, pattern_rows), corners_refined, found
                )
                out_path = os.path.join(corner_dir, os.path.basename(fname))
                cv2.imwrite(out_path, vis)
        else:
            print(f"  [FAIL] Checkerboard not found: {os.path.basename(fname)}")

    if len(objpoints) < 5:
        raise RuntimeError(
            f"Only {len(objpoints)} valid images found. Need at least "
            f"5-10 good detections for a stable calibration; 15-20 is "
            f"recommended."
        )

    print(f"\nUsing {len(objpoints)} valid images for calibration.\n")

    # Run calibration: solves for the intrinsic matrix K, distortion
    # coefficients, and per-image extrinsics (rotation + translation)
    # that best explain the observed 2D corner locations.
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Compute per-image reprojection error: project the known 3D object
    # points back into the image using the estimated parameters, and
    # compare against the actually detected 2D corners.
    per_image_errors = []
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )
        # Use numpy (rather than cv2.norm) to compute the error: some OpenCV
        # builds enforce strict, exact-matching internal array types (e.g.
        # single-channel vs two-channel) for cv2.norm's two inputs, which can
        # raise a type-mismatch error even though both arrays represent the
        # same Nx2 point data. Casting to plain float64 numpy arrays avoids
        # that OpenCV-version-specific incompatibility entirely.
        detected = np.asarray(imgpoints[i], dtype=np.float64).reshape(-1, 2)
        projected = np.asarray(imgpoints_proj, dtype=np.float64).reshape(-1, 2)
        error = float(np.linalg.norm(detected - projected)) / len(projected)
        per_image_errors.append(error)

    mean_error = float(np.mean(per_image_errors))

    return {
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_image_errors": per_image_errors,
        "mean_error": mean_error,
        "image_size": image_size,
        "used_images": used_images,
        "overall_rms": ret,  # OpenCV's own overall RMS re-projection error
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a smartphone camera using a checkerboard pattern."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                         help="Folder containing checkerboard calibration photos.")
    parser.add_argument("--cols", type=int, default=9,
                         help="Number of INTERNAL corners along board width (default: 9).")
    parser.add_argument("--rows", type=int, default=6,
                         help="Number of INTERNAL corners along board height (default: 6).")
    parser.add_argument("--square_size", type=float, required=True,
                         help="Real-world size of one checkerboard square, in mm.")
    parser.add_argument("--out", type=str, default="calibration_result.npz",
                         help="Output file to save calibration results (.npz).")
    args = parser.parse_args()

    results = calibrate_camera(
        images_dir=args.images_dir,
        pattern_cols=args.cols,
        pattern_rows=args.rows,
        square_size_mm=args.square_size,
    )

    np.savez(
        args.out,
        K=results["K"],
        dist=results["dist"],
        rvecs=np.array(results["rvecs"], dtype=object),
        tvecs=np.array(results["tvecs"], dtype=object),
        image_size=results["image_size"],
    )

    print("=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print("Intrinsic matrix K:\n", results["K"])
    print("\nDistortion coefficients (k1, k2, p1, p2, k3):\n", results["dist"].ravel())
    print(f"\nOpenCV overall RMS reprojection error: {results['overall_rms']:.4f} px")
    print(f"Mean per-image reprojection error:     {results['mean_error']:.4f} px")
    print("\nPer-image reprojection errors (px):")
    for fname, err in zip(results["used_images"], results["per_image_errors"]):
        print(f"  {os.path.basename(fname):30s} {err:.4f}")

    print(f"\nSaved calibration parameters to: {args.out}")
    print("Note: extract K[0,0] and K[1,1] as fx, fy (focal length in pixels),")
    print("and K[0,2], K[1,2] as the principal point (cx, cy). These will be")
    print("used directly in Step 2 (perspective projection dimension script).")


if __name__ == "__main__":
    main()
