# CSc 8830 — Computer Vision — Module 2 Assignment

Camera calibration, perspective-projection dimension estimation, statistical
validation, and two-camera epipolar theory.

## Contents

| File | Purpose |
|---|---|
| `camera_calibration.py` | Step 1: OpenCV checkerboard camera calibration |
| `perspective_dimension.py` | Step 2: real-world 2D dimension from a single image + known depth |
| `validation_experiment.py` | Step 3: batch validation over 20 trials + error statistics |
| `theory_derivation.md` | Theory: two-camera image-coordinate relationship (epipolar geometry) |
| `webapp/` | Flask web app exposing Steps 1–2 through a browser UI |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install opencv-python numpy flask
```

## Execution order

### Step 1 — Calibrate your camera
1. Print the OpenCV checkerboard pattern (or any checkerboard) and measure
   one square's real edge length in mm.
2. Take 15–20 photos with your smartphone, varying angle/distance/position.
3. Run:
   ```bash
   python camera_calibration.py --images_dir ./calib_images \
       --cols 9 --rows 6 --square_size 25.0 --out calibration_result.npz
   ```
4. Confirm the mean reprojection error is small (well under 1 px is ideal).
   Include the printed K matrix, distortion coefficients, and error stats
   in your report, plus a couple of the `detected_corners/` images as
   visual evidence.

### Step 2 — Measure a real-world dimension
```bash
python perspective_dimension.py --calib calibration_result.npz \
    --image object.jpg --distance 2500 --interactive
```
Click the two endpoints of the edge you want measured; the script prints
the real-world distance in mm/cm. (`--distance` is the known camera-to-object
distance in mm, measured independently at capture time.)

### Step 3 — Validate with 20 measurements
```bash
# generate a blank CSV to fill in
python validation_experiment.py --make_template trials_template.csv

# after filling in 20 rows (image path, pixel points, distance, ground truth):
python validation_experiment.py --calib calibration_result.npz \
    --trials trials_filled.csv --out results.csv
```
This prints mean error, MAE, RMSE, standard deviation, and mean percentage
error across all 20 trials — include this table/summary directly in your
report.

### Theory
See `theory_derivation.md` for the full derivation of the relationship
between image coordinates of a 3D point P as seen by two cameras (one
static, one at an oblique offset), including the epipolar constraint,
fundamental/essential matrices, and a clear list of which parameters are
static (intrinsics K1/K2, extrinsics R/t) versus per-point variables
((u,v) pixel coordinates, 3D position). Convert this into typed/scanned
form as required and append it to your final PDF.

### Web application demo
```bash
cd webapp
python app.py
```
Open `http://localhost:5000`. The page lets you:
1. Upload checkerboard photos and run calibration in the browser (Step 1).
2. Upload an object photo, enter the known distance, and click two points
   to get the real-world measurement (Step 2), reusing the calibration
   from step 1 in the same session.

For your submission, deploy this (or run it on a machine reachable over
your network) so it is "accessible via a webpage" as required, and record
a screen capture of you exercising both steps as your demonstration video.


## Notes on the perspective-projection model used

Steps 2–3 assume the measured object lies at a single known depth Z from
the camera (a fronto-parallel or thin planar object) and undistort pixel
coordinates using the Step 1 calibration before applying the inverted
pinhole equations:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Real-world distance between two points is then the Euclidean distance
between their recovered (X, Y) coordinates. This is documented in detail
in the docstring at the top of `perspective_dimension.py`.
