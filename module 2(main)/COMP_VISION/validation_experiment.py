"""
==============================================================================
CSc 8830 - Computer Vision - Module 2 Assignment - Step 3
Validation Experiment: Statistical Error Analysis over 20 Measurements

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Runs the Step 2 dimension-measurement pipeline over a batch of trials
    (20+ recommended) where you have both:
      (a) the pixel coordinates of the two measurement points in a photo, and
      (b) the ground-truth real-world measurement (obtained with a tape
          measure / ruler), for the same object edge.
    It computes the predicted measurement for each trial, compares it to
    ground truth, and reports error statistics (mean error, MAE, RMSE,
    standard deviation, percentage error) across all trials.

EXPERIMENT PROTOCOL (how to collect the 20 data points):
    1. Choose one fixed camera-to-object distance greater than 2 meters
       (e.g., 2.5 m) and measure/verify it accurately (tape measure, laser
       distance meter, or known floor markings).
    2. Photograph 20 different objects (or 20 different edges on objects)
       from that fixed distance, keeping the camera perpendicular to the
       object plane as closely as possible.
    3. For each photo, measure the object's true dimension by hand (ruler /
       tape measure) - this is your ground truth.
    4. For each photo, identify the pixel coordinates of the two endpoints
       of the measured edge (e.g., using the --interactive mode in
       perspective_dimension.py, or an image viewer that reports cursor
       pixel coordinates).
    5. Fill in these values into a CSV file (see the template this script
       can generate) with columns:
           image_path, u1, v1, u2, v2, distance_mm, ground_truth_mm

USAGE:
    # Step A: generate a blank CSV template to fill in
    python validation_experiment.py --make_template trials_template.csv

    # Step B: after filling in the template with your 20 trials, run:
    python validation_experiment.py --calib calibration_result.npz \
                                     --trials trials_filled.csv \
                                     --out results.csv

OUTPUT:
    - A results CSV with per-trial predicted measurement, absolute error,
      and percentage error.
    - Printed summary statistics: mean error, mean absolute error (MAE),
      root-mean-square error (RMSE), standard deviation of error, mean
      percentage error, and min/max error - suitable for direct inclusion
      in your report.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 2 Assignment
==============================================================================
"""

import argparse
import csv

import numpy as np

from perspective_dimension import load_calibration, measure_distance

TEMPLATE_HEADERS = [
    "image_path", "u1", "v1", "u2", "v2", "distance_mm", "ground_truth_mm"
]


def make_template(path, n_rows=20):
    """Write a blank CSV template with n_rows empty rows for the user to fill in."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TEMPLATE_HEADERS)
        for _ in range(n_rows):
            writer.writerow(["" for _ in TEMPLATE_HEADERS])
    print(f"Template with {n_rows} blank rows written to: {path}")
    print("Fill in each row with:")
    print("  image_path     - path to the photo for this trial")
    print("  u1, v1, u2, v2 - pixel coords of the two measurement points")
    print("  distance_mm    - known camera-to-object distance for this trial (mm)")
    print("  ground_truth_mm- true measurement from a physical ruler/tape (mm)")


def load_trials(path):
    """Load filled-in trial data from CSV."""
    trials = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["image_path"]:
                continue  # skip blank rows
            trials.append({
                "image_path": row["image_path"],
                "p1": (float(row["u1"]), float(row["v1"])),
                "p2": (float(row["u2"]), float(row["v2"])),
                "distance_mm": float(row["distance_mm"]),
                "ground_truth_mm": float(row["ground_truth_mm"]),
            })
    return trials


def run_validation(calib_path, trials_path, out_path):
    K, dist = load_calibration(calib_path)
    trials = load_trials(trials_path)

    if len(trials) == 0:
        raise ValueError("No valid trials found in the CSV file.")

    results = []
    for i, trial in enumerate(trials):
        measurement = measure_distance(trial["p1"], trial["p2"],
                                        trial["distance_mm"], K, dist)
        predicted = measurement["distance_mm"]
        gt = trial["ground_truth_mm"]
        error = predicted - gt
        abs_error = abs(error)
        pct_error = 100.0 * abs_error / gt if gt != 0 else float("nan")

        results.append({
            "trial": i + 1,
            "image_path": trial["image_path"],
            "predicted_mm": predicted,
            "ground_truth_mm": gt,
            "error_mm": error,
            "abs_error_mm": abs_error,
            "pct_error": pct_error,
        })

    # Write per-trial results
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    errors = np.array([r["error_mm"] for r in results])
    abs_errors = np.array([r["abs_error_mm"] for r in results])
    pct_errors = np.array([r["pct_error"] for r in results])

    stats = {
        "n_trials": len(results),
        "mean_error_mm": float(np.mean(errors)),
        "mae_mm": float(np.mean(abs_errors)),
        "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "std_error_mm": float(np.std(errors)),
        "mean_pct_error": float(np.mean(pct_errors)),
        "min_abs_error_mm": float(np.min(abs_errors)),
        "max_abs_error_mm": float(np.max(abs_errors)),
    }

    return results, stats


def print_stats(stats):
    print("=" * 70)
    print("VALIDATION SUMMARY STATISTICS (n = {})".format(stats["n_trials"]))
    print("=" * 70)
    print(f"Mean signed error:      {stats['mean_error_mm']:.2f} mm")
    print(f"Mean absolute error:    {stats['mae_mm']:.2f} mm")
    print(f"RMSE:                   {stats['rmse_mm']:.2f} mm")
    print(f"Std dev of error:       {stats['std_error_mm']:.2f} mm")
    print(f"Mean percentage error:  {stats['mean_pct_error']:.2f} %")
    print(f"Min / Max abs error:    {stats['min_abs_error_mm']:.2f} / "
          f"{stats['max_abs_error_mm']:.2f} mm")


def main():
    parser = argparse.ArgumentParser(
        description="Run the 20-trial validation experiment and report error statistics."
    )
    parser.add_argument("--make_template", type=str,
                         help="Write a blank CSV template to this path and exit.")
    parser.add_argument("--calib", type=str,
                         help="Path to calibration_result.npz from Step 1.")
    parser.add_argument("--trials", type=str,
                         help="Path to filled-in trials CSV.")
    parser.add_argument("--out", type=str, default="results.csv",
                         help="Path to write per-trial results CSV.")
    args = parser.parse_args()

    if args.make_template:
        make_template(args.make_template)
        return

    if not args.calib or not args.trials:
        raise ValueError("--calib and --trials are required unless using --make_template.")

    results, stats = run_validation(args.calib, args.trials, args.out)
    print_stats(stats)
    print(f"\nPer-trial results written to: {args.out}")


if __name__ == "__main__":
    main()
