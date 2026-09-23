"""
Builds the Module 2 assignment PDF report.

USAGE:
    Put your REAL calibration_result.npz and results.csv (produced by
    running camera_calibration.py and validation_experiment.py on your own
    photos) in this same folder, then run:

        python3 build_report.py

    If those files are present, their real numbers are read and inserted
    into the report automatically. If they are not found, the report is
    still built, but with a clearly marked placeholder telling you to add
    your own results before submitting - this script never invents
    experimental numbers, since Step 3 requires real physical measurements.
"""
import os
import csv
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
    Table, TableStyle, ListFlowable, ListItem
)

# ---------------------------------------------------------------------------
# EDIT THESE before building
# ---------------------------------------------------------------------------
GITHUB_LINK = "PASTE_YOUR_GITHUB_REPO_LINK_HERE"
WEBAPP_LINK = "PASTE_YOUR_DEPLOYED_WEB_APP_LINK_HERE"
CALIBRATION_NPZ = "calibration_result.npz"   # put your real file here, or leave missing
VALIDATION_CSV = "results.csv"               # put your real file here, or leave missing
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8.5,
                           leading=11, backColor=colors.whitesmoke,
                           borderPadding=6, spaceAfter=8, spaceBefore=4))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=16))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=12))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5,
                           textColor=colors.grey))
styles.add(ParagraphStyle(name="Warn", parent=styles["Normal"], fontSize=10,
                           textColor=colors.HexColor("#8a4b00"),
                           backColor=colors.HexColor("#fff3e0"),
                           borderPadding=8, spaceAfter=10, spaceBefore=6))

story = []

# ---- Title ----
story.append(Paragraph("CSc 8830: Computer Vision", styles["Title"]))
story.append(Paragraph("Module 2 Assignment", styles["Heading2"]))
story.append(Paragraph("Camera Calibration, Perspective-Projection Dimension "
                        "Estimation, and Two-Camera Epipolar Theory", styles["Heading3"]))
story.append(Spacer(1, 12))
story.append(Paragraph(f"<b>GitHub repository:</b> {GITHUB_LINK}", styles["Body"]))
story.append(Paragraph(f"<b>Deployed web application:</b> {WEBAPP_LINK}", styles["Body"]))
story.append(Spacer(1, 12))

# ---- Section 1: Assignment ----
story.append(Paragraph("1. Assignment", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph("<b>Step 1:</b> Perform camera calibration using built-in "
                        "OpenCV/MATLAB tools, using a smartphone camera.", styles["Body"])),
    ListItem(Paragraph("<b>Step 2:</b> Implement a script to find the real-world 2D "
                        "dimensions of an object using perspective projection "
                        "equations.", styles["Body"])),
    ListItem(Paragraph("<b>Step 3:</b> Validate using an experiment imaging an object "
                        "from a fixed distance greater than 2 meters, across 20 "
                        "different object measurements, reporting error statistics.",
                        styles["Body"])),
    ListItem(Paragraph("<b>Theory:</b> Derive the relationship between the image "
                        "coordinates of a point P(X,Y,Z) as seen by two cameras "
                        "(camera 1 static, camera 2 at an oblique offset), with "
                        "clarity on static parameters, variables, and how each is "
                        "determined.", styles["Body"])),
]))

# ---- Section 2: Implementation ----
story.append(Paragraph("2. Implementation", styles["H1"]))

story.append(Paragraph("2.1 Step 1 — Camera Calibration", styles["H2"]))
story.append(Paragraph(
    "<font face='Courier'>camera_calibration.py</font> calibrates a smartphone "
    "camera using OpenCV's chessboard-based calibration: 15-20 photos of a "
    "checkerboard are taken from varied angles/distances, "
    "<font face='Courier'>cv2.findChessboardCorners</font> and "
    "<font face='Courier'>cornerSubPix</font> locate the corners to sub-pixel "
    "precision, and <font face='Courier'>cv2.calibrateCamera</font> solves for "
    "the intrinsic matrix K, distortion coefficients, and per-image extrinsics. "
    "Reprojection error is computed per image to validate the calibration "
    "quality.", styles["Body"]))

story.append(Paragraph("2.2 Step 2 — Perspective-Projection Dimension Estimation", styles["H2"]))
story.append(Paragraph(
    "<font face='Courier'>perspective_dimension.py</font> inverts the pinhole "
    "projection model: given a known camera-to-object distance Z and pixel "
    "coordinates (u,v) (undistorted using the Step 1 calibration), it recovers "
    "real-world (X,Y) via X=(u-cx)Z/fx, Y=(v-cy)Z/fy, then computes the "
    "Euclidean distance between two such points.", styles["Body"]))
code2 = """def pixel_to_real_world(u, v, Z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y"""
story.append(Paragraph(code2.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["Mono"]))

story.append(Paragraph("2.3 Step 3 — Statistical Validation", styles["H2"]))
story.append(Paragraph(
    "<font face='Courier'>validation_experiment.py</font> runs the Step 2 "
    "pipeline across a batch of trials logged in a CSV (image, pixel points, "
    "distance, ground truth), and reports mean error, MAE, RMSE, standard "
    "deviation, and mean percentage error across all trials.", styles["Body"]))

story.append(Paragraph("2.4 Web Application", styles["H2"]))
story.append(Paragraph(
    "A Flask web app (<font face='Courier'>webapp/app.py</font>) exposes both "
    "Steps 1-2 through a browser: uploading checkerboard photos runs "
    "calibration and returns K/distortion/reprojection error; uploading an "
    "object photo and clicking two points on an HTML canvas returns the "
    "real-world measured distance using the calibration from the same "
    "session.", styles["Body"]))

story.append(PageBreak())

# ---- Section 3: Theory ----
story.append(Paragraph("3. Theory: Relating Two Camera Views of Point P", styles["H1"]))

story.append(Paragraph("3.1 Setup", styles["H2"]))
story.append(Paragraph(
    "Camera 1 (C1) is static and defines the world frame; camera 2 (C2) is "
    "displaced by rotation R and translation t. A point P=(X,Y,Z) in C1's frame "
    "is visible to both cameras. Each camera's intrinsics (K1, K2) are known "
    "from individual calibration (Step 1); R and t are determined once via "
    "stereo calibration.", styles["Body"]))

story.append(Paragraph("3.2 Projections and the relationship", styles["H2"]))
story.append(Paragraph("Camera 1 projects P directly (pinhole model):", styles["Body"]))
story.append(Paragraph("u1 = fx1&middot;(X/Z) + cx1 ,&nbsp;&nbsp; v1 = fy1&middot;(Y/Z) + cy1", styles["Mono"]))
story.append(Paragraph("P is transformed into camera 2's frame via the rigid transform:", styles["Body"]))
story.append(Paragraph("P2 = R&middot;P + t &nbsp;&rArr;&nbsp; (X2, Y2, Z2)", styles["Mono"]))
story.append(Paragraph("Camera 2 then projects P2 with its own intrinsics:", styles["Body"]))
story.append(Paragraph("u2 = fx2&middot;(X2/Z2) + cx2 ,&nbsp;&nbsp; v2 = fy2&middot;(Y2/Z2) + cy2", styles["Mono"]))
story.append(Paragraph(
    "Combining these (inverting camera 1's equations for X,Y given depth Z, "
    "substituting into the rigid transform, then into camera 2's projection) "
    "expresses (u2,v2) directly in terms of (u1,v1,Z). If Z is unknown, "
    "(u2,v2) is constrained to lie on the corresponding <b>epipolar line</b>, "
    "expressed compactly as p2<super>T</super>&middot;F&middot;p1 = 0, where F "
    "is the fundamental matrix (F = K2<super>-T</super>&middot;E&middot;K1"
    "<super>-1</super>, E = [t]<sub>x</sub>R).", styles["Body"]))

story.append(Paragraph("3.3 Static parameters vs. variables", styles["H2"]))
data_params = [
    ["Static (fixed by one-time calibration)", "Variables (change per point/frame)"],
    ["K1, K2 - intrinsics of each camera (Step 1 calibration)",
     "(u1,v1), (u2,v2) - observed pixel coordinates of P"],
    ["R, t - relative pose between cameras (stereo calibration)",
     "(X,Y,Z) - 3D position of P in camera 1's frame"],
    ["Lens distortion coefficients for each camera",
     "Z2 - depth of P as seen from camera 2 (derived from Z, R, t)"],
]
tparam = Table(data_params, colWidths=[3.0*inch, 3.0*inch])
tparam.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
]))
story.append(tparam)

story.append(Paragraph("3.4 Recovering P without a known Z (triangulation)", styles["H2"]))
story.append(Paragraph(
    "If P is observed in both images but no depth is known ahead of time, P "
    "can instead be recovered by <b>triangulation</b>: each image ray "
    "(camera center through its pixel) is a 3D line, and P is the point "
    "minimizing the sum of squared reprojection distances to both rays "
    "(e.g. via <font face='Courier'>cv2.triangulatePoints</font>), which "
    "requires R and t to already be known from stereo calibration.", styles["Body"]))

story.append(Paragraph("3.5 Assumptions and justification", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph("<b>Rigid relative pose (R,t fixed):</b> valid as long as "
                        "neither camera moves after stereo calibration; otherwise "
                        "R,t must be re-estimated from matched features.", styles["Body"])),
    ListItem(Paragraph("<b>Pinhole model post-undistortion:</b> valid for typical "
                        "consumer cameras after radial/tangential distortion "
                        "correction from individual calibration.", styles["Body"])),
    ListItem(Paragraph("<b>P visible in both FoVs:</b> required for any two-view "
                        "relationship (epipolar constraint or triangulation) to "
                        "apply at all.", styles["Body"])),
]))

story.append(PageBreak())

# ---- Section 4: Experimental Validation ----
story.append(Paragraph("4. Experimental Validation", styles["H1"]))

# --- Step 1 results (from real calibration_result.npz if present) ---
story.append(Paragraph("4.1 Camera Calibration Results (Step 1)", styles["H2"]))
if os.path.exists(CALIBRATION_NPZ):
    data = np.load(CALIBRATION_NPZ, allow_pickle=True)
    K = data["K"]
    dist = data["dist"]
    story.append(Paragraph("Intrinsic matrix K (from your real calibration run):", styles["Body"]))
    k_text = "\n".join("  ".join(f"{v:10.3f}" for v in row) for row in K)
    story.append(Paragraph(k_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["Mono"]))
    dist_text = ", ".join(f"{v:.5f}" for v in dist.ravel())
    story.append(Paragraph(f"Distortion coefficients (k1,k2,p1,p2,k3): {dist_text}", styles["Body"]))
else:
    story.append(Paragraph(
        "&#9888; PLACEHOLDER: no calibration_result.npz found next to this "
        "script. Run camera_calibration.py on your real checkerboard photos, "
        "then place the resulting calibration_result.npz in this folder and "
        "rebuild this report to insert your real K matrix, distortion "
        "coefficients, and reprojection error here.", styles["Warn"]))

# --- Step 3 results (from real results.csv if present) ---
story.append(Paragraph("4.2 20-Trial Validation Statistics (Step 3)", styles["H2"]))
if os.path.exists(VALIDATION_CSV):
    rows = []
    with open(VALIDATION_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    errors = np.array([float(r["error_mm"]) for r in rows])
    abs_errors = np.array([float(r["abs_error_mm"]) for r in rows])
    pct_errors = np.array([float(r["pct_error"]) for r in rows])

    story.append(Paragraph(f"Results from {len(rows)} real object measurements:", styles["Body"]))
    stat_data = [
        ["Statistic", "Value"],
        ["Mean signed error", f"{np.mean(errors):.2f} mm"],
        ["Mean absolute error (MAE)", f"{np.mean(abs_errors):.2f} mm"],
        ["RMSE", f"{np.sqrt(np.mean(errors**2)):.2f} mm"],
        ["Std dev of error", f"{np.std(errors):.2f} mm"],
        ["Mean percentage error", f"{np.mean(pct_errors):.2f} %"],
        ["Min / Max absolute error", f"{np.min(abs_errors):.2f} / {np.max(abs_errors):.2f} mm"],
    ]
    tstat = Table(stat_data, colWidths=[3.0*inch, 2.5*inch])
    tstat.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
    ]))
    story.append(tstat)
    story.append(Spacer(1, 10))

    # per-trial table
    per_trial = [["Trial", "Predicted (mm)", "Ground truth (mm)", "Abs error (mm)", "% error"]]
    for r in rows:
        per_trial.append([
            r["trial"],
            f"{float(r['predicted_mm']):.1f}",
            f"{float(r['ground_truth_mm']):.1f}",
            f"{float(r['abs_error_mm']):.2f}",
            f"{float(r['pct_error']):.2f}",
        ])
    tp = Table(per_trial, colWidths=[0.6*inch, 1.3*inch, 1.4*inch, 1.2*inch, 0.9*inch])
    tp.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
    ]))
    story.append(Paragraph("Per-trial results:", styles["Body"]))
    story.append(tp)
else:
    story.append(Paragraph(
        "&#9888; PLACEHOLDER: no results.csv found next to this script. "
        "Photograph 20 real objects at your fixed distance (&gt;2m), measure "
        "each by hand, fill in trials_template.csv, run "
        "validation_experiment.py to produce results.csv, place it in this "
        "folder, and rebuild this report to insert your real error "
        "statistics and per-trial table here.", styles["Warn"]))

story.append(Paragraph("5. Conclusion", styles["H1"]))
story.append(Paragraph(
    "This report documents the implementation of smartphone camera "
    "calibration, perspective-projection-based real-world dimension "
    "estimation, statistical validation across real object measurements, and "
    "the mathematical relationship between image coordinates of a 3D point as "
    "seen by two cameras. See Section 4 above for real experimental evidence "
    "from the author's own calibration and validation runs.", styles["Body"]))

story.append(Paragraph("6. Demonstration Video", styles["H1"]))
story.append(Paragraph(
    "See the submitted screen recording for a live demonstration of the web "
    "application: running calibration on uploaded checkerboard photos, then "
    "uploading an object photo and clicking two points to obtain a real-world "
    "measurement.", styles["Body"]))

doc = SimpleDocTemplate("Module_2_Report.pdf", pagesize=letter,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.8*inch, rightMargin=0.8*inch)
doc.build(story)
print("PDF built: Module_2_Report.pdf")
if not os.path.exists(CALIBRATION_NPZ):
    print("NOTE: calibration_result.npz not found - Section 4.1 has a placeholder.")
if not os.path.exists(VALIDATION_CSV):
    print("NOTE: results.csv not found - Section 4.2 has a placeholder.")
